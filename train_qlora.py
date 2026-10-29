"""
使用 4-bit QLoRA 微調繁體中文多輪對話模型。
我這裡用 Intel ARC GPU 訓練，所以可直接用 BF16。

整體流程可以理解成：

    JSONL 對話資料
        ↓
    Tokenizer 把文字轉成 token IDs
        ↓
    載入 Qwen2.5-0.5B-Instruct
        ↓
    Base model 權重以 4-bit NF4 儲存
        ↓
    prepare_model_for_kbit_training()
        ↓
    加上 LoRA adapter
        ↓
    只有少量 LoRA 參數參與訓練
        ↓
    SFTTrainer 執行 forward / loss / backward / optimizer step
        ↓
    定期 evaluation
        ↓
    儲存 checkpoint
        ↓
    early stopping
        ↓
    載入 validation loss 最低的最佳 checkpoint
        ↓
    儲存最後的 LoRA adapter 與 tokenizer

這裡的「QLoRA」可以簡單理解成：

    Q = Quantization
        Base model 使用低精度 4-bit 表示，降低 GPU 記憶體需求。

    LoRA = Low-Rank Adaptation
        不直接修改完整 base model，而是在部分線性層旁邊加入少量可訓練參數。

例如原始模型可能有約 5 億個參數，但真正需要更新的 LoRA 參數可能只有幾百萬個。
這可以大幅降低微調所需的 GPU 記憶體。
"""

from pathlib import Path

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

# =============================================================================
# 1. 模型與路徑設定
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = "sft_data"
TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"
OUTPUT_DIR = "outputs/qwen25_05b_lora"
RESUME_FROM_CHECKPOINT = None
TRUST_REMOTE_CODE = False

# =============================================================================
# 2. 訓練設定
# =============================================================================

MAX_LENGTH = 768
EPOCHS = 100
LEARNING_RATE = 1e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 16
GRADIENT_CHECKPOINTING = False
OPTIM = "paged_adamw_8bit"
LOGGING_STEPS = 5
CHECK_STRATEGY = "steps"
CHECK_STEPS = 25 # 每隔多少 optimizer steps 做一次 validation 存一次 checkpoint
CHECKPOINT_LIMIT = 3
EARLY_STOPPING_PATIENCE = 5
COMPLETION_ONLY_LOSS = True # 對 prompt-completion / chat 類型資料，只針對 assistant 要回答的部分計算 loss。
EOS_TOKEN = "<|im_end|>" # Qwen chat template 使用的對話結束 token。
REPORT_TO = "none" # "none" 表示這個程式不連接外部 tracking service。
SEED = 42

# =============================================================================
# 3. LoRA 設定
# =============================================================================

LORA_R = 8 # LoRA rank，通常寫成 r。
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear" # 對模型中的各種 linear layers 套用 LoRA。

# =============================================================================
# 4. 輔助函式
# =============================================================================

def resolve_path(path_value: str) -> Path:
    """
    將相對路徑轉成「相對於 train_qlora.py」的實際路徑。

    為什麼不直接使用 Path("data/sft")？

    因為使用者可能從不同工作目錄執行：

        cd /content
        python llm_finetune/train_qlora.py

    如果所有相對路徑都依賴目前 shell 的 cwd，
    很容易找不到資料。

    這個函式固定以 train_qlora.py 所在位置為基準。
    """

    script_dir = Path(__file__).resolve().parent
    path = Path(path_value)

    if path.is_absolute():
        return path

    return script_dir / path


def get_gpu_info() -> None:
    """
    檢查 GPU，並列印目前硬體與 precision 資訊。
    """

    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    bf16 = False

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"偵測到 Intel GPU (XPU): {torch.xpu.get_device_name(0)}")
        print(f"是否支援 BF16: {torch.xpu.is_bf16_supported()}")
        bf16 = torch.xpu.is_bf16_supported()

    elif torch.cuda.is_available():
        print(f"偵測到 NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")
        print(f"是否支援 BF16: {torch.cuda.is_bf16_supported()}")
        bf16 = torch.xpu.is_bf16_supported()

    else:
        print(f"沒偵測到任何 GPU")

    return bf16

# =============================================================================
# 5. 主訓練流程
# =============================================================================

def main() -> None:
    """
    執行完整 QLoRA Supervised Fine-Tuning。

    流程：

    1. 檢查 GPU
    2. 檢查資料
    3. 載入 dataset
    4. 載入 tokenizer
    5. 建立 4-bit quantization config
    6. 載入 base model
    7. 準備 k-bit training
    8. 建立 LoRA adapter
    9. 建立 SFTConfig
    10. 建立 SFTTrainer
    11. train
    12. 儲存最佳模型與 tokenizer
    """

    # -------------------------------------------------------------------------
    # 確認 GPU 型號和是否可用 BF16 訓練
    # -------------------------------------------------------------------------

    is_bf16 = get_gpu_info()
    train_dtype = torch.bfloat16 if is_bf16 else torch.float32

    # -------------------------------------------------------------------------
    # 解析資料與輸出路徑
    # -------------------------------------------------------------------------

    data_dir = resolve_path(DATA_DIR)
    train_file = data_dir / TRAIN_FILE
    valid_file = data_dir / VALID_FILE
    output_dir = resolve_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 確認 training / validation files 存在
    # -------------------------------------------------------------------------

    if not train_file.is_file():
        raise FileNotFoundError(f"找不到 train dataset：{train_file}")

    if not valid_file.is_file():
        raise FileNotFoundError(f"找不到 validation dataset：{valid_file}")

    # -------------------------------------------------------------------------
    # 載入 dataset
    # -------------------------------------------------------------------------

    data = load_dataset("json", data_files={"train": str(train_file), "validation": str(valid_file)})

    # -------------------------------------------------------------------------
    # 載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)

    # 如果 tokenizer 沒有專門 pad token，就沿用 eos token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # 建立 4-bit QLoRA quantization 設定
    # -------------------------------------------------------------------------

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, # Base model 權重以 4-bit 載入。
        bnb_4bit_quant_type="nf4", # 4-bit quantization format，NF4 = NormalFloat4。
        bnb_4bit_use_double_quant=True, # 不只量化模型權重，也進一步量化部分 quantization constants，以再降低一些記憶體。
        bnb_4bit_compute_dtype=train_dtype, # 訓練用格式
    )

    # -------------------------------------------------------------------------
    # 載入 4-bit base model
    # -------------------------------------------------------------------------

    # 這裡不直接把 MODEL_ID 字串交給 SFTTrainer，
    # 而是自行先載入模型。
    #
    # 好處：可以非常清楚控制：
    # - quantization
    # - dtype
    # - device placement
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config, # 套用上面定義的 4-bit 設定。
            dtype=train_dtype, # 非量化部分資料格式
            device_map={"": 0},
            trust_remote_code=TRUST_REMOTE_CODE,
        )

    # -------------------------------------------------------------------------
    # 關閉 KV cache
    # -------------------------------------------------------------------------

    # KV cache 主要是「推論」時加速 autoregressive generation。
    #
    # 訓練時使用 gradient checkpointing，
    # use_cache=True 可能造成衝突或額外記憶體消耗。
    #
    # 所以 training 時關掉。
    model.config.use_cache = False

    # -------------------------------------------------------------------------
    # 準備 k-bit model 進行訓練
    # -------------------------------------------------------------------------

    # prepare_model_for_kbit_training()
    # 是 PEFT 在量化模型微調中的標準步驟。
    #
    # 它會做一些必要處理，例如：
    # - 凍結 base model parameters
    # - 處理 layer normalization precision
    # - 配合 gradient checkpointing
    #
    # 目的是讓：
    #
    # quantized base model
    #
    # 可以安全地搭配 LoRA adapter 訓練。
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)

    # -------------------------------------------------------------------------
    # 建立 LoRA 設定
    # -------------------------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, # 對所有 linear layers 套用 LoRA。
        task_type="CAUSAL_LM", # 這是 causal language model。
    )

    # -------------------------------------------------------------------------
    # 把 LoRA adapter 加到 base model
    # -------------------------------------------------------------------------

    # get_peft_model() 之後，
    # model 仍然包含原本的 base model，
    # 但多出少量 LoRA adapter parameters。
    #
    # Base model：
    # requires_grad=False
    #
    # LoRA：
    # requires_grad=True
    model = get_peft_model(model, lora_config)

    # -------------------------------------------------------------------------
    # 確保 LoRA trainable parameters 使用 XPU 設備的 BF16
    # -------------------------------------------------------------------------
    
    for param in model.parameters():
        if (param.requires_grad and param.dtype != train_dtype):
            param.data = param.data.to(train_dtype)

    # -------------------------------------------------------------------------
    # 顯示 trainable parameter 數量
    # -------------------------------------------------------------------------

    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # 建立 SFT 訓練設定
    # -------------------------------------------------------------------------

    args = SFTConfig(
        output_dir=str(output_dir), # checkpoint 與最後輸出的儲存位置。
        max_length=MAX_LENGTH, # 每筆序列最大 token 長度。
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        fp16=False,
        bf16=True, # Mixed Precision，XPU 可用 BF16
        tf32=False,
        optim=OPTIM, # 使用較省記憶體的 optimizer。
        logging_steps=LOGGING_STEPS,
        eval_strategy=CHECK_STRATEGY,
        eval_steps=CHECK_STEPS,
        save_strategy=CHECK_STRATEGY,
        save_steps=CHECK_STEPS,
        save_total_limit=CHECKPOINT_LIMIT,
        # 訓練結束後，自動把 validation 指標最好的 checkpoint 載回記憶體。
        load_best_model_at_end=True,
        # 用 eval_loss 判斷最佳模型。
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # eval_loss 越低越好。
        # 只對 assistant completion 計算 loss。
        completion_only_loss=COMPLETION_ONLY_LOSS,
        eos_token=EOS_TOKEN, # 指定 Qwen chat EOS token。
        report_to=REPORT_TO, # 不使用 wandb 等外部 tracker。

        # 從 checkpoint resume 時，
        # 一併恢復 callback state。
        #
        # 對 early stopping 特別重要，
        # 不然 patience counter 可能重新開始計算。
        restore_callback_states_from_checkpoint=True,
        seed=SEED,  # 固定 random seed。
    )

    # -------------------------------------------------------------------------
    # 建立 SFTTrainer
    # -------------------------------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        args=args,

        # 真正拿來計算 gradient 的 training split。
        train_dataset=data["train"],

        # 只用來做 evaluation 的 validation split。
        eval_dataset=data["validation"],

        # tokenizer / processor。
        processing_class=tokenizer,

        # EarlyStoppingCallback：
        #
        # 如果 eval_loss 連續多次不再改善，
        # 自動結束 training。
        callbacks=[EarlyStoppingCallback(early_stopping_patience=(EARLY_STOPPING_PATIENCE))],

        # 注意：
        #
        # 這裡「故意不再」傳：
        #
        # quantization_config=...
        # peft_config=...
        #
        # 因為前面已經手動完成：
        #
        # quantization + LoRA
        #
        # 如果再傳一次，
        # 可能導致重複處理。
    )

    # -------------------------------------------------------------------------
    # 開始訓練
    # -------------------------------------------------------------------------

    resume_from_checkpoint = RESUME_FROM_CHECKPOINT
    
    # 如果使用字串指定 checkpoint，
    # 將它解析成相對於 train_qlora.py 的完整路徑。
    if (isinstance(resume_from_checkpoint, str) and resume_from_checkpoint):
        resume_from_checkpoint = str(resolve_path(resume_from_checkpoint))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # -------------------------------------------------------------------------
    # 顯示最佳 checkpoint
    # -------------------------------------------------------------------------

    # 例如：
    # outputs/qwen25_05b_lora/checkpoint-250
    print(f"最佳 checkpoint：{trainer.state.best_model_checkpoint}")

    # 例如：
    # 最佳 eval_loss：0.8421
    print(f"最佳 eval_loss：{trainer.state.best_metric}")

    # -------------------------------------------------------------------------
    # 儲存最佳模型及相關資料
    # -------------------------------------------------------------------------

    # 因為：
    # load_best_model_at_end=True
    # 所以此時 trainer.model 已經重新載入
    # validation 表現最好的 checkpoint。
    #
    # QLoRA / PEFT 情況下，
    # save_model() 主要會儲存 LoRA adapter，
    # 而不是重新複製完整 base model。
    trainer.save_model(str(output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(str(output_dir))

# =============================================================================
# 6. Python 程式進入點
# =============================================================================

if __name__ == "__main__":
    main()
