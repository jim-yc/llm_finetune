"""
載入 train_qlora.py 儲存的最佳 LoRA adapter，進行「多輪互動式對話」。

如果要測試特定 checkpoint，也可以把 ADAPTER_DIR 改成：

    outputs/qwen25_05b_lora/checkpoint-200
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import torch
from peft import AutoPeftModelForCausalLM # 直接載入 Base Model + LoRA Adapter
from transformers import AutoTokenizer

# =============================================================================
# 2. 推論設定
# =============================================================================

ADAPTER_DIR = "outputs/qwen25_05b_lora"
SYSTEM_PROMPT = "你是一位頂尖的情話大師，請用充滿愛意、溫柔體貼的口吻，巧妙地回應使用者的每句話。"
MAX_NEW_TOKENS = 256 # 每次 assistant 最多可以新產生多少 tokens
TEMPERATURE = 0.7
TOP_P = 0.9 # 保留累積機率約 90% 的候選 token

# 是否啟用 sampling。
#
# True：
#     使用 temperature / top_p，
#     每次回答可能稍有不同。
#
# False：
#     通常接近 greedy decoding，
#     回答比較固定。
#
DO_SAMPLE = True
MAX_HISTORY_TURNS = 10 # 最多保留多少輪 user-assistant 對話

# =============================================================================
# 3. 輔助函式：裁切過長的對話歷史
# =============================================================================

def trim_history(messages):
    """
    保留 system prompt，以及最近 MAX_HISTORY_TURNS 輪對話。

    messages 的格式例如：

        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "我要冰拿鐵"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "再加濃縮"},
            {"role": "assistant", "content": "..."},
        ]

    第一筆 system message 永遠保留。

    後面的 user / assistant message，
    最多保留：

        MAX_HISTORY_TURNS × 2

    筆。

    例如 MAX_HISTORY_TURNS = 10：

        10 個 user
        +
        10 個 assistant
        =
        最多 20 筆歷史 message
    """

    # 如果只有 system prompt，
    # 或歷史還沒有超過限制，就直接回傳。
    if len(messages) <= 1 + MAX_HISTORY_TURNS * 2:
        return messages

    # 第一筆是 system prompt。
    system_message = messages[0]

    # 從最後面取最近 N 輪。
    recent_messages = messages[-MAX_HISTORY_TURNS * 2:]

    # 重新組合
    return [system_message] + recent_messages


# =============================================================================
# 4. 輔助函式：產生單次 assistant 回覆
# =============================================================================

def generate_reply(model, tokenizer, messages):
    """
    把目前完整對話歷史送進模型，
    並回傳「這一輪 assistant 新產生的文字」。

    例如目前 messages：

        system
        user: 我想點冰拿鐵
        assistant: 好的
        user: 幫我加一份濃縮

    這個函式會把全部 context 送進模型，
    讓模型知道「加濃縮」是針對前面的冰拿鐵。
    """

    # -------------------------------------------------------------------------
    # Step 1：套用 Qwen 原生 chat template
    # -------------------------------------------------------------------------

    # apply_chat_template() 會把：
    #
    # [
    #   {"role": "system", ...},
    #   {"role": "user", ...}
    # ]
    #
    # 轉成 Qwen 真正訓練時使用的特殊格式。
    #
    # 概念上可能類似：
    #
    # <|im_start|>system
    # ...
    # <|im_end|>
    # <|im_start|>user
    # ...
    # <|im_end|>
    # <|im_start|>assistant
    #
    # 不應該自己手動拼接這些特殊 token，
    # 因為 tokenizer 已經知道正確格式。
    inputs = tokenizer.apply_chat_template(
        messages,

        # 在最後補上 assistant 開始回答的 generation prompt。
        add_generation_prompt=True,

        # 直接做 tokenization。
        tokenize=True,

        # 回傳 dict，例如：
        #
        # {
        #     "input_ids": ...,
        #     "attention_mask": ...
        # }
        return_dict=True,

        # 回傳 PyTorch tensor。
        return_tensors="pt",
    )

    # -------------------------------------------------------------------------
    # Step 2：把 input tensors 放到模型所在 GPU
    # -------------------------------------------------------------------------

    # AutoPeftModel 使用 device_map="auto" 後，
    # 這個 0.5B 模型在單張 T4 上通常整個都會放到 cuda:0。
    #
    # model.device 會告訴我們主要模型所在裝置。
    inputs = inputs.to(model.device)

    # 記錄 prompt 本身用了多少 token。
    #
    # 假設：
    #
    # input_length = 150
    #
    # 模型最後輸出：
    #
    # [原本 150 tokens] + [新產生 30 tokens]
    #
    # 我們等一下只取後面那 30 個。
    input_length = inputs["input_ids"].shape[-1]

    # -------------------------------------------------------------------------
    # Step 3：執行推論
    # -------------------------------------------------------------------------

    with torch.inference_mode():

        output = model.generate(
            **inputs,

            # 最多產生多少新 token。
            max_new_tokens=MAX_NEW_TOKENS,

            # 是否使用 sampling。
            do_sample=DO_SAMPLE,

            # sampling temperature。
            temperature=TEMPERATURE,

            # nucleus sampling。
            top_p=TOP_P,

            # 如果 tokenizer 有 pad token，
            # 明確告訴 generate() 使用哪一個。
            pad_token_id=tokenizer.pad_token_id,

            # EOS token：
            # 模型產生這個 token 時可以停止 generation。
            eos_token_id=tokenizer.eos_token_id,
        )

    # -------------------------------------------------------------------------
    # Step 4：只取新生成的 assistant tokens
    # -------------------------------------------------------------------------

    # generate() 回傳的 sequence 通常包含：
    #
    # 原始 prompt
    # +
    # 新產生內容
    #
    # 所以：
    #
    # output[0][input_length:]
    #
    # 表示只留下模型新生成的部分。
    new_tokens = output[0][input_length:]

    # -------------------------------------------------------------------------
    # Step 5：token IDs → 中文文字
    # -------------------------------------------------------------------------

    reply = tokenizer.decode(
        new_tokens,

        # 不把 <|im_end|> 等特殊 token 顯示給使用者。
        skip_special_tokens=True,
    ).strip()

    return reply


# =============================================================================
# 5. 主程式
# =============================================================================

def main():
    """
    啟動多輪互動式聊天。

    可用指令：exit, quit, q 結束程式。

        /reset
            清除目前對話歷史，
            但保留 system prompt。
        /history
            顯示目前保存的完整 conversation history。
        /help
            顯示可用指令。
    """

    # -------------------------------------------------------------------------
    # Step 1：載入 tokenizer
    # -------------------------------------------------------------------------

    print("正在載入 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    # 某些模型沒有 pad token。
    #
    # generation 時通常可以讓 pad_token = eos_token。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Step 2：載入 base model + LoRA adapter
    # -------------------------------------------------------------------------

    print("正在載入 Base Model + LoRA Adapter...")

    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        dtype=torch.bfloat16,
        device_map="xpu",
    )
    model.eval()

    # -------------------------------------------------------------------------
    # Step 3：初始化 conversation history
    # -------------------------------------------------------------------------

    # 一開始只有 system prompt。
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # -------------------------------------------------------------------------
    # Step 4：顯示操作說明
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("LoRA 多輪對話模式已啟動")
    print("=" * 72)
    print("輸入訊息後按 Enter 即可與模型對話。")
    print()
    print("可用指令：")
    print("  /reset    清除目前對話歷史")
    print("  /history  顯示目前對話歷史")
    print("  /help     顯示指令說明")
    print("  exit      結束程式")
    print("  quit      結束程式")
    print("  q         結束程式")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Step 5：進入多輪對話迴圈
    # -------------------------------------------------------------------------

    while True: 
        user_text = input("\n你：").strip() # 去掉字串前後空白

        if not user_text: # 空字串不送給模型
            continue

        # -------------------------------------------------------------
        # 處理非對話邏輯
        # -------------------------------------------------------------

        if user_text.lower() in {"exit", "quit", "q"}:
            print("\n對話結束。")
            break

        if user_text == "/help":
            print()
            print("可用指令：")
            print("  /reset    清除對話歷史")
            print("  /history  顯示目前對話歷史")
            print("  /help     顯示這份說明")
            print("  exit      結束程式")
            print("  quit      結束程式")
            print("  q         結束程式")
            continue

        if user_text == "/reset":
            # 清除 user / assistant history，只留下 system prompt。
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]

            print("\n系統：已清除對話歷史，開始新的對話。")
            continue

        if user_text == "/history":

            print("\n目前對話歷史：")
            print("-" * 72)

            for index, message in enumerate(messages, start=1):
                print(
                    f"[{index}] "
                    f"{message['role']}: "
                    f"{message['content']}"
                )

            print("-" * 72)
            continue


        # -------------------------------------------------------------
        # Step 6：把新的 user message 加進 conversation history
        # -------------------------------------------------------------

        messages.append({"role": "user", "content": user_text})

        # -------------------------------------------------------------
        # Step 7：限制 history 長度
        # -------------------------------------------------------------

        messages = trim_history(messages)


        # -------------------------------------------------------------
        # Step 8：讓模型產生回答
        # -------------------------------------------------------------

        try:
            reply = generate_reply(model, tokenizer, messages)

        except torch.cuda.OutOfMemoryError:
            # 如果真的遇到 GPU OOM，
            # 清掉 PyTorch CUDA allocator 中沒有使用的 cached blocks。
            torch.cuda.empty_cache()

            print(
                "\n系統：GPU 記憶體不足。"
                "可以先使用 /reset 清除較長的對話歷史，"
                "或降低 MAX_HISTORY_TURNS。"
            )

            # 因為這一輪沒有成功得到 assistant 回答，
            # 把剛才新增的 user message 移除，
            # 避免 history 出現 user message 卻沒有 assistant response。
            if (
                messages
                and messages[-1]["role"] == "user"
            ):
                messages.pop()

            continue


        # -------------------------------------------------------------
        # Step 9：把 assistant 回答加入 history
        # -------------------------------------------------------------

        messages.append({"role": "assistant", "content": reply})

        # -------------------------------------------------------------
        # Step 10：顯示回答
        # -------------------------------------------------------------

        print(f"\n助理：{reply}")


# =============================================================================
# 6. Python 程式進入點
# =============================================================================

if __name__ == "__main__":
    main()
