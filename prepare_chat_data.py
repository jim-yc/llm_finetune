"""
把完整的多輪 conversation，展開成適合 TRL Supervised Fine-Tuning（SFT）
使用的 prompt / completion JSONL。

這支程式位在整個資料處理流程的「第二階段」。

完整流程可以先理解成：

    Hugging Face Dataset
        ↓
    download_chat_data.py
        ↓
    data/chat.jsonl
        ↓
    prepare_chat_data.py   ← 這支程式
        ↓
    data/sft/train.jsonl
    data/sft/valid.jsonl
        ↓
    train_qlora.py
        ↓
    QLoRA 微調

------------------------------------------------------------
這支程式主要做 4 件事
------------------------------------------------------------

1. 讀取完整 conversation。
2. 先用「完整 conversation」切 train / validation。
3. 再把每段 conversation 展開成多筆：
       prompt → completion
4. 輸出成 TRL 可直接讀取的 conversational JSONL。

------------------------------------------------------------
為什麼不能先把每一個 turn 展開，再切 train / validation？
------------------------------------------------------------

假設原始 conversation 是：

    user: 我想點一杯冰拿鐵
    assistant: 好的，一杯大杯冰拿鐵
    user: 幫我加一份濃縮
    assistant: 好的，已幫您加一份濃縮咖啡
    user: 再來一杯熱的
    assistant: 好的，再加一杯大杯熱拿鐵

如果直接先展開，可能得到：

    Example A
    prompt:
        user: 我想點一杯冰拿鐵
    completion:
        assistant: 好的，一杯大杯冰拿鐵

    Example B
    prompt:
        user: 我想點一杯冰拿鐵
        assistant: 好的，一杯大杯冰拿鐵
        user: 幫我加一份濃縮
    completion:
        assistant: 好的，已幫您加一份濃縮咖啡

    Example C
    prompt:
        ...
    completion:
        assistant: 好的，再加一杯大杯熱拿鐵

如果 A 被分到 train，
B 或 C 被分到 validation，
那麼 train 與 validation 其實來自「同一段原始對話」。

這會造成 turn-level leakage：

    train 看過：
        「我想點一杯冰拿鐵」

    validation 又出現：
        「我想點一杯冰拿鐵」
        「好的，一杯大杯冰拿鐵」
        ...

模型會因為已經在 train 看過高度相似的上下文，
導致 validation loss 看起來過度樂觀。

因此正確順序應該是：

    完整 conversation
        ↓
    先切 train / validation
        ↓
    再各自展開

這樣可以確保同一段 conversation 的所有 turns，
只會存在於 train 或 validation 其中一邊。

------------------------------------------------------------
輸出的 TRL conversational 格式
------------------------------------------------------------

每一行會長得像：

    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "我想點一杯冰拿鐵"},
            {"role": "assistant", "content": "好的，一杯大杯冰拿鐵"},
            {"role": "user", "content": "幫我加一份濃縮"}
        ],
        "completion": [
            {"role": "assistant", "content": "好的，已幫您加一份濃縮咖啡"}
        ]
    }

也就是：

    prompt
        =
        模型在回答前可以看到的上下文

    completion
        =
        這一筆訓練資料希望模型學會產生的 assistant 回覆

如果 train_qlora.py 使用：

    completion_only_loss=True

那麼 loss 主要會針對 completion，
也就是 assistant 的回答部分計算。
"""

# =============================================================================
# 1. 匯入 Python 標準函式庫
# =============================================================================

import json
import random
from pathlib import Path

# =============================================================================
# 2. 資料設定
# =============================================================================

INPUT_FILE = Path("data.jsonl")
OUTPUT_DIR = Path("sft_data")
VALID_RATIO = 0.1 # Validation conversation 的比例。
MAX_CONTEXT_MESSAGES = 6 # 每一筆 prompt 最多保留最近多少則「非 system message」。
SEED = 42

# =============================================================================
# 3. 將完整 conversation 展開成 SFT examples
# =============================================================================

def build_examples(conversations):
    """
    將多段完整 conversation，
    展開成多筆「歷史上下文 → 下一個 assistant 回覆」。

    參數
    ----
    conversations:
        Python list。

        每一個元素都是一整段 messages，例如：

        [
            [
                {"role": "user", "content": "我要冰拿鐵"},
                {"role": "assistant", "content": "好的"},
                {"role": "user", "content": "加濃縮"},
                {"role": "assistant", "content": "沒問題"}
            ],

            [
                ...
            ]
        ]

    回傳
    ----
    examples:
        每一筆都是：

        {
            "prompt": [...],
            "completion": [...]
        }

    ------------------------------------------------------------
    展開概念
    ------------------------------------------------------------

    假設一段 conversation：

        user: U1
        assistant: A1
        user: U2
        assistant: A2
        user: U3
        assistant: A3

    會展開成：

        Example 1
        prompt:
            U1
        completion:
            A1

        Example 2
        prompt:
            U1
            A1
            U2
        completion:
            A2

        Example 3
        prompt:
            U1
            A1
            U2
            A2
            U3
        completion:
            A3

    也就是：
    每一個 assistant message，
    都可以成為一個 supervision target。
    """

    examples = []
    for messages in conversations:
        prompt = []
        for message in messages:
            if message["role"] != "assistant":
                prompt.append(message)
            
            else:
                if len(prompt) > MAX_CONTEXT_MESSAGES:
                    prompt = prompt[-MAX_CONTEXT_MESSAGES:]
                
                example = {"prompt": prompt.copy(), "completion": [message]}
                examples.append(example)
                prompt.append(message)

    return examples

# =============================================================================
# 4. 將 Python dict list 寫成 JSONL
# =============================================================================

def save_jsonl(rows, path):
    """
    將多筆 Python dict 寫成 UTF-8 JSONL。

    例如 rows：

        [
            {"prompt": [...], "completion": [...]},
            {"prompt": [...], "completion": [...]},
        ]

    會輸出：

        {...}
        {...}

    一行一筆 JSON。
    """

    with path.open("w", encoding="utf-8") as file:
        for row in rows: # 逐筆寫入。
            json_line = json.dumps(row, ensure_ascii=False) # ensure_ascii=False 讓繁體中文直接保留
            file.write(json_line + "\n")

# =============================================================================
# 5. 主流程
# =============================================================================

def main():
    """
    執行完整資料準備流程：

    1. 讀取 data.jsonl
    2. 建立完整 conversation list
    3. 以固定 seed shuffle
    4. 先切 train / validation conversation
    5. 各自展開成 prompt / completion
    6. 建立 data/sft
    7. 寫入 train.jsonl / valid.jsonl
    8. 顯示統計資訊
    """

    # -------------------------------------------------------------------------
    # 確認輸入檔存在
    # -------------------------------------------------------------------------

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{INPUT_FILE}")

    # -------------------------------------------------------------------------
    # 建立 conversation 清單
    # -------------------------------------------------------------------------

    conversations = []

    # -------------------------------------------------------------------------
    # 讀取完整 conversation JSONL
    # -------------------------------------------------------------------------

    with INPUT_FILE.open("r", encoding="utf-8") as file:

        # line_number 從 1 開始，
        # 若 JSON 格式壞掉時比較容易定位是哪一行。
        for line_number, line in enumerate(file, start=1):

            # strip() 去掉前後空白與換行。
            # 如果整行只有空白，就不需要處理。
            if not line.strip():
                continue

            try:
                line = line.rstrip().rstrip(",")
                row = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(f"{INPUT_FILE} 第 {line_number} 行不是合法 JSON：{error}") from error

            conversations.append(row["conversations"])

    # -------------------------------------------------------------------------
    # 確認資料不是空的
    # -------------------------------------------------------------------------

    if not conversations:
        raise RuntimeError(f"{INPUT_FILE} 沒有任何 conversation。")

    # -------------------------------------------------------------------------
    # 使用固定 seed 打亂 conversation
    # -------------------------------------------------------------------------
    
    rng = random.Random(SEED)
    rng.shuffle(conversations) # 原地打亂 conversations。

    # -------------------------------------------------------------------------
    # 計算 validation conversation 數量
    # -------------------------------------------------------------------------

    # validation 至少保留 1 段 conversation。
    valid_size = max(1, int(len(conversations) * VALID_RATIO))

    # -------------------------------------------------------------------------
    # 把 conversation 切分
    # -------------------------------------------------------------------------

    train_conversations = conversations[valid_size:]
    valid_conversations = conversations[:valid_size]

    # -------------------------------------------------------------------------
    # 展開 conversations
    # -------------------------------------------------------------------------
    
    train_rows = build_examples(train_conversations)
    valid_rows = build_examples(valid_conversations)

    # -------------------------------------------------------------------------
    # 建立輸出資料夾
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 儲存 data
    # -------------------------------------------------------------------------

    train_path = OUTPUT_DIR / "train.jsonl"
    save_jsonl(train_rows, train_path)

    valid_path = OUTPUT_DIR / "valid.jsonl"
    save_jsonl(valid_rows, valid_path)

    # -------------------------------------------------------------------------
    # 顯示統計資訊
    # -------------------------------------------------------------------------

    print("=" * 72)

    print(
        f"總 conversation 數："
        f"{len(conversations)}"
    )

    print(
        f"Train conversations："
        f"{len(train_conversations)}"
    )

    print(
        f"Validation conversations："
        f"{len(valid_conversations)}"
    )

    print(
        f"Train SFT examples："
        f"{len(train_rows)}"
    )

    print(
        f"Validation SFT examples："
        f"{len(valid_rows)}"
    )

    print(
        f"Train JSONL："
        f"{train_path}"
    )

    print(
        f"Validation JSONL："
        f"{valid_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
