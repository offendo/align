# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """
You are a classifier. You are given a pair of statements:

1. An **informal** natural-language mathematical description.
2. A **formal** mathematical translation of that description.

Your task is to output **only one token**: either `"correct"` or `"incorrect"`.

- Output `"correct"` if the formal translation faithfully and accurately matches the informal description.
- Output `"incorrect"` if the formal statement does not correspond to the informal description.

Do NOT explain your answer. Output strictly `"correct"` or `"incorrect"`.
""".strip()


# ============================================================
# DATA PREPROCESSING → Conversation format
# ============================================================
def make_conversations(example, training: bool = False):
    """
    Convert dataset row into conversation format expected by TRL.
    """

    # Build user message
    user_msg = (
        "Formal statement:\n"
        f"{example['formal']}\n\n"
        "Informal explanation:\n"
        f"{example['informal']}\n\n"
        "Are they aligned? Respond with 'correct' or 'incorrect'."
    )

    # Convert label to textual answer

    msgs =  {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    }
    if training:
        label = "correct" if example["label"] == 1 else "incorrect"
        msgs['messages'].append({"role": "assistant", "content": label})
    return msgs


# ============================================================
# GROUP–WISE TRAIN/VAL SPLIT (NO ID LEAKAGE)
# ============================================================
def grouped_train_test_split(ds, val_ratio=0.02, seed=42):
    """
    Ensures that all items with the same ID stay in the same split.
    """

    # Group rows by ID
    grouped = defaultdict(list)
    for i, row in enumerate(ds):
        grouped[row["id"]].append(i)

    all_ids = list(grouped.keys())
    random.Random(seed).shuffle(all_ids)

    n_val = int(len(all_ids) * val_ratio)
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])

    train_indices, val_indices = [], []

    for id_, idxs in grouped.items():
        if id_ in train_ids:
            train_indices.extend(idxs)
        else:
            val_indices.extend(idxs)

    return (
        ds.select(train_indices),
        ds.select(val_indices)
    )

def formatting_func(tokenizer, example, training: bool = False):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=not training
    )


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # Basic configs
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_name", type=str, default="offendo/mathlib_v4.18.0_misaligned_by_default")
    parser.add_argument("--output_dir", type=str, default="./alignment_classifier")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load model with Unsloth
    # ------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name,
        load_in_4bit=False,
        load_in_8bit=True,
        full_finetuning=True,
        max_seq_length=4096,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    raw_ds = load_dataset(args.dataset_name, split='train')
    processed_ds = raw_ds.map(lambda ex: make_conversations(ex, training=True))
    processed_ds = processed_ds.map(lambda batch: {'text': formatting_func(tokenizer, batch, training=True)}, batched=True)

    # Split without ID leakage
    train_ds, val_ds = grouped_train_test_split(processed_ds)

    # ------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------
    config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=5000,
        eval_strategy="steps",
        eval_steps=5000,
        logging_steps=10,
        gradient_accumulation_steps=1,
        report_to="none",
        assistant_only_loss=True,
        eos_token=tokenizer.eos_token,
        dataset_text_field="text",
    )

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=config,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    trainer.train()

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    from unsloth import FastLanguageModel
    import argparse
    from datasets import load_dataset, DatasetDict
    from trl import SFTTrainer, SFTConfig
    import random
    from collections import defaultdict

    main()
