import argparse
import numpy as np
import re
import os
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["UNSLOTH_COMPILE_MAXIMUM"] = "0" 

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """
You are a classifier. You are given a pair of statements:

1. An **informal** natural-language mathematical description.
2. A **formal** mathematical translation of that description.

Your task is to output either `"aligned"` or `"misaligned"`.

- Output `"aligned"` if the formal translation faithfully and accurately matches the informal description.
- Output `"misaligned"` if the formal statement does not correspond to the informal description.

Output **only** one word: `aligned` or `misaligned`.
""".strip()


# ============================================================
# DATA PREPROCESSING
# ============================================================
def make_conversations(example, training: bool = False):
    """
    Convert dataset row into chat format.
    Also keep gold label as plain text for generation-based eval.
    """
    user_msg = (
        "Informal:\n"
        f"{example['informal']}\n\n"
        "Formal:\n"
        f"{example['formal_no_comments']}\n\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if training:
        messages.append(
            {"role": "assistant", "content": example["label"]}
        )

    return {
        "messages": messages,
        "label_text": example["label"].strip().lower(),
    }


def formatting_func(tokenizer, example, training: bool = False):
    """
    Apply chat template.
    - During training: include assistant answer
    - During eval: add generation prompt
    """
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=not training,
        enable_thinking=False,
    )


# ============================================================
# METRICS (GENERATION-BASED)
# ============================================================
def normalize_pred(text: str) -> str:
    text = re.split(r"</think>", text)[-1].strip().lower()
    if "misaligned" in text:
        return "misaligned"
    if "aligned" in text:
        return "aligned"
    return "unknown"

def make_compute_metrics(tokenizer, eval_dataset):
    def compute_metrics(eval_preds):
        preds = eval_preds.predictions

        # Decode generated text
        decoded_preds = tokenizer.batch_decode(
            np.where(preds != -100, preds, tokenizer.pad_token_id), skip_special_tokens=True
        )

        gold = eval_dataset["label_text"]

        y_pred = [normalize_pred(p) for p in decoded_preds]
        y_true = [g for g in gold]

        label_map = {"misaligned": 0, "aligned": 1}

        y_pred_i = [label_map.get(p, -1) for p in y_pred]
        y_true_i = [label_map[t] for t in y_true]

        # Drop invalid predictions
        valid = [i for i, p in enumerate(y_pred_i) if p != -1]
        y_pred_i = [y_pred_i[i] for i in valid]
        y_true_i = [y_true_i[i] for i in valid]

        p, r, f1, _ = precision_recall_fscore_support(
            y_true_i, y_pred_i, average="macro"
        )
        acc = accuracy_score(y_true_i, y_pred_i)

        print(
            classification_report(
                y_true_i,
                y_pred_i,
                target_names=["misaligned", "aligned"],
            )
        )

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": p,
            "recall": r,
        }

    return compute_metrics


def tokenize_fn(tokenizer, training: bool):
    def _tok(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    return _tok


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # Model / data
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="offendo/mathlib_v4.18.0_misaligned_by_default",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./alignment_classifier",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load model (Unsloth)
    # ------------------------------------------------------------
    from unsloth import FastLanguageModel

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
    from datasets import load_dataset

    raw_ds = load_dataset(args.dataset_name)

    train_ds = raw_ds["train"].map(
        lambda ex: make_conversations(ex, training=True),
        remove_columns=raw_ds["train"].column_names,
    )

    test_ds = raw_ds["test"].map(
        lambda ex: make_conversations(ex, training=False),
        remove_columns=raw_ds["test"].column_names,
    )

    train_ds = train_ds.map(
        lambda ex: {
            "text": formatting_func(tokenizer, ex, training=True)
        }
    )
    test_ds = test_ds.map(
        lambda ex: {
            "text": formatting_func(tokenizer, ex, training=False)
        }
    )
    train_ds = train_ds.map(
        tokenize_fn(tokenizer, training=True),
        batched=True,
    )
    
    test_ds = test_ds.map(
        tokenize_fn(tokenizer, training=False),
        batched=True,
    )
    

    # ------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)


    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        predict_with_generate=True,
        prediction_loss_only=False,
        remove_unused_columns=True,
        generation_config=GenerationConfig(max_new_tokens=10)
    )

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer, test_ds),
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    trainer.train()
    trainer.save_model(args.output_dir)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

