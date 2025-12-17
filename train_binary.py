import argparse
import random
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

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
        "Informal:\n"
        f"{example['informal']}\n\n"
        "Formal:\n"
        f"{example['formal_no_comments']}\n\n"
    )

    # Convert label to textual answer

    msgs =  {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    }
    if training:
        msgs['messages'].append({"role": "assistant", "content": example['label']})
    return msgs


def formatting_func(tokenizer, example, training: bool = False):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=not training,
        enable_thinking=True,
    )

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        """
        Computes exact-match accuracy for generated text.
        Expected labels: 'aligned' or 'misaligned'
        """
        preds, labels = eval_preds

        # Replace -100 in the preds as we can't decode them
        preds = np.where(labels != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # If predictions are token IDs (common with generate)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Normalize
        preds = [int(p.strip().lower().split()[-1] == 'aligned') for p in preds]
        labels = [int(l.strip().lower().split()[-1] == 'aligned') for l in labels]
        p,r,f1,_ = precision_recall_fscore_support(labels, preds, average='macro')
        accuracy = accuracy_score(labels, preds)

        print(classification_report(labels, preds, target_names=['misaligned', 'aligned']))

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": p,
            "recall": r,
        }

    return compute_metrics

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

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
    parser.add_argument("--max_steps", type=int, default=5000)
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
    raw_ds = load_dataset(args.dataset_name)
    train_ds = raw_ds['train']
    test_ds = raw_ds['test']

    train_ds = train_ds.map(lambda ex: make_conversations(ex, training=True))
    test_ds = test_ds.map(lambda ex: make_conversations(ex, training=False))

    train_ds = train_ds.map(lambda batch: {'text': formatting_func(tokenizer, batch, training=True)}, batched=True)
    test_ds = test_ds.map(lambda batch: {'text': formatting_func(tokenizer, batch, training=False)}, batched=True)


    print('Example:\n===========================================')
    print(test_ds[0]['text'])
    print('===========================================')

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
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        gradient_accumulation_steps=1,
        report_to="none",
        assistant_only_loss=True,
        eos_token=tokenizer.eos_token,
        dataset_text_field="text",
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=config,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    trainer.train()
    trainer.save_model(args.output_dir)



if __name__ == "__main__":
    from unsloth import FastLanguageModel
    from datasets import load_dataset, DatasetDict, Dataset
    from trl import SFTTrainer, SFTConfig

    main()
