import argparse
import random
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """
You are a classifier. You are given a pair of statements:

1. An **informal** natural-language mathematical description.
2. A **formal** mathematical translation of that description.

Your task is to output **only one token**: either `"aligned"` or `"distinct"`.

- Output `"aligned"` if the formal translation faithfully and accurately matches the informal description.
- Output `"distinct"` if the formal statement does not correspond to the informal description.

Do NOT explain your answer. Output strictly `"aligned"` or `"distinct"`.
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
        "Are they aligned? Respond with 'aligned' if they are aligned, or 'distinct' if not."
    )

    # Convert label to textual answer

    msgs =  {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    }
    if training:
        label = "aligned" if example["label"] else "distinct"
        msgs['messages'].append({"role": "assistant", "content": label})
    return msgs


def split_by_module(df, module_col="modules", test_size=0.2, random_state=None):
    """
    Split a dataframe into train and test such that all rows with the same
    module value appear entirely in either split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    module_col : str
        Column whose unique values define the grouping.
    test_size : float
        Fraction of modules to assign to the test split.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    """
    # Unique modules
    modules = df[module_col].unique()

    # Split modules into train/test
    train_modules, test_modules = train_test_split(
        modules,
        test_size=test_size,
        random_state=random_state
    )

    # Select rows belonging to those module sets
    df_train = df[df[module_col].isin(train_modules)].copy()
    df_test = df[df[module_col].isin(test_modules)].copy()

    return df_train, df_test

def split_by_module_hf(dataset, module_col="modules", test_size=0.2, seed=None):
    """
    Split a HuggingFace dataset into train and test such that all rows with the
    same module value appear entirely in either split.

    Parameters
    ----------
    dataset : datasets.Dataset
        The input dataset.
    module_col : str
        Column whose unique values define grouping.
    test_size : float
        Fraction of modules to put in the test split.
    seed : int or None
        Optional RNG seed for reproducibility.

    Returns
    -------
    train_ds : datasets.Dataset
    test_ds : datasets.Dataset
    """

    # Get unique module labels
    modules = list(set(dataset[module_col]))

    # Shuffle modules reproducibly
    rng = random.Random(seed)
    rng.shuffle(modules)

    # Compute split point
    n_test = int(len(modules) * test_size)
    test_modules = set(modules[:n_test])
    train_modules = set(modules[n_test:])

    # Filter entire dataset based on module membership
    train_ds = dataset.filter(lambda x: x[module_col] in train_modules)
    test_ds = dataset.filter(lambda x: x[module_col] in test_modules)

    return train_ds, test_ds


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

    # Split without Module leakage
    train_ds, val_ds = split_by_module(processed_ds)

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
    from datasets import load_dataset, DatasetDict, Dataset
    from trl import SFTTrainer, SFTConfig

    main()
