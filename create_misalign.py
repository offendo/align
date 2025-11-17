"""
Create misaligned versions of responses from a HuggingFace dataset.

This script:
- Loads a HuggingFace dataset
- Applies modification strategies to each sample
- Returns a combined dataset: original + modified

Only HF datasets are supported now.

Example:
    python create_misalign.py --dataset_name my_org/my_dataset --output_path out --seed 42
"""

import argparse
import os
import random
import re
import string

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

# ----------------------------
# Modification Functions
# ----------------------------


def modify_constant(expression):
    parts = expression.split(":", 1)
    if len(parts) < 2:
        print("Invalid format (expected 'theorem ... : ...'): ```", expression, "```")
        return expression

    prefix, body = parts
    constants = re.findall(r"(?<!\\u)\b\d+\b", body)

    if constants:
        chosen = random.choice(constants)
        new_val = str(int(chosen) + random.randint(1, 100))
        body = re.sub(r"(?<!\\u)\b" + re.escape(chosen) + r"\b", new_val, body, count=1)

    return prefix + ":" + body


def modify_exponent(expression):
    pattern = r"(\s*-?\s*\w+\s*|\s*\(-?\s*\d+\s*\))\s*\^\s*(\s*-?\s*\d+\.?\d*\s*|\s*\w+\s*|\s*\(.+?\)\s*)"
    matches = list(re.finditer(pattern, expression))

    if not matches:
        return expression

    old_expr = random.choice(matches).group(0)
    base, old_exp = re.match(pattern, old_expr).groups()

    base = base.strip()
    old_exp = old_exp.strip().strip("(").strip(")")

    try:
        orig = float(old_exp)
        delta = random.randint(-5, 5)
        new_val = orig + delta
        while new_val == 0 or new_val == orig:
            delta = random.randint(-5, 5)
            new_val = orig + delta

        if new_val % 1 == 0:
            new_val = int(new_val)

        if new_val < 0:
            new_exp = f"({new_val})"
        else:
            new_exp = str(new_val)

    except ValueError:
        new_exp = f"({old_exp}{random.choice(['+', '-'])}{random.randint(1, 5)})"

    new_expr = f"{base} ^ {new_exp}"
    return expression.replace(old_expr.strip(), new_expr, 1)


def introduce_variable(expression):
    parts = expression.split(":=", 1)
    expr = parts[0]

    thm = expr.split(":", 1)
    if len(thm) < 2:
        print("Invalid format for variable introduction:", expression)
        return expression

    prefix, body = thm
    decls = re.findall(r"(?<!\w)([a-zA-Z0-9]\w*(?:\s+[a-zA-Z0-9]\w*)*)\s*:\s*([^\s,()]+)", body)

    if decls:
        vars_str, typ = random.choice(decls)
        new_var = random.choice(list(string.ascii_lowercase))
        old_decl = f"{vars_str} : {typ}"
        new_decl = f"{vars_str} {new_var} : {typ}"
        body = body.replace(old_decl, new_decl, 1)

    out = f"{prefix}:{body}"
    if len(parts) > 1:
        out += ":=" + parts[1]
    return out


def change_variable_type(expression):
    available_types = ["ℕ", "ℤ", "ℚ", "ℝ", "𝔹", "𝕊", "𝕋", "α", "×", "β", "ℒ", "𝕎", "𝕌", "𝕍", "𝕏", "𝕐", "𝕄", "𝕀", "𝕆"]

    decls = re.findall(r"\(([^:]+ : [^\s]+)\)", expression)
    if not decls:
        return expression

    chosen = random.choice(decls)
    match = re.search(r": ([^\s]+)", chosen)
    if not match:
        return expression

    current = match.group(1)
    new_type = current
    while new_type == current:
        new_type = random.choice(available_types)

    new_decl = chosen.replace(current, new_type)
    return expression.replace(chosen, new_decl)


def modify_equality(expression):
    parts = expression.split(":=", 1)
    expr = parts[0]

    positions = [i for i, c in enumerate(expr) if c in "=≠"]
    if positions:
        i = random.choice(positions)
        chars = list(expr)
        chars[i] = "≠" if chars[i] == "=" else "="
        expr = "".join(chars)

    return expr + (":=" + parts[1] if len(parts) > 1 else "")


def modify_unpaired(responses, current):
    options = [r for r in responses if r != current]
    return random.choice(options) if options else current


# ----------------------------
# Dispatch
# ----------------------------


def modify_response(response, modification_type, all_responses):
    if modification_type == "constant":
        return modify_constant(response)
    if modification_type == "exponent":
        return modify_exponent(response)
    if modification_type == "variable_new":
        return introduce_variable(response)
    if modification_type == "variable_type":
        return change_variable_type(response)
    if modification_type == "equality":
        return modify_equality(response)
    if modification_type == "unpaired":
        return modify_unpaired(all_responses, response)
    raise ValueError("Invalid modification type:", modification_type)


# ----------------------------
# HF Batch Modifier
# ----------------------------


def modify_batch(batch, seed, full_dataset):
    random.seed(seed)

    modification_types = [
        "constant",
        "exponent",
        "variable_new",
        "variable_type",
        "equality",
        "unpaired",
    ]

    num_modifications = 10
    total = len(batch["input"])

    # Pre-allocate per-theorem plans
    plans = [
        modification_types[:] * (num_modifications // len(modification_types))
        + random.sample(modification_types, num_modifications % len(modification_types))
        for _ in range(total)
    ]

    all_responses = full_dataset["output"]
    out = {"input": [], "output": [], "label": [], "misalign_type": []}

    for i, (nat, form) in enumerate(zip(batch["input"], batch["output"])):
        modified_set = set()

        for j, mtype in enumerate(plans[i]):
            new = modify_response(form, mtype, all_responses)
            if new == form:
                continue

            out["input"].append(nat)
            out["output"].append(new)
            out["label"].append(0)
            out["misalign_type"].append(mtype)
            modified_set.add(new)

    return out


def modify_dataset(dataset, num_proc, seed):
    modified = dataset.map(lambda batch: modify_batch(batch, seed, dataset), batched=True, num_proc=num_proc)
    return concatenate_datasets([dataset, modified])


# ----------------------------
# Main
# ----------------------------


def main(dataset_name, output_path, num_proc, seed):
    random.seed(seed)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"Loading HF dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    print("Modifying dataset…")
    out_ds = modify_dataset(dataset, num_proc, seed)
    print(f"Starting dataset size: {len(dataset)}")
    print(f"New dataset size: {len(out_ds)}")
    print(f"Number of misaligned examples added: {len(out_ds) - len(dataset)}")

    print(f"Saving to {output_path}")
    out_ds.save_to_disk(output_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create misaligned HF dataset samples.")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="HuggingFace dataset name (or local HF script)."
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=None)

    args = parser.parse_args()
    main(args.dataset_name, args.output_path, args.num_proc, seed=args.seed)
