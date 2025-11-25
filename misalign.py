#!/usr/bin/env python3
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
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import string
from functools import partial
from typing import Iterable, MutableMapping

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

# ----------------------------
# Logging / Constants
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default available types to switch to (kept from original)
AVAILABLE_TYPES = [
    "ℕ",
    "ℤ",
    "ℚ",
    "ℝ",
    "𝔹",
    "𝕊",
    "𝕋",
    "α",
    "×",
    "β",
    "ℒ",
    "𝕎",
    "𝕌",
    "𝕍",
    "𝕏",
    "𝕐",
    "𝕄",
    "𝕀",
    "𝕆",
]

# Precompile regexes used frequently
_RE_DIGITS_WORD = re.compile(r"(?<!\\u)\b\d+\b")
# a fairly permissive exponent pattern (kept similar to original intent)
_RE_EXPONENT = re.compile(
    r"(\s*-?\s*\w+\s*|\s*\(-?\s*\d+\s*\))\s*\^\s*(\s*-?\s*\d+\.?\d*\s*|\s*\w+\s*|\s*\(.+?\)\s*)"
)
# simple declaration finder like "x : Type"
_RE_DECL = re.compile(r"(?<!\w)([a-zA-Z0-9]\w*(?:\s+[a-zA-Z0-9]\w*)*)\s*:\s*([^\s,()]+)")
# find (name : Type) style tuples
_RE_PAREN_DECL = re.compile(r"\(([^:]+ : [^\s]+)\)")
# find colon-split into prefix/body like "theorem ... : ..." etc.
_RE_COLON_SPLIT = re.compile(r"^(.*?):(.*)$", flags=re.DOTALL)


# ----------------------------
# Modification Functions
# ----------------------------


def modify_constant(expression: str, rng: random.Random) -> str:
    """
    Finds a numeric constant in the part after the first ':' and increases it by 1..100
    Returns modified expression or original if nothing found / parsing fails.
    """
    m = _RE_COLON_SPLIT.match(expression)
    if not m:
        logger.debug("modify_constant: no ':' found in expression; returning original")
        return expression

    prefix, body = m.group(1), m.group(2)
    constants = _RE_DIGITS_WORD.findall(body)
    if not constants:
        return expression

    chosen = rng.choice(constants)
    try:
        new_val = str(int(chosen) + rng.randint(1, 100))
    except ValueError:
        logger.debug("modify_constant: chosen constant is not int-like: %s", chosen)
        return expression

    # replace only the first occurrence of the chosen numeric token
    new_body = re.sub(r"(?<!\\u)\b" + re.escape(chosen) + r"\b", new_val, body, count=1)
    return prefix + ":" + new_body


def modify_exponent(expression: str, rng: random.Random) -> str:
    """
    Find an exponent expression like `x ^ 2` (loosely) and change the exponent.
    If exponent is numeric, add a small delta; otherwise wrap the old exponent with +/- small int.
    """
    matches = list(_RE_EXPONENT.finditer(expression))
    if not matches:
        return expression

    chosen = rng.choice(matches)
    old_expr = chosen.group(0)
    # get groups for base and exponent
    groups = _RE_EXPONENT.match(old_expr).groups()
    if not groups or len(groups) < 2:
        return expression

    base, old_exp = groups[0].strip(), groups[1].strip()
    # strip surrounding parentheses for numeric attempt
    stripped_exp = old_exp.strip().lstrip("(").rstrip(")")

    # try numeric modification
    try:
        orig = float(stripped_exp)
        # delta not zero and not equal to orig
        delta = rng.randint(-5, 5)
        new_val = orig + delta
        # avoid leaving unchanged or zero
        attempts = 0
        while (new_val == orig or new_val == 0) and attempts < 20:
            delta = rng.randint(-5, 5)
            new_val = orig + delta
            attempts += 1

        if abs(new_val - int(new_val)) < 1e-9:
            new_exp = str(int(new_val))
        else:
            new_exp = str(new_val)

        # if negative, keep parentheses
        if float(new_exp) < 0:
            new_exp = f"({new_exp})"

    except ValueError:
        # non-numeric exponent: append small +/- expression
        op = rng.choice(["+", "-"])
        amt = rng.randint(1, 5)
        new_exp = f"({stripped_exp}{op}{amt})"

    new_expr = f"{base} ^ {new_exp}"
    # replace the first occurrence of the old expression (trimmed)
    return expression.replace(old_expr.strip(), new_expr, 1)


def introduce_variable(expression: str, rng: random.Random) -> str:
    """
    Introduce a fresh single-letter variable within a declaration of the form
    'x : Type' by changing 'x : Type' -> 'x y : Type' where y is new.
    Works on the portion before ':=' if present, matching earlier behavior.
    """
    parts = expression.split(":=", 1)
    head = parts[0]
    m = _RE_COLON_SPLIT.match(head)
    if not m:
        logger.debug("introduce_variable: no ':' found in head; returning original")
        return expression

    prefix, body = m.group(1), m.group(2)
    decls = _RE_DECL.findall(body)
    if not decls:
        return expression

    vars_str, typ = rng.choice(decls)
    # choose a new lower-case variable not present in vars_str
    existing_letters = set(vars_str.replace(" ", ""))
    candidates = [c for c in string.ascii_lowercase if c not in existing_letters]
    if not candidates:
        # fallback: pick any lowercase
        new_var = rng.choice(list(string.ascii_lowercase))
    else:
        new_var = rng.choice(candidates)

    old_decl = f"{vars_str} : {typ}"
    new_decl = f"{vars_str} {new_var} : {typ}"
    new_body = body.replace(old_decl, new_decl, 1)
    out = f"{prefix}:{new_body}"
    if len(parts) > 1:
        out += ":=" + parts[1]
    return out


def change_variable_type(expression: str, rng: random.Random, available_types: Iterable[str] = AVAILABLE_TYPES) -> str:
    """
    Change a type annotation inside parentheses '(x : Type)' to another random type
    from available_types. If no parenthesized declarations are found, return unchanged.
    """
    decls = _RE_PAREN_DECL.findall(expression)
    if not decls:
        return expression

    chosen = rng.choice(decls)
    match = re.search(r":\s*([^\s]+)", chosen)
    if not match:
        return expression

    current = match.group(1)
    new_type = current
    available = list(available_types)
    # get a different one if possible
    if len(available) == 0:
        return expression

    attempts = 0
    while new_type == current and attempts < 50:
        new_type = rng.choice(available)
        attempts += 1

    if new_type == current:
        return expression

    new_decl = chosen.replace(current, new_type)
    return expression.replace(chosen, new_decl, 1)


def modify_equality(expression: str, rng: random.Random) -> str:
    """
    Flip a single occurrence of '=' to '≠' or vice versa in the portion before ':=' (if present).
    """
    parts = expression.split(":=", 1)
    head = parts[0]
    positions = [i for i, c in enumerate(head) if c in "=≠"]
    if not positions:
        return expression

    i = rng.choice(positions)
    chars = list(head)
    chars[i] = "≠" if chars[i] == "=" else "="
    new_head = "".join(chars)
    return new_head + (":=" + parts[1] if len(parts) > 1 else "")


def modify_unpaired(responses: list[str], current: str, rng: random.Random) -> str:
    """
    Select another item from a list of candidate responses (not equal to current).
    If no alternative exists, return current.
    """
    options = [r for r in responses if r != current]
    if not options:
        return current
    return rng.choice(options)


# ----------------------------
# Dispatch
# ----------------------------


_MODIFIERS = {
    "constant": modify_constant,
    "exponent": modify_exponent,
    "variable_new": introduce_variable,
    "variable_type": change_variable_type,
    "equality": modify_equality,
    # 'unpaired' handled specially because it needs all_responses and different signature
}


def modify_response(response: str, modification_type: str, all_responses: list[str] | None, rng: random.Random) -> str:
    """
    Dispatch wrapper that calls the appropriate modifier with the provided RNG.
    For 'unpaired' the all_responses list is required.
    """
    if modification_type == "unpaired":
        if all_responses is None:
            logger.debug("modify_response: 'unpaired' requested but all_responses is None")
            return response
        return modify_unpaired(all_responses, response, rng)

    func = _MODIFIERS.get(modification_type)
    if not func:
        raise ValueError(f"Unknown modification type: {modification_type}")
    # call modifier with rng where accepted
    # note: type ignore because mypy can't track different call signatures nicely here
    return func(response, rng)  # type: ignore[arg-type]


# ----------------------------
# HF Batch Modifier
# ----------------------------


def modify_batch(
    batch: MutableMapping[str, list[str]],
    rng: random.Random,
    full_dataset: Dataset | None,
    formal_column: str,
    informal_column: str,
    modification_types: list[str] | None = None,
    num_modifications_per_example: int = 10,
) -> dict[str, list]:
    """
    Given a batched dict from datasets.map (lists of strings), produce a dict with generated
    misaligned examples. Returned mapping keys should be dataset column names.

    This function is intentionally pure: it uses only the passed rng and arguments.
    """
    if modification_types is None:
        modification_types = ["constant", "exponent", "variable_new", "variable_type", "equality", "unpaired"]

    # defensive checks
    if formal_column not in batch or informal_column not in batch:
        raise KeyError(f"Expected batch to contain columns '{formal_column}' and '{informal_column}'")

    formal_list = batch[formal_column]
    informal_list = batch[informal_column]
    total = len(formal_list)
    out: dict[str, list] = {"nl_statement": [], "fl_misaligned": [], "fl_statement": [], "label": [], "misalign_type": []}

    # prepare plans for each example (deterministic given rng)
    plans: list[list[str]] = []
    for _ in range(total):
        # basic evenly distributed plan with random remainder
        base = modification_types[:] * (num_modifications_per_example // len(modification_types))
        remainder = rng.sample(modification_types, num_modifications_per_example % len(modification_types))
        rng.shuffle(base)
        plan = base + remainder
        rng.shuffle(plan)
        plans.append(plan)

    # gather all responses for 'unpaired' selection
    all_responses: list[str] | None = None
    if full_dataset is not None:
        # try to find a sensible 'output' column; original code used full_dataset["output"]
        # If that column doesn't exist, fall back to formal_column
        all_responses = list(full_dataset[formal_column])

    # iterate through examples
    for i in range(total):
        nat = informal_list[i]
        form = formal_list[i]
        modified_set = set()
        for mtype in plans[i]:
            new = modify_response(form, mtype, all_responses, rng)
            # skip no-op modifications
            if new == form:
                continue
            # avoid duplicates per-example
            if new in modified_set:
                continue
            modified_set.add((new, mtype))
        modified_exs, modified_types = zip(*modified_set)
        out["nl_statement"].append(nat)
        out["fl_misaligned"].append(modified_exs)
        out["misalign_type"].append(modified_types)
        out["fl_statement"].append(form)
        out["label"].append(0)

    return out


def modify_dataset(
    dataset: Dataset,
    num_proc: int | None,
    seed: int,
    formal_column: str,
    informal_column: str,
    modification_types: list[str] | None = None,
    num_modifications_per_example: int = 10,
) -> Dataset:
    """
    Apply `modify_batch` across the whole dataset using datasets.Dataset.map (batched).
    Returns concatenated dataset: original + modified.
    """
    rng = random.Random(seed)

    # partial function to pass to datasets.map (must be picklable; the RNG state is embedded via integer seed slices)
    # To ensure deterministic behavior across processes, create per-batch RNG seeds deterministically:
    def map_wrapper(batch: MutableMapping[str, list[str]]) -> dict[str, list]:
        # derive a new local seed for this batch from global rng
        local_seed = rng.randint(0, 2**30 - 1)
        local_rng = random.Random(local_seed)
        return modify_batch(
            batch=batch,
            rng=local_rng,
            full_dataset=dataset,
            formal_column=formal_column,
            informal_column=informal_column,
            modification_types=modification_types,
            num_modifications_per_example=num_modifications_per_example,
        )

    logger.info("Mapping modifications over dataset (num_proc=%s)...", str(num_proc))
    modified = dataset.map(map_wrapper, batched=True, num_proc=num_proc, batch_size=100)
    logger.info("Concatenating original and modified datasets...")

    return modified


# ----------------------------
# Main
# ----------------------------


def main(
    dataset_name: str,
    output_path: str,
    num_proc: int | None,
    seed: int,
    formal_column: str,
    informal_column: str,
    num_modifications_per_example: int,
):
    rng = random.Random(seed)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    logger.info("Loading HF dataset: %s", dataset_name)
    dataset = load_dataset(dataset_name, split="train")
    logger.info("Dataset loaded with %d examples", len(dataset))

    # basic validation of columns
    for col in (formal_column, informal_column):
        if col not in dataset.column_names:
            raise KeyError(f"Column '{col}' not found in dataset. Available columns: {dataset.column_names}")

    logger.info("Modifying dataset…")
    out_ds = modify_dataset(
        dataset, 
        num_proc=num_proc,
        seed=seed,
        formal_column=formal_column,
        informal_column=informal_column,
        num_modifications_per_example=num_modifications_per_example,
    )

    logger.info("Starting dataset size: %d", len(dataset))
    logger.info("New dataset size: %d", len(out_ds))
    logger.info("Number of misaligned examples added: %d", len(out_ds) - len(dataset))

    logger.info("Saving to %s", output_path)
    out_ds.save_to_disk(output_path)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create misaligned HF dataset samples.")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset name (or local HF script).")
    parser.add_argument("--formal_column", type=str, required=True, help="column name for formal statement to modify.")
    parser.add_argument("--informal_column", type=str, required=True, help="column name for informal statement.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the resulting dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic output.")
    parser.add_argument("--num_proc", type=int, default=None, help="Number of processes to use for dataset.map.")
    parser.add_argument("--num_modifications_per_example", type=int, default=10, help="Number of modifications to use for each example.")

    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        num_proc=args.num_proc,
        seed=args.seed,
        formal_column=args.formal_column,
        informal_column=args.informal_column,
        num_modifications_per_example=args.num_modifications_per_example,
    )

