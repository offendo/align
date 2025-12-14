import time
import pandas as pd
import json
import os
from pathlib import Path
from datasets import load_dataset
from vllm import LLM, SamplingParams
from more_itertools import chunked
from transformers import AutoTokenizer
from train_binary import make_conversations, SYSTEM_PROMPT, formatting_func
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    # -----------------------------
    # Config
    # -----------------------------
    MODEL_PATH = Path("alignment_classifier_filtered_bleu/checkpoint-5000/")
    TOKENIZER_PATH = Path("alignment_classifier_filtered_bleu/tokenizer")
    SYSTEM_PROMPT_PATH = "aligner_prompt.txt"
    # DATA_PATH = Path("data/minif2f_test_formatted.json")
    DATA_PATH = Path("mathatlas_30_benchmark.jsonl")
    FINAL_OUTPUT_PATH = Path(MODEL_PATH, DATA_PATH.name)
    Path(FINAL_OUTPUT_PATH).parent.mkdir(exist_ok=True, parents=True)


    # -----------------------------
    # Load data
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if Path(DATA_PATH).exists():
        ds = load_dataset('json', data_files=[str(DATA_PATH)], split='train')
    else:
        ds = load_dataset(str(DATA_PATH), split="val")
    ds = ds.map(make_conversations, load_from_cache_file=False)
    ds = ds.map(lambda batch: {'text': formatting_func(tokenizer, batch)}, batched=True, load_from_cache_file=False)
    prompts = list(ds['text'])

    # -----------------------------
    # Load model
    # -----------------------------
    llm = LLM(model=str(MODEL_PATH), enforce_eager=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=25,
        seed=1337,
        skip_special_tokens=True,
    )

    # -----------------------------
    # Process batches
    # -----------------------------
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for record, out in zip(ds, outputs):
        results.append({
            "kind": record['kind'],
            "formal": record['formal'],
            "informal": record['informal'],
            "prompt": record['text'],
            "pred": out.outputs[0].text,
            "pred_label": False if "distinct" in out.outputs[0].text else True,
            "label": record['label'],
        })

    # -----------------------------
    # Save final results
    # -----------------------------
    outdf = pd.DataFrame.from_records(results)
    outdf.to_json(FINAL_OUTPUT_PATH)
    
    acc = accuracy_score(outdf['label'], outdf["pred_label"])
    f1 = f1_score(outdf['label'], outdf["pred_label"], average='macro')
    print('Accuracy: ', acc)
    print('F1: ', f1)

    total_time = time.time() - start_time
    print(f"\nDone! Saved {len(results)} results.")
    print(f"Total time: {total_time/60:.2f} minutes")
