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
from sklearn.metrics import accuracy_score, f1_score, classification_report

if __name__ == "__main__":
    # -----------------------------
    # Config
    # -----------------------------
    MODEL_PATH = Path("alignment_classifier_qwen8b_2/")
    # TOKENIZER_PATH = "Qwen/Qwen3-4B-Instruct-2507"
    TOKENIZER_PATH = "Qwen/Qwen3-8B"
    # DATA_PATH = Path("data/minif2f_test_formatted.json")
    DATA_PATH = "offendo/math-atlas-alignment-3600"
    FINAL_OUTPUT_PATH = Path(MODEL_PATH, str(DATA_PATH).split('/')[-1])
    Path(FINAL_OUTPUT_PATH).parent.mkdir(exist_ok=True, parents=True)


    # -----------------------------
    # Load data
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if Path(DATA_PATH).exists():
        ds = load_dataset('json', data_files=[str(DATA_PATH)], split='train')
    else:
        ds = load_dataset(str(DATA_PATH), split="test")
    ds = ds.map(make_conversations, load_from_cache_file=False)
    ds = ds.map(lambda batch: {'text': formatting_func(tokenizer, batch)}, batched=True, load_from_cache_file=False)
    ds = ds.shuffle(seed=1234)
    prompts = list(ds['text'])

    # -----------------------------
    # Load model
    # -----------------------------
    llm = LLM(model=str(MODEL_PATH), enforce_eager=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
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
            "formal": record.get('formal_no_comments', record.get('formal')), # default to formal
            "informal": record['informal'],
            "prompt": record['text'],
            "pred": out.outputs[0].text,
            "pred_label": False if "misaligned" in out.outputs[0].text.split()[-1].strip() else True,
            "label": record['label'] == 'aligned',
        })

    # -----------------------------
    # Save final results
    # -----------------------------
    outdf = pd.DataFrame.from_records(results)
    outdf.to_json(FINAL_OUTPUT_PATH)
    
    print(classification_report(outdf['label'], outdf["pred_label"], target_names=['misaligned', 'aligned']))

    total_time = time.time() - start_time
    print(f"\nDone! Saved {len(results)} results.")
    print(f"Total time: {total_time/60:.2f} minutes")
