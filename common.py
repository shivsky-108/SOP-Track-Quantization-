import json, os, time, torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LETTERS = "ABCDEFGH"   # option letters used in the multiple-choice prompt


def set_seed(seed):
    # Seeds every random source. Inference is deterministic anyway, but this
    # makes the seed explicit and covers any library that uses randomness.
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_processor(model_id=MODEL_ID):
    # min/max_pixels cap the number of visual tokens per image.
    # Fewer visual tokens = less GPU memory. Keep identical across ALL runs.
    return AutoProcessor.from_pretrained(
        model_id, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)


def get_data(split, n, seed=0):
    # ScienceQA image subset. Every example has an image, a question,
    # a list of choices, an optional hint, and an integer answer index.
    ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split=split)
    # fixed seed -> the same n examples every run (fair comparison)
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def build_inputs(processor, ex, device):
    # Build a multiple-choice prompt: hint + question + lettered options.
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(ex["choices"]))
    hint = (ex["hint"] + "\n") if ex.get("hint") else ""
    prompt = (f"{hint}Question: {ex['question']}\nOptions:\n{opts}\n"
              "Answer with the option's letter from the given choices directly.")
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[ex["image"].convert("RGB")],
                     return_tensors="pt").to(device)


@torch.no_grad()
def evaluate(model, processor, data, before_forward=None):
    """Score = which option letter has the highest next-token logit.
    This avoids free-text generation, so scoring is exact and fast.
    Returns accuracy, prefill latency, and peak GPU memory."""
    dev = model.device
    tok = processor.tokenizer
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]

    torch.cuda.reset_peak_memory_stats()
    correct = nan_count = 0
    fwd_time = 0.0
    for ex in data:
        inputs = build_inputs(processor, ex, dev)
        if before_forward:            # hook used by the simulated-quant script
            before_forward(inputs)    # (sets the visual-token mask)
        torch.cuda.synchronize()
        t0 = time.time()
        # one forward pass over the whole prompt = prefill, a proxy for
        # time-to-first-token. use_cache=False saves memory.
        logits = model(**inputs, use_cache=False).logits[0, -1]
        torch.cuda.synchronize()
        fwd_time += time.time() - t0

        if torch.isnan(logits).any():
            nan_count += 1            # fp16 overflow check (see troubleshooting)
        k = len(ex["choices"])        # only consider letters that are valid options
        pred = logits[letter_ids[:k]].argmax().item()
        correct += int(pred == ex["answer"])

    n = len(data)
    return {"acc": 100 * correct / n, "n": n,
            "prefill_sec_per_sample": fwd_time / n,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "nan_count": nan_count}


def log_result(rec, path="logs/results.jsonl"):
    # one JSON line per configuration; summarize.py reads this file
    os.makedirs("logs", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))