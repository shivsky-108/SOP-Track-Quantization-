import json

rows = [json.loads(l) for l in open("logs/results.jsonl")]
print("| Config | Acc (%) | Peak mem (GB) | Prefill s/sample | Model size (GB) |")
print("|---|---|---|---|---|")
for r in rows:
    print(f"| {r['name']} | {r['acc']:.2f} | {r['peak_mem_gb']:.2f} | "
          f"{r['prefill_sec_per_sample']:.3f} | {r.get('model_size_gb', float('nan')):.2f} |")