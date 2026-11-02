import argparse, torch
from transformers import Qwen2VLForConditionalGeneration
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--n_eval", type=int, default=500)
ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
args = ap.parse_args()

dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype).to("cuda").eval()
processor = get_processor()

# model size in GB, as loaded (parameters + buffers)
size_gb = model.get_memory_footprint() / 1e9

data = get_data("test", args.n_eval, seed=0)
res = evaluate(model, processor, data)
log_result({"name": f"baseline_{args.dtype}", "model_size_gb": size_gb, **res})