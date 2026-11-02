import argparse, torch
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
ap.add_argument("--n_eval", type=int, default=500)
ap.add_argument("--seed", type=int, default=40)    # last two digits of ID
args = ap.parse_args()
set_seed(args.seed)                                # must come after parse_args()

# Quantize only the language model. Keep the vision tower and lm_head in
# higher precision, since they are small and sensitive to quantization.
skip = ["visual", "lm_head"]
if args.bits == 8:
    cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
else:
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # 4-bit "NormalFloat" format
        bnb_4bit_compute_dtype=torch.float16,  # matmuls run in fp16
        llm_int8_skip_modules=skip)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=cfg, device_map="cuda",
    torch_dtype=torch.float16).eval()
processor = get_processor()

size_gb = model.get_memory_footprint() / 1e9

data = get_data("test", args.n_eval, seed=args.seed)   # was seed=0
res = evaluate(model, processor, data)
log_result({"name": f"bnb_{args.bits}bit", "seed": args.seed,
            "model_size_gb": size_gb, **res})