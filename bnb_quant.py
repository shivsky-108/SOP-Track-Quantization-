import argparse, torch
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
ap.add_argument("--n_eval", type=int, default=500)
args = ap.parse_args()

# Quantize only the language model. Keep the vision tower and lm_head in
# higher precision, since they are small and sensitive to quantization.
