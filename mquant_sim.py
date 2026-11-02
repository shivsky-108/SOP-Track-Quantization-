import argparse, torch, torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration
from common import *

# Global state read by every QLinear layer:
#   mode     : which quantization scheme to apply
#   vis_mask : (B, T) bool tensor, True where the token is a visual token
STATE = {"mode": "fp", "vis_mask": None}


def fq(x, scale, bits):
    """Fake quantization: round to the integer grid, then map back to float.
    The number is now restricted to 2^bits levels, but stays a float, so we
    can measure the accuracy effect without writing int8 kernels."""
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale


class QLinear(nn.Module):
    def __init__(self, lin, w_bits, a_bits):
        super().__init__()
        self.a_bits = a_bits
        w = lin.weight.data
        self.register_buffer("w_fp", w.clone())            # original weights
        # weights: per-output-channel symmetric round-to-nearest quantization
        qmax = 2 ** (w_bits - 1) - 1
        ws = (w.abs().amax(dim=1, keepdim=True).float() / qmax).clamp(min=1e-8)
        self.register_buffer("w_q", fq(w.float(), ws, w_bits).to(w.dtype))
        self.bias = lin.bias
        # calibration results: largest |activation| seen per modality
        self.register_buffer("max_vis", torch.zeros((), device=w.device))
        self.register_buffer("max_txt", torch.zeros((), device=w.device))

    def forward(self, x):                                   # x: (B, T, D)
        mode = STATE["mode"]
        if mode == "fp":                                    # unquantized reference
            return nn.functional.linear(x, self.w_fp, self.bias)

        vm = STATE["vis_mask"]
        if vm is None or vm.shape[1] != x.shape[1]:
            vm = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        xf = x.float()

        if mode == "calib":                                 # record ranges, no quant
            ax = xf.abs()
            if vm.any():
                self.max_vis = torch.maximum(self.max_vis, ax[vm].max())
            if (~vm).any():
                self.max_txt = torch.maximum(self.max_txt, ax[~vm].max())
            return nn.functional.linear(x, self.w_fp, self.bias)

        qmax = 2 ** (self.a_bits - 1) - 1
        if mode == "dyn_token":
            # scale computed at runtime for EACH token: accurate but slow
            s = xf.abs().amax(dim=-1, keepdim=True) / qmax
        elif mode == "static_shared":
            # ONE fixed scale for visual + text tokens (naive static)
            s = torch.maximum(self.max_vis, self.max_txt) / qmax
        elif mode == "static_msq":
            # MSQ idea: one fixed scale for visual tokens, another for text
            s = torch.where(vm.unsqueeze(-1), self.max_vis, self.max_txt) / qmax
        else:
            raise ValueError(mode)

        xq = fq(xf, s.clamp(min=1e-8), self.a_bits).to(x.dtype)
        return nn.functional.linear(xq, self.w_q, self.bias)


def quantize_llm_linears(model, w_bits, a_bits):
    """Swap nn.Linear -> QLinear inside the LLM decoder layers only.
    The vision tower and lm_head stay in FP16."""
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if (isinstance(child, nn.Linear) and ".layers." in full
                    and "visual" not in full):
                setattr(mod, cname, QLinear(child, w_bits, a_bits))
                n += 1
    print(f"Replaced {n} linear layers")   # must be > 0! (see troubleshooting)
    assert n > 0, "No layers replaced: check module names with model.named_modules()"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--n_calib", type=int, default=32)
    ap.add_argument("--w_bits", type=int, default=8)
    ap.add_argument("--a_bits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=40)
    args = ap.parse_args()
    set_seed(args.seed)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to("cuda").eval()
    processor = get_processor()
    img_id = model.config.image_token_id    # token id used for visual tokens

    def set_mask(inputs):                   # runs before each forward pass
        STATE["vis_mask"] = inputs["input_ids"] == img_id

    quantize_llm_linears(model, args.w_bits, args.a_bits)

    # ---- Calibration: collect per-layer, per-modality activation ranges ----
    # Uses the VALIDATION split so calibration data never overlaps the test set.
    STATE["mode"] = "calib"
    calib = get_data("validation", args.n_calib, seed=args.seed)
    for i, ex in enumerate(calib):
        inp = build_inputs(processor, ex, "cuda")
        set_mask(inp)
        if i == 0:                          # sanity check on the mask
            print("visual tokens in 1st sample:", int(STATE["vis_mask"].sum()),
                  "of", inp["input_ids"].shape[1])
            assert STATE["vis_mask"].sum() > 0, "visual mask is empty!"
        with torch.no_grad():
            model(**inp, use_cache=False)

    # Evidence for the paper's motivation: how much larger are visual ranges?
    qls = [m for m in model.modules() if isinstance(m, QLinear)]
    ratios = [(m.max_vis / m.max_txt.clamp(min=1e-6)).item() for m in qls]
    print(f"visual/text abs-max ratio: mean={sum(ratios)/len(ratios):.2f} "
          f"max={max(ratios):.2f}  layers with ratio>1: "
          f"{sum(r > 1 for r in ratios)}/{len(ratios)}")

    # ---- Evaluate every mode on the SAME test subset ----
    data = get_data("test", args.n_eval, seed=args.seed)
    for mode in ["fp", "dyn_token", "static_shared", "static_msq"]:
        STATE["mode"] = mode
        res = evaluate(model, processor, data, before_forward=set_mask)
        log_result({"name": f"sim_W{args.w_bits}A{args.a_bits}_{mode}",
            "seed": args.seed,
            "ratio_mean": sum(ratios) / len(ratios),
            "ratio_max": max(ratios), **res})


if __name__ == "__main__":
    main()