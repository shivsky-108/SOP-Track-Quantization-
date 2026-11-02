| Config | Acc (%) | Peak mem (GB) | Prefill s/sample | Model size (GB) |
|---|---|---|---|---|
| baseline_fp16 | 80.40 | 4.68 | 0.152 | 4.42 |
| bnb_8bit | 81.00 | 2.72 | 0.372 | 2.44 |
| bnb_4bit | 78.20 | 1.88 | 0.190 | 1.46 |
| sim_W8A8_fp | 80.40 | 7.32 | 0.158 | nan |
| sim_W8A8_dyn_token | 81.00 | 7.32 | 0.186 | nan |
| sim_W8A8_static_shared | 64.60 | 7.32 | 0.179 | nan |
| sim_W8A8_static_msq | 64.40 | 7.32 | 0.181 | nan |
