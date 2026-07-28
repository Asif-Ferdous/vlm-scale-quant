# Bigger or Cheaper? Scale and Quantization Effects on Uncertainty Signals in Vision-Language Models

M M Asif Ferdous · Independent Researcher · 2026

Code, data, and figures for the paper. arXiv preprint — (http://arxiv.org/abs/2607.24440)
## Summary

A practitioner deploying a vision-language model on a fixed 16 GB GPU can fit three
configurations, and must pick one:

| Arm | Model | Precision |
|-----|-------|-----------|
| A | Qwen2-VL-2B-Instruct | fp16 |
| B | Qwen2-VL-2B-Instruct | NF4 4-bit |
| C | Qwen2-VL-7B-Instruct | NF4 4-bit |

We measure how scale (**B vs C**) and 4-bit quantization (**A vs B**) affect two
confidence signals — the number the model *states* (verbalized) and its own mean
token probability (internal) — across 5,700 predictions under six realistic image
degradations at three severities.

| Finding | Detail |
|---------|--------|
| Scale helps the internal signal a lot | Mean error-detection AUROC 0.80 → 0.98 (2B → 7B, both 4-bit) |
| Scale barely helps verbalized confidence | Mean AUROC 0.61 → 0.69; at chance under low light even at 7B |
| The know–say gap widens with scale | Internal−verbalized AUROC gap 0.19 (2B) → 0.29 (7B) |
| Quantization is cheap for accuracy | −1.6 accuracy points (2B fp16 → 4-bit) |
| Quantization is expensive for the signal | Internal AUROC 0.95 → 0.80; verbalized parse rate 99% → 64% |
| Deployment answer | 7B-4bit wins on accuracy *and* uncertainty; prefer parameters over precision |
| Everyone fails under severe low light | Accuracy ≤ 0.80 even at 7B; needs an upstream image-quality check |

## Repository layout

```
├── notebook/
│   └── vlm_scale_study.ipynb     full experiment, runs on a free Colab T4
├── code/
│   ├── scale_quant_experiment.py the experiment script (source for the notebook)
│   └── analysis.py               re-scoring, metrics, bootstrap CIs, figures
├── data/
│   ├── allpreds_A_2B_fp16.csv    per-prediction records, 1,900 rows
│   ├── allpreds_B_2B_nf4.csv     per-prediction records, 1,900 rows
│   ├── allpreds_C_7B_nf4.csv     per-prediction records, 1,900 rows
│   └── summary_*.csv             per-condition accuracy/AUROC/AURC with 95% CIs
├── figures/                      the four paper figures
└── paper/
    ├── main.tex                  the paper (ACL format)
    ├── references.bib            bibliography
    └── fig_*.png                 figures (flat paths, alongside the .tex)
```

## Reproducing

Each arm reproduces on a single free-tier NVIDIA T4 in roughly one hour.

1. Open `notebook/vlm_scale_study.ipynb` in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Set `ARM` (`smoke` → `pilot` → `B` → `A` → `C`) and Run all, one arm per session.

Results checkpoint to Drive after every condition, so a disconnect never loses
work. An item manifest enforces that all arms score identical images and option
sets. Random seeds are fixed throughout.

## Method

- **Data**: 100-item subset of Food101 (`ethz/food101`, validation split, seed 0),
  four-option multiple choice.
- **Degradations**: JPEG, motion blur, low light, glare, rotation, resample —
  three severities each, plus clean = 19 conditions per arm.
- **Signals**: verbalized confidence (parsed from a two-line template) and internal
  confidence (mean token probability over the answer span only).
- **Metrics**: accuracy, ECE, Brier, error-detection AUROC, and area under the
  risk–coverage curve (AURC), with B = 2000 bootstrap intervals.

## Scoring note

Small and quantized models often reply with a bare label ("Beignets") instead of
the requested `Answer:` format. Such replies are correct and are scored as such:
`analysis.py` re-derives correctness from the raw model output uniformly across all
arms, so no model is penalised for skipping the format. Verbalized-confidence parse
rate is reported separately (and is itself a finding).

## Limitations

Two scale points give a direction, not a scaling law. Quantization is measured at
one model size. n ≈ 100 per condition, so some intervals are wide. One dataset, one
task format, one model family. Multiple choice forces an answer, so refusal is not
observed. Degradations are synthetic approximations of camera artifacts.

## License

MIT for the code. Food101 and the model weights are subject to their own licenses.
