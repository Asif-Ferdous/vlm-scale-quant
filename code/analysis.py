"""
paper2_analysis.py
==================
Analysis for the scale x quantization study. Re-scores all three arms uniformly
from the raw model outputs (so scoring is identical across arms regardless of
which code version produced each CSV), then computes:

  - per-condition accuracy, verbalized-AUROC, internal-AUROC with bootstrap CIs
  - the scale contrast (2B-nf4 vs 7B-nf4) and quantization contrast (2B-fp16 vs 2B-nf4)
  - risk-coverage curves and AURC per arm/signal
  - all figures used in the paper

Usage:
    python paper2_analysis.py \
        --a allpreds_A_2B_fp16.csv --b allpreds_B_2B_nf4.csv --c allpreds_C_7B_nf4.csv \
        --out ./paper2_out
"""
import argparse, re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
B_BOOT = 2000
FAM_ORDER = ["clean", "jpeg", "motion_blur", "low_light", "glare", "rotation", "resample"]

# --------------------------------------------------------------- scoring
def _norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _cand(raw, ans):
    if isinstance(ans, str) and ans.strip():
        return ans
    c = re.split(r"\|?\s*Confidence:", str(raw), flags=re.IGNORECASE)[0]
    return c.replace("|", " ").strip()

def _score(raw, ans, gold):
    p, g = _norm(_cand(raw, ans)), _norm(gold)
    if not p:
        return 0
    if p == g:
        return 1
    if g and re.search(r"(?:^|\s)" + re.escape(g) + r"(?:\s|$)", p):
        return 1
    return 0

def load(path, label):
    d = pd.read_csv(path)
    d["ok"] = [_score(r.raw, r.answer, r.gold) for r in d.itertuples()]
    d["cond"] = d["degradation"] + "_s" + d["severity"].astype(str)
    d["arm_label"] = label
    return d

# --------------------------------------------------------------- metrics
def _fast_auroc(y, s):
    """Rank-based AUROC (Mann-Whitney U), handles ties. y in {0,1}."""
    n = len(y)
    npos = y.sum(); nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    r = 1
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (r + (r + (j - i))) / 2.0
        r += (j - i + 1)
        i = j + 1
    rank_full = np.empty(n, dtype=float)
    rank_full[order] = ranks
    sum_pos = rank_full[y == 1].sum()
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    if len(y) == 0:
        return np.nan
    return _fast_auroc(y, s)

def boot_auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    n = len(y)
    if n == 0 or y.sum() in (0, n):
        return (np.nan, np.nan, np.nan)
    pt = _fast_auroc(y, s)
    stats = []
    for _ in range(B_BOOT):
        idx = RNG.integers(0, n, n)
        a = _fast_auroc(y[idx], s[idx])
        if not np.isnan(a):
            stats.append(a)
    if not stats:
        return (pt, np.nan, np.nan)
    return (pt, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))

def boot_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    stats = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(B_BOOT)]
    return (float(x.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))

def risk_coverage(conf, ok):
    """Return coverage grid and selective accuracy sorted by descending confidence."""
    conf = np.asarray(conf, dtype=float); ok = np.asarray(ok)
    m = ~np.isnan(conf); conf, ok = conf[m], ok[m]
    order = np.argsort(-conf)
    ok_sorted = ok[order]
    n = len(ok_sorted)
    cov = np.arange(1, n + 1) / n
    sel_acc = np.cumsum(ok_sorted) / np.arange(1, n + 1)
    return cov, sel_acc

def aurc(conf, ok):
    """Area under the risk-coverage curve (risk = 1 - selective accuracy). Lower=better."""
    cov, sel = risk_coverage(conf, ok)
    risk = 1 - sel
    return float(np.trapz(risk, cov))

# --------------------------------------------------------------- per-condition table
def summary(d):
    rows = []
    for (fam, sev), g in d.groupby(["degradation", "severity"]):
        acc, alo, ahi = boot_mean(g["ok"].values)
        vi, vlo, vhi = boot_auroc(g["ok"].values, g["verbalized"].values)
        ii, ilo, ihi = boot_auroc(g["ok"].values, g["internal"].values)
        rows.append(dict(degradation=fam, severity=sev,
                         n=len(g), acc=acc, acc_lo=alo, acc_hi=ahi,
                         verb_parse=g["verbalized"].notna().mean(),
                         auroc_verb=vi, verb_lo=vlo, verb_hi=vhi,
                         auroc_int=ii, int_lo=ilo, int_hi=ihi,
                         aurc_int=aurc(g["internal"].values, g["ok"].values)))
    df = pd.DataFrame(rows)
    df["_o"] = df["degradation"].map({f: i for i, f in enumerate(FAM_ORDER)})
    return df.sort_values(["_o", "severity"]).drop(columns="_o").reset_index(drop=True)

# --------------------------------------------------------------- plots
def fig_scale_auroc(sB, sC, out):
    conds = sB["degradation"] + " s" + sB["severity"].astype(str)
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, sC["auroc_int"], "o-", label="7B internal", color="#1f77b4")
    ax.plot(x, sB["auroc_int"], "o--", label="2B internal", color="#7fb0d4")
    ax.plot(x, sC["auroc_verb"], "s-", label="7B verbalized", color="#d62728")
    ax.plot(x, sB["auroc_verb"], "s--", label="2B verbalized", color="#e69595")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("error-detection AUROC"); ax.set_ylim(0.35, 1.02)
    ax.set_title("Scale improves internal confidence far more than verbalized (both 4-bit)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def fig_quant_bars(mean_tbl, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    arms = list(mean_tbl.keys())
    x = np.arange(len(arms))
    acc = [mean_tbl[a]["acc"] for a in arms]
    ai = [mean_tbl[a]["int_auroc"] for a in arms]
    w = 0.38
    ax.bar(x - w/2, acc, w, label="accuracy", color="#4c72b0")
    ax.bar(x + w/2, ai, w, label="internal AUROC", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_ylim(0, 1.05); ax.set_ylabel("mean over 19 conditions")
    ax.set_title("Accuracy survives quantization; the confidence signal does not")
    for xi, (a, b) in enumerate(zip(acc, ai)):
        ax.text(xi - w/2, a + .01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + w/2, b + .01, f"{b:.2f}", ha="center", fontsize=8)
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def fig_risk_coverage(arms, out):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    styles = {"2B-fp16": ("#4c72b0", "-"), "2B-nf4": ("#55a868", "-"), "7B-nf4": ("#c44e52", "-")}
    for label, d in arms.items():
        cov, sel = risk_coverage(d["internal"].values, d["ok"].values)
        c, ls = styles.get(label, ("k", "-"))
        ax.plot(cov, sel, ls, color=c, label=f"{label} internal (AURC={aurc(d['internal'].values, d['ok'].values):.3f})")
    # one verbalized curve for contrast (7B)
    d = arms["7B-nf4"]
    cov, sel = risk_coverage(d["verbalized"].values, d["ok"].values)
    ax.plot(cov, sel, "--", color="gray", label="7B verbalized")
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("selective accuracy")
    ax.set_title("Risk-coverage: deferring by internal confidence")
    ax.set_xlim(0.3, 1.0); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def fig_lowlight(arms, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sev = [0, 1, 2, 3]
    for label, d in arms.items():
        accs = []
        for s in sev:
            sub = d[(d.degradation == "low_light") & (d.severity == s)] if s else d[d.degradation == "clean"]
            accs.append(sub["ok"].mean())
        ax.plot(sev, accs, "o-", label=f"{label} accuracy")
    ax.axhline(0.25, ls=":", color="red", lw=1, label="4-way chance")
    ax.set_xticks(sev); ax.set_xlabel("low-light severity (0 = clean)")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.03)
    ax.set_title("Low-light collapse is softened but not removed by scale")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--c", required=True); ap.add_argument("--out", default="paper2_out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    A = load(args.a, "2B-fp16"); B = load(args.b, "2B-nf4"); C = load(args.c, "7B-nf4")
    arms = {"2B-fp16": A, "2B-nf4": B, "7B-nf4": C}

    sA, sB, sC = summary(A), summary(B), summary(C)
    sA.to_csv(out / "summary_A_2B_fp16.csv", index=False)
    sB.to_csv(out / "summary_B_2B_nf4.csv", index=False)
    sC.to_csv(out / "summary_C_7B_nf4.csv", index=False)

    def arm_means(d, s):
        return dict(acc=d["ok"].mean(),
                    verb_parse=d["verbalized"].notna().mean(),
                    int_auroc=np.nanmean(s["auroc_int"]),
                    verb_auroc=np.nanmean(s["auroc_verb"]),
                    aurc_int=np.nanmean(s["aurc_int"]))
    means = {"2B-fp16": arm_means(A, sA), "2B-nf4": arm_means(B, sB), "7B-nf4": arm_means(C, sC)}

    fig_scale_auroc(sB, sC, out / "fig_scale_auroc.png")
    fig_quant_bars(means, out / "fig_quant_bars.png")
    fig_risk_coverage(arms, out / "fig_risk_coverage.png")
    fig_lowlight(arms, out / "fig_lowlight.png")

    headline = {
        "n_pred_total": int(len(A) + len(B) + len(C)),
        "n_per_arm": int(len(A)),
        "means": {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in means.items()},
        "scale_acc_gain_lowlight_s3": round(
            float(sC[(sC.degradation=="low_light")&(sC.severity==3)]["acc"].iloc[0]
                  - sB[(sB.degradation=="low_light")&(sB.severity==3)]["acc"].iloc[0]), 3),
    }
    (out / "headline_numbers.json").write_text(json.dumps(headline, indent=2))

    print(json.dumps(headline, indent=2))
    print("\nWrote summaries + 4 figures to", out)

if __name__ == "__main__":
    main()
