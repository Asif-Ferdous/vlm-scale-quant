"""
scale_quantization_experiment.py
=================================
Scale and quantization effects on uncertainty signals in vision-language models.

A practitioner with a 16 GB GPU can fit any of three configurations. This study
measures which one produces the most reliable confidence signal under image
degradation.

    Arm A   Qwen2-VL-2B-Instruct   fp16       small, full precision
    Arm B   Qwen2-VL-2B-Instruct   NF4 4-bit  small, quantized
    Arm C   Qwen2-VL-7B-Instruct   NF4 4-bit  large, quantized

    B vs C  ->  scale effect        (precision held constant)
    A vs B  ->  quantization effect (parameters held constant)

Fully self-contained: degradations, dataset construction, prompting, and metrics
are all defined here. No external repository or previously generated data is
required.

Free-tier Colab T4. Checkpoints after every condition; re-running skips completed
conditions, so a disconnect costs at most one condition.

RUN ORDER - do not skip:
    1. ARM = "smoke"   ~20 min   7B loads in NF4, probabilities sane
    2. ARM = "pilot"   ~45 min   template parse rates, run once per model size
    3. ARM = "B"       ~1.5 h    cheap; exercises the whole pipeline
    4. ARM = "A"       ~1.5 h
    5. ARM = "C"       ~4-6 h    only after a short arm has produced a sane CSV
"""

# ============================================================================
# CELL 1 - Install
# ============================================================================
# !pip install -q -U transformers accelerate bitsandbytes datasets

# ============================================================================
# CELL 2 - Mount Drive
# ============================================================================
# Must run BEFORE the config cell: config creates the output directory, and
# mounting onto a path that already contains files fails in Colab.
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================================
# CELL 3 - Config
# ============================================================================

import os, re, gc, io, json, math, random, hashlib, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---- CHOOSE THE ARM ---------------------------------------------------------
ARM = "smoke"      # "smoke" | "pilot" | "A" | "B" | "C"
# -----------------------------------------------------------------------------

ARMS = {
    "smoke": dict(model_id="Qwen/Qwen2-VL-7B-Instruct", quant="nf4",
                  n_items=5,   all_conditions=False, label="smoke"),
    "pilot": dict(model_id="Qwen/Qwen2-VL-7B-Instruct", quant="nf4",
                  n_items=20,  all_conditions=False, label="pilot_7B"),
    "A":     dict(model_id="Qwen/Qwen2-VL-2B-Instruct", quant="fp16",
                  n_items=100, all_conditions=True,  label="A_2B_fp16"),
    "B":     dict(model_id="Qwen/Qwen2-VL-2B-Instruct", quant="nf4",
                  n_items=100, all_conditions=True,  label="B_2B_nf4"),
    "C":     dict(model_id="Qwen/Qwen2-VL-7B-Instruct", quant="nf4",
                  n_items=100, all_conditions=True,  label="C_7B_nf4"),
}
CFG = ARMS[ARM]

SEED     = 0
MAX_NEW  = 32
ROOT     = Path("/content/drive/MyDrive/vlm_scale_study")
OUT_DIR  = ROOT / "checkpoints"
MANIFEST = ROOT / "item_manifest.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Arm {ARM}: {CFG['model_id']}  quant={CFG['quant']}  n={CFG['n_items']}")

# ============================================================================
# CELL 4 - Prompt templates
# ============================================================================

TEMPLATES = {
    # Few-shot, two-line. Worked example makes the format unambiguous.
    "fewshot": (
        "You must reply in exactly two lines.\n\n"
        "Example reply:\n"
        "Answer: pizza\n"
        "Confidence: 85\n\n"
        "Now do the same for this image.\n"
        "Question: {question}\n"
        "Options: {options}"
    ),
    # Direct instruction, no example.
    "direct": (
        "Question: {question}\n"
        "Options: {options}\n"
        "Answer the question, then rate your confidence from 0 to 100.\n"
        "Use exactly this format and nothing else:\n"
        "Answer: <option>\n"
        "Confidence: <number>"
    ),
    # Minimal scaffold.
    "minimal": (
        "{question}\n"
        "Choose one: {options}\n\n"
        "Reply with two lines only:\n"
        "Answer: (your choice)\n"
        "Confidence: (0-100)"
    ),
}

# Set per arm after the pilot. Default is the usual winner.
MAIN_TEMPLATE = "fewshot"
QUESTION = "What food is shown in this image?"

# ============================================================================
# CELL 5 - Parsing and answer matching
# ============================================================================

def parse_answer_and_confidence(text: str):
    """Extract answer string and verbalized confidence in [0,1]."""
    answer, conf = "", None
    m = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    if m:
        answer = m.group(1).strip().splitlines()[0].strip()
    m = re.search(r"Confidence:\s*([0-9]{1,3})", text, re.IGNORECASE)
    if m:
        conf = max(0.0, min(1.0, float(m.group(1)) / 100.0))
    return answer, conf


def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def answer_candidate(raw: str, parsed_answer: str) -> str:
    """
    The string to score for correctness.

    Smaller / quantized models frequently ignore the two-line template and reply
    with a bare label ("Beignets") instead of "Answer: beignets". Those replies
    are correct but have no "Answer:" prefix, so parse_answer_and_confidence
    returns an empty answer. Scoring only the parsed field would then mark a
    correct bare reply as wrong, and would penalise a model purely for not
    following the format --- a confound that would corrupt any cross-model
    comparison. We therefore fall back to the whole raw reply (minus any trailing
    "Confidence:" clause) when no explicit answer line was produced.

    Verbalized-confidence parsing is unaffected: a bare reply still yields no
    confidence value, which is the correct and separately reported behaviour.
    """
    if parsed_answer:
        return parsed_answer
    cand = re.split(r"\|?\s*Confidence:", str(raw), flags=re.IGNORECASE)[0]
    return cand.replace("|", " ").strip()


def match_answer(pred: str, gold: str, options: list):
    """Normalize -> exact -> whole-word gold -> unique containment."""
    p, g = _norm(pred), _norm(gold)
    if not p:
        return 0, "empty"
    if p == g:
        return 1, "exact"
    # gold appearing as a whole-word phrase inside a longer reply
    if g and re.search(r"(?:^|\s)" + re.escape(g) + r"(?:\s|$)", p):
        return 1, "gold_phrase"
    hits = [o for o in options if _norm(o) and _norm(o) in p]
    if len(hits) == 1:
        return (1 if _norm(hits[0]) == g else 0), "contains"
    return (0, "ambiguous") if hits else (0, "nomatch")

# ============================================================================
# CELL 6 - Degradations (self-contained)
# ============================================================================
import numpy as np
from PIL import Image

PARAMS = {
    "jpeg":        {1: 30,   2: 15,   3: 7},      # JPEG quality
    "motion_blur": {1: 2,    2: 4,    3: 7},      # horizontal box kernel radius
    "low_light":   {1: 0.5,  2: 0.3,  3: 0.15},   # brightness multiplier
    "glare":       {1: 1.6,  2: 2.2,  3: 3.0},    # brightness amplification
    "rotation":    {1: 5,    2: 12,   3: 20},     # degrees
    "resample":    {1: 0.5,  2: 0.3,  3: 0.15},   # downscale factor
}
FAMILIES = list(PARAMS.keys())
LOW_LIGHT_NOISE_SIGMA = 8.0     # Gaussian noise added with underexposure


def deg_jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def deg_motion_blur(img, r):
    a = np.asarray(img).astype(np.float32)
    k = int(r) * 2 + 1
    pad = np.pad(a, ((0, 0), (k // 2, k // 2), (0, 0)), mode="edge")
    out = np.zeros_like(a)
    for i in range(k):
        out += pad[:, i:i + a.shape[1], :]
    return Image.fromarray(np.clip(out / k, 0, 255).astype(np.uint8))


def deg_low_light(img, f, rng):
    a = np.asarray(img).astype(np.float32) * float(f)
    a += rng.normal(0, LOW_LIGHT_NOISE_SIGMA, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def deg_glare(img, f):
    a = np.asarray(img).astype(np.float32) * float(f)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def deg_rotation(img, deg):
    return img.rotate(float(deg), resample=Image.BILINEAR,
                      expand=False, fillcolor=(0, 0, 0))


def deg_resample(img, f):
    w, h = img.size
    small = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def apply_degradation(img, family, severity, item_idx=0):
    """Deterministic given (family, severity, item_idx)."""
    if family == "clean" or severity == 0:
        return img
    p = PARAMS[family][severity]
    if family == "jpeg":        return deg_jpeg(img, p)
    if family == "motion_blur": return deg_motion_blur(img, p)
    if family == "glare":       return deg_glare(img, p)
    if family == "rotation":    return deg_rotation(img, p)
    if family == "resample":    return deg_resample(img, p)
    if family == "low_light":
        rng = np.random.default_rng(SEED * 100003 + severity * 1009 + item_idx)
        return deg_low_light(img, p, rng)
    raise ValueError(family)


CONDITIONS = ([("clean", 0)] + [(f, s) for f in FAMILIES for s in (1, 2, 3)]
              if CFG["all_conditions"] else [("clean", 0)])
print(f"{len(CONDITIONS)} conditions")

# ============================================================================
# CELL 7 - Dataset + item manifest
# ============================================================================
from datasets import load_dataset


FOOD101_REPO  = "ethz/food101"   # bare "food101" is rejected by newer datasets libs
FOOD101_SPLIT = "validation"     # ethz/food101 calls the Food-101 test set "validation"


def build_items(n_items: int):
    ds = load_dataset(FOOD101_REPO, split=FOOD101_SPLIT, streaming=True)
    names = ds.features["label"].names

    imgs, labs = [], []
    for i, ex in enumerate(ds):
        if i >= n_items:
            break
        imgs.append(ex["image"].convert("RGB"))
        labs.append(ex["label"])

    rng = random.Random(SEED)
    items = []
    for idx, (img, lab) in enumerate(zip(imgs, labs)):
        gold = names[lab].replace("_", " ")
        pool = [c.replace("_", " ") for c in names if c.replace("_", " ") != gold]
        options = [gold] + rng.sample(pool, 3)
        rng.shuffle(options)
        items.append(dict(item=idx, image=img, gold=gold, options=options))
    return items


def manifest_of(items):
    payload = [{"item": it["item"], "gold": it["gold"], "options": it["options"]}
               for it in items]
    blob = json.dumps(payload, sort_keys=True)
    return {"n": len(items),
            "sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "items": payload}


def check_manifest(items):
    """
    HARD CHECK. All arms must score identical images with identical option sets,
    otherwise every cross-arm comparison is meaningless - and nothing about the
    output CSVs would look wrong. Fail loudly here instead.
    """
    new = manifest_of(items)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(new, indent=2))
        print(f"[OK] Wrote item manifest ({new['n']} items, "
              f"sha {new['sha256'][:12]}).")
        return
    old = json.loads(MANIFEST.read_text())
    if old["n"] != new["n"]:
        print(f"[WARN] Manifest has {old['n']} items, this arm uses {new['n']} "
              "(expected for smoke/pilot). Skipping strict check.")
        return
    if old["sha256"] == new["sha256"]:
        print(f"[OK] Item manifest matches (sha {new['sha256'][:12]}).")
        return
    bad = [i for i in range(new["n"])
           if old["items"][i]["gold"] != new["items"][i]["gold"]
           or old["items"][i]["options"] != new["items"][i]["options"]]
    raise SystemExit(
        f"ITEM MISALIGNMENT: {len(bad)}/{new['n']} items differ from the manifest.\n"
        f"  first bad index {bad[0]}:\n"
        f"    manifest gold={old['items'][bad[0]]['gold']!r} "
        f"options={old['items'][bad[0]]['options']}\n"
        f"    this run  gold={new['items'][bad[0]]['gold']!r} "
        f"options={new['items'][bad[0]]['options']}\n"
        "Arms would not be comparable. Fix build_items() or delete the manifest "
        "and re-run ALL arms."
    )


ITEMS = build_items(CFG["n_items"])
print(f"Loaded {len(ITEMS)} items. First gold: {ITEMS[0]['gold']!r}")
if CFG["all_conditions"]:
    check_manifest(ITEMS)

# ============================================================================
# CELL 8 - Model loading
# ============================================================================
import torch
from transformers import (Qwen2VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig)


def load_model(model_id: str, quant: str):
    kw = dict(device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,   # T4 has no bf16
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kw).eval()
    # Capping visual tokens is the main OOM lever on a 16 GB T4
    proc = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    return model, proc


MODEL, PROC = load_model(CFG["model_id"], CFG["quant"])
print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================================
# CELL 9 - Generation with answer-span internal confidence
# ============================================================================

@torch.no_grad()
def generate_with_confidence(image, prompt_text):
    """
    Returns (decoded_text, internal_conf, n_answer_tokens).

    internal_conf = mean p(token) over the ANSWER SPAN ONLY.

    The span restriction is load-bearing. Generation looks like
    "Answer: X \\n Confidence: 85"; averaging over all tokens would fold the
    verbalized confidence digits into the internal signal, making the paper's
    central comparison circular.
    """
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}]}]
    chat = PROC.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = PROC(text=[chat], images=[image], return_tensors="pt").to(MODEL.device)

    out = MODEL.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)

    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    tok = PROC.tokenizer

    probs, pieces = [], []
    for t, step_scores in enumerate(out.scores):
        if t >= len(gen_ids):
            break
        tid = gen_ids[t]
        probs.append(torch.softmax(step_scores[0].float(), dim=-1)[tid].item())
        pieces.append(tok.decode([tid], skip_special_tokens=True))

    full = "".join(pieces)

    offsets, pos = [], 0
    for pc in pieces:
        offsets.append((pos, pos + len(pc)))
        pos += len(pc)

    span = None
    m = re.search(r"Answer:\s*", full, re.IGNORECASE)
    if m:
        s = m.end()
        nl = full.find("\n", s)
        e = nl if nl != -1 else len(full)
        span = [i for i, (a, b) in enumerate(offsets) if b > s and a < e]

    sel = span if span else list(range(len(probs)))   # fallback: all tokens
    conf = float(sum(probs[i] for i in sel) / len(sel)) if sel else float("nan")

    return tok.decode(gen_ids, skip_special_tokens=True), conf, len(sel)

# ============================================================================
# CELL 10 - Main loop
# ============================================================================
import pandas as pd
from time import time


def run_condition(family, severity, template_key=None, tag=""):
    tkey = template_key or MAIN_TEMPLATE
    template = TEMPLATES[tkey]
    name = f"{CFG['label']}{tag}__{family}__s{severity}"
    path = OUT_DIR / f"{name}.csv"
    if path.exists():
        print(f"  skip {name} (done)")
        return pd.read_csv(path)

    t0, rows = time(), []
    for it in ITEMS:
        img = apply_degradation(it["image"], family, severity, it["item"])
        prompt = template.format(question=QUESTION,
                                 options=", ".join(it["options"]))
        try:
            raw, internal, ntok = generate_with_confidence(img, prompt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            raw, internal, ntok = generate_with_confidence(img, prompt)

        ans, verb = parse_answer_and_confidence(raw)
        cand = answer_candidate(raw, ans)   # falls back to bare reply if no "Answer:"
        correct, method = match_answer(cand, it["gold"], it["options"])

        rows.append(dict(
            arm=ARM, model=CFG["model_id"], precision=CFG["quant"],
            template=tkey, degradation=family, severity=severity,
            item=it["item"], gold=it["gold"], answer=ans, matched_on=cand,
            verbalized=verb, internal=internal, correct=correct,
            match_method=method, n_answer_tokens=ntok,
            raw=raw.replace("\n", " | "),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  {name}: acc={df['correct'].mean():.2f}  "
          f"int={df['internal'].mean():.3f}  "
          f"verb_parse={df['verbalized'].notna().mean():.0%}  [{time()-t0:.0f}s]")
    torch.cuda.empty_cache(); gc.collect()
    return df


if ARM == "pilot":
    print("\n=== Template pilot ===")
    print("Run this once per model size; set MAIN_TEMPLATE to the winner.")
    for tkey in TEMPLATES:
        d = run_condition("clean", 0, template_key=tkey, tag=f"__tmpl-{tkey}")
        print(f"  -> {tkey}: parse rate {d['verbalized'].notna().mean():.0%}")
else:
    print(f"\n=== Arm {ARM}: {len(CONDITIONS)} conditions "
          f"(template={MAIN_TEMPLATE}) ===")
    for fam, sev in CONDITIONS:
        run_condition(fam, sev)

# ============================================================================
# CELL 11 - Consolidate + sanity checks
# ============================================================================

parts = sorted(OUT_DIR.glob(f"{CFG['label']}*__*.csv"))
allp = None
if parts:
    allp = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    out = OUT_DIR / f"allpreds_{CFG['label']}.csv"
    allp.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(allp)} predictions)")
    print(allp.groupby(["degradation", "severity"])
              .agg(acc=("correct", "mean"),
                   verb=("verbalized", "mean"),
                   internal=("internal", "mean"),
                   parse=("verbalized", lambda s: s.notna().mean()))
              .round(3))

if ARM == "smoke" and allp is not None:
    print("\n=== SMOKE TEST CHECKS ===")
    ok = True
    if not allp["internal"].between(0, 1).all():
        print("  FAIL internal confidence outside [0,1]"); ok = False
    if allp["internal"].isna().any():
        print("  FAIL internal confidence has NaNs"); ok = False
    if allp["internal"].std() == 0:
        print("  FAIL internal confidence constant - span logic broken"); ok = False
    if allp["n_answer_tokens"].max() >= MAX_NEW:
        print("  WARN answer span = whole generation; check the 'Answer:' regex")
    pr = allp["verbalized"].notna().mean()
    if pr < 0.6:
        print(f"  WARN verbalized parse rate {pr:.0%} - try another template")
    print(f"  mean internal = {allp['internal'].mean():.3f}  (expect ~0.85-0.95)")
    print("  PASS - run the template pilot next" if ok else "  FIX BEFORE PROCEEDING")
