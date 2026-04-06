#!/usr/bin/env python3
"""
Critical Fluctuation Measurement

Tests the phase transition hypothesis by measuring output variance
at and near cliff points. True phase transitions show:
  1. Increased variance (critical fluctuations) near the transition
  2. Bimodal output distributions at the cliff
  3. Sharp order parameter discontinuity

Usage:
  python run_fluctuation.py \
    --model /cache/zhangjing/models/Qwen3-8B \
    --vector-dir /cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors \
    --terrain-data /cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/terrain_data.json \
    --gpu 5 --tag fluctuation-8b-bf16 --n-samples 20
"""

import argparse, json, os, re, sys, time
import numpy as np
import torch
from pathlib import Path

os.environ.setdefault("HF_HOME", "/cache/zhangjing/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/zhangjing/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector, ControlModel
from repeng.control import model_layer_list

MAX_TOKENS = 150
OUT_BASE = Path("/cache/zhangjing/repeng_terrain/cross_model")

QUERIES = [
    {"id": "factual", "text": "用两句话告诉我什么是transformer", "type": "knowledge"},
    {"id": "encourage", "text": "写一句鼓励人的话", "type": "emotion"},
]

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FEFF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
    "\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]+"
)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)
TAG_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")


def clean(text):
    text = THINK_RE.sub("", text)
    text = THINK_OPEN_RE.sub("", text)
    text = TAG_RE.sub("", text)
    return text.strip()


def compute_metrics(text, logprobs_list=None):
    cleaned = clean(text)
    avg_logprob = 0.0
    if logprobs_list:
        lps = [lp for lp in logprobs_list if lp is not None]
        avg_logprob = sum(lps) / len(lps) if lps else 0.0
    trigrams = [cleaned[i:i+3] for i in range(len(cleaned)-2)] if len(cleaned) > 2 else []
    trigram_rep = 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0
    return {
        "avg_logprob": round(avg_logprob, 4),
        "trigram_rep": round(trigram_rep, 4),
        "char_len": len(cleaned),
        "cleaned_text": cleaned[:200],
    }


def build_chat_prompt(tokenizer, query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate_stochastic(model, tokenizer, prompt, max_tokens, temperature=0.7):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=True, temperature=temperature,
            max_new_tokens=max_tokens, repetition_penalty=1.15,
            use_cache=True, return_dict_in_generate=True, output_scores=True)
    gen_ids = out.sequences[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    logprobs = []
    if hasattr(out, "scores") and out.scores:
        for step_idx, scores in enumerate(out.scores):
            if step_idx >= len(gen_ids):
                break
            log_softmax = torch.nn.functional.log_softmax(scores[0], dim=-1)
            logprobs.append(float(log_softmax[gen_ids[step_idx].item()].cpu()))
    return text, logprobs


def detect_cliffs(terrain_data):
    """Find cliff points from previous terrain sweep."""
    cliffs = []
    for dim, points in terrain_data["sweeps"].items():
        reps = []
        coefs = []
        for p in points:
            qmetrics = [p["queries"][qid]["metrics"]["trigram_rep"]
                        for qid in p["queries"]]
            reps.append(float(np.mean(qmetrics)))
            coefs.append(p["value"])

        steps = [abs(reps[i] - reps[i-1]) for i in range(1, len(reps))]
        if not steps:
            continue
        mean_step = np.mean(steps)
        std_step = np.std(steps)

        for i, s in enumerate(steps):
            z = (s - mean_step) / std_step if std_step > 0 else 0
            if z > 1.5:
                cliff_coeff = (coefs[i] + coefs[i+1]) / 2
                cliffs.append({
                    "dimension": dim,
                    "coeff": round(cliff_coeff, 2),
                    "z_score": round(z, 2),
                    "from_coeff": coefs[i],
                    "to_coeff": coefs[i+1],
                })
    return cliffs


def main():
    parser = argparse.ArgumentParser(description="Critical Fluctuation Measurement")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--vector-dir", required=True)
    parser.add_argument("--terrain-data", required=True,
                        help="Previous terrain_data.json for cliff detection")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of stochastic samples per point")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    out_dir = OUT_BASE / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = Path(args.vector_dir)

    print(f"{'='*70}")
    print(f"Critical Fluctuation Measurement")
    print(f"Model: {args.model} | GPU: {device}")
    print(f"Samples per point: {args.n_samples} | Temp: {args.temperature}")
    print(f"{'='*70}\n")

    t0 = time.time()

    # Load terrain data and detect cliffs
    print("Detecting cliffs from previous terrain data...")
    with open(args.terrain_data) as f:
        terrain = json.load(f)
    cliffs = detect_cliffs(terrain)
    print(f"  Found {len(cliffs)} cliff points:")
    for c in cliffs:
        print(f"    {c['dimension']} @ coeff={c['coeff']:+.2f} (z={c['z_score']:.1f}σ)")

    if not cliffs:
        print("No cliffs detected — nothing to measure. Exiting.")
        sys.exit(0)

    # Build measurement points: cliff ± fine steps, plus control points far from cliffs
    measure_points = []
    for cliff in cliffs:
        dim = cliff["dimension"]
        center = cliff["coeff"]
        for offset in [-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6]:
            measure_points.append({
                "dimension": dim,
                "coeff": round(center + offset, 2),
                "cliff_dist": abs(offset),
                "is_cliff": abs(offset) <= 0.1,
                "cliff_center": center,
            })
    # Control points: well away from any cliff
    all_dims = list(terrain["sweeps"].keys())
    cliff_zones = {(c["dimension"], c["coeff"]) for c in cliffs}
    for dim in all_dims[:2]:
        for coeff in [0.0, -1.0, 1.0]:
            if not any(d == dim and abs(coeff - c) < 0.8 for d, c in cliff_zones):
                measure_points.append({
                    "dimension": dim,
                    "coeff": coeff,
                    "cliff_dist": 99.0,
                    "is_cliff": False,
                    "cliff_center": None,
                })

    print(f"\n  Total measurement points: {len(measure_points)}")
    print(f"  Samples per point: {args.n_samples}")
    print(f"  Total generations: {len(measure_points) * args.n_samples * len(QUERIES)}\n")

    # Load model
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = dict(trust_remote_code=True)
    model_lower = args.model.lower()
    if "gptq" in model_lower:
        load_kwargs["device_map"] = device
    elif "awq" in model_lower:
        load_kwargs["device_map"] = device
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = device
    try:
        import flash_attn
        if "gptq" not in model_lower and "awq" not in model_lower:
            load_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    try:
        n_layers = len(model_layer_list(model))
    except ValueError:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            n_layers = len(model.model.layers)
            model.repeng_layers = model.model.layers
        else:
            raise

    layer_start = max(1, n_layers // 4)
    layer_end = min(n_layers - 1, n_layers * 3 // 4)
    ctrl_model = ControlModel(model, list(range(layer_start, layer_end)))

    # Load vectors
    print("Loading control vectors...")
    vectors = {}
    for pt in measure_points:
        dim = pt["dimension"]
        if dim not in vectors:
            vectors[dim] = ControlVector.import_gguf(str(vec_dir / f"{dim}.gguf"))
            print(f"  {dim}: loaded")

    # Run fluctuation measurement
    results = {
        "metadata": {
            "model": args.model,
            "tag": args.tag,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "cliffs_detected": cliffs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "measurements": [],
    }

    total_pts = len(measure_points)
    for pt_idx, pt in enumerate(measure_points):
        dim = pt["dimension"]
        coeff = pt["coeff"]

        print(f"\n[{pt_idx+1}/{total_pts}] {dim} @ {coeff:+.2f} "
              f"(cliff_dist={pt['cliff_dist']:.1f}, {'CLIFF' if pt['is_cliff'] else 'control'})")

        all_samples = {q["id"]: [] for q in QUERIES}

        for sample_idx in range(args.n_samples):
            ctrl_model.reset()
            if coeff != 0.0:
                ctrl_model.set_control(vectors[dim], coeff)

            for q in QUERIES:
                prompt = build_chat_prompt(tokenizer, q["text"])
                text, logprobs = generate_stochastic(
                    ctrl_model, tokenizer, prompt, args.max_tokens, args.temperature)
                metrics = compute_metrics(text, logprobs)
                all_samples[q["id"]].append(metrics)

            if (sample_idx + 1) % 5 == 0:
                reps = [s["trigram_rep"] for s in all_samples[QUERIES[0]["id"]]]
                print(f"    sample {sample_idx+1}/{args.n_samples}  "
                      f"rep_mean={np.mean(reps):.3f} rep_std={np.std(reps):.4f}")

        # Compute statistics
        measurement = {
            "dimension": dim,
            "coeff": coeff,
            "cliff_dist": pt["cliff_dist"],
            "is_cliff": pt["is_cliff"],
            "cliff_center": pt["cliff_center"],
            "stats": {},
        }

        for qid, samples in all_samples.items():
            reps = [s["trigram_rep"] for s in samples]
            logps = [s["avg_logprob"] for s in samples]
            lens = [s["char_len"] for s in samples]

            measurement["stats"][qid] = {
                "trigram_rep": {
                    "mean": round(float(np.mean(reps)), 4),
                    "std": round(float(np.std(reps)), 4),
                    "min": round(float(np.min(reps)), 4),
                    "max": round(float(np.max(reps)), 4),
                    "cv": round(float(np.std(reps) / (np.mean(reps) + 1e-8)), 4),
                },
                "avg_logprob": {
                    "mean": round(float(np.mean(logps)), 4),
                    "std": round(float(np.std(logps)), 4),
                },
                "char_len": {
                    "mean": round(float(np.mean(lens)), 1),
                    "std": round(float(np.std(lens)), 1),
                },
                "n_samples": len(samples),
                "sample_texts": [s["cleaned_text"][:100] for s in samples[:3]],
            }

        results["measurements"].append(measurement)

        out_path = out_dir / "fluctuation_data.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    ctrl_model.reset()

    # Summary analysis
    print(f"\n{'='*70}")
    print("FLUCTUATION SUMMARY")
    print(f"{'='*70}")

    cliff_stds = []
    control_stds = []
    for m in results["measurements"]:
        for qid, stats in m["stats"].items():
            std = stats["trigram_rep"]["std"]
            if m["is_cliff"]:
                cliff_stds.append(std)
            elif m["cliff_dist"] > 0.5:
                control_stds.append(std)

    if cliff_stds and control_stds:
        mean_cliff = np.mean(cliff_stds)
        mean_control = np.mean(control_stds)
        ratio = mean_cliff / (mean_control + 1e-8)
        print(f"\n  Cliff-zone std (trigram_rep):   {mean_cliff:.4f}")
        print(f"  Control-zone std (trigram_rep): {mean_control:.4f}")
        print(f"  Ratio (cliff/control):         {ratio:.2f}x")
        print(f"\n  {'PHASE TRANSITION SIGNAL DETECTED' if ratio > 1.5 else 'No clear phase transition signal'}")
        results["summary"] = {
            "cliff_std_mean": round(mean_cliff, 4),
            "control_std_mean": round(mean_control, 4),
            "fluctuation_ratio": round(ratio, 2),
            "phase_transition_signal": ratio > 1.5,
        }

    elapsed = time.time() - t0
    results["metadata"]["elapsed_seconds"] = round(elapsed, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDONE — {elapsed/60:.1f} minutes")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
