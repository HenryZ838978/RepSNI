#!/usr/bin/env python3
"""Deep analysis of terrain map data — identify discontinuities and domain shifts."""
import json, math
import numpy as np

with open("/cache/zhangjing/repeng_terrain/terrain_data.json") as f:
    DATA = json.load(f)

def extract(dim, metric, query_id=None):
    points = DATA["sweeps"][dim]
    xs, ys = [], []
    for p in points:
        xs.append(p["value"])
        if query_id:
            ys.append(p["queries"][query_id]["metrics"].get(metric, 0))
        else:
            vals = [p["queries"][qid]["metrics"].get(metric, 0) for qid in p["queries"]]
            ys.append(np.mean(vals))
    return np.array(xs), np.array(ys)

print("=" * 80)
print("REPRESENTATION TERRAIN MAP — QUANTITATIVE ANALYSIS")
print("=" * 80)
print(f"Model: {DATA['metadata']['model']}")
print(f"Sweep: {DATA['metadata']['range'][0]} to {DATA['metadata']['range'][1]}, step={DATA['metadata']['step']}")
print(f"Queries: {len(DATA['metadata']['queries'])}")
print(f"Total points: {sum(len(DATA['sweeps'][d]) for d in DATA['sweeps'])} × 3 = {sum(len(DATA['sweeps'][d]) for d in DATA['sweeps']) * 3} generations")

metrics = ["avg_logprob", "trigram_rep", "cosine_sim_to_baseline"]
metric_names = {"avg_logprob": "Log-Probability", "trigram_rep": "Trigram Repetition", "cosine_sim_to_baseline": "Cosine Sim"}

print("\n" + "=" * 80)
print("SECTION 1: VOLATILITY RANKING (which dimensions are smooth vs. rough)")
print("=" * 80)

for metric in metrics:
    print(f"\n--- {metric_names[metric]} ---")
    dim_stats = []
    for dim in DATA["sweeps"]:
        xs, ys = extract(dim, metric)
        diffs = np.abs(np.diff(ys))
        dim_stats.append({
            "dim": dim,
            "range": float(ys.max() - ys.min()),
            "max_jump": float(diffs.max()),
            "mean_jump": float(diffs.mean()),
            "std": float(ys.std()),
            "jump_loc": f"{xs[diffs.argmax()]:.1f}→{xs[diffs.argmax()+1]:.1f}",
        })
    dim_stats.sort(key=lambda x: x["max_jump"], reverse=True)
    for s in dim_stats:
        bar = "█" * int(s["max_jump"] / dim_stats[0]["max_jump"] * 30)
        print(f"  {s['dim']:<20} range={s['range']:.4f}  max_jump={s['max_jump']:.4f} at {s['jump_loc']}  {bar}")

print("\n" + "=" * 80)
print("SECTION 2: CLIFF DETECTION (discrete jumps > 2σ above mean)")
print("=" * 80)

all_cliffs = []
for dim in DATA["sweeps"]:
    for metric in metrics:
        xs, ys = extract(dim, metric)
        diffs = np.abs(np.diff(ys))
        mean_d = diffs.mean()
        std_d = diffs.std()
        threshold = mean_d + 2 * std_d
        for i in range(len(diffs)):
            if diffs[i] > threshold:
                all_cliffs.append({
                    "dim": dim,
                    "metric": metric_names[metric],
                    "from": float(xs[i]),
                    "to": float(xs[i+1]),
                    "delta": float(diffs[i]),
                    "z_score": float((diffs[i] - mean_d) / std_d) if std_d > 0 else 0,
                    "direction": "↑" if ys[i+1] > ys[i] else "↓",
                    "before": float(ys[i]),
                    "after": float(ys[i+1]),
                })

all_cliffs.sort(key=lambda c: c["z_score"], reverse=True)
print(f"\nDetected {len(all_cliffs)} cliffs (>2σ):\n")
for c in all_cliffs[:20]:
    print(f"  [{c['dim']:<20}] {c['metric']:<15} α={c['from']:+.1f}→{c['to']:+.1f}  "
          f"Δ={c['delta']:.4f} ({c['direction']})  z={c['z_score']:.1f}  "
          f"({c['before']:.3f}→{c['after']:.3f})")

print("\n" + "=" * 80)
print("SECTION 3: DIMENSION TOPOLOGY CLASSIFICATION")
print("=" * 80)

for dim in DATA["sweeps"]:
    xs_rep, ys_rep = extract(dim, "trigram_rep")
    xs_cos, ys_cos = extract(dim, "cosine_sim_to_baseline")
    xs_lp, ys_lp = extract(dim, "avg_logprob")
    
    rep_range = ys_rep.max() - ys_rep.min()
    cos_range = ys_cos.max() - ys_cos.min()
    rep_max_jump = np.abs(np.diff(ys_rep)).max()
    cos_max_jump = np.abs(np.diff(ys_cos)).max()
    
    neg_rep = ys_rep[:len(ys_rep)//2].mean()
    pos_rep = ys_rep[len(ys_rep)//2:].mean()
    asymmetry = abs(neg_rep - pos_rep) / max(neg_rep, pos_rep, 0.001)
    
    if rep_max_jump > 0.15:
        topology = "CLIFF (sharp domain boundary)"
    elif rep_range > 0.1:
        topology = "ROUGH (moderate terrain variation)"
    elif asymmetry > 0.5:
        topology = "ASYMMETRIC (lopsided landscape)"
    else:
        topology = "SMOOTH (continuous gradient)"
    
    print(f"\n  {dim}:")
    print(f"    Topology: {topology}")
    print(f"    Rep range: {rep_range:.4f}  Max jump: {rep_max_jump:.4f}")
    print(f"    Cos range: {cos_range:.4f}  Max jump: {cos_max_jump:.4f}")
    print(f"    Asymmetry (neg vs pos rep): {asymmetry:.2f}")

print("\n" + "=" * 80)
print("SECTION 4: SAFE OPERATING ENVELOPE (per-dimension)")
print("=" * 80)

for dim in DATA["sweeps"]:
    xs, ys_rep = extract(dim, "trigram_rep")
    xs, ys_cos = extract(dim, "cosine_sim_to_baseline")
    
    baseline_rep = ys_rep[len(ys_rep)//2]
    baseline_cos = ys_cos[len(ys_cos)//2]
    
    safe_min, safe_max = xs[0], xs[-1]
    for i, x in enumerate(xs):
        rep_ok = ys_rep[i] < baseline_rep + 0.15
        cos_ok = ys_cos[i] > baseline_cos - 0.15
        if rep_ok and cos_ok:
            if x < 0 and x < safe_min: safe_min = x
            if x >= 0 and x > safe_max: safe_max = x
    
    safe_neg = xs[0]
    for i in range(len(xs)//2, -1, -1):
        if ys_rep[i] >= baseline_rep + 0.15 or ys_cos[i] <= baseline_cos - 0.15:
            safe_neg = xs[min(i+1, len(xs)-1)]
            break
    
    safe_pos = xs[-1]
    for i in range(len(xs)//2, len(xs)):
        if ys_rep[i] >= baseline_rep + 0.15 or ys_cos[i] <= baseline_cos - 0.15:
            safe_pos = xs[max(i-1, 0)]
            break
    
    print(f"  {dim:<20} safe range: [{safe_neg:+.1f}, {safe_pos:+.1f}]")

print("\n" + "=" * 80)
print("SECTION 5: SAMPLE OUTPUTS AT CLIFF POINTS")
print("=" * 80)

for dim in DATA["sweeps"]:
    xs, ys_rep = extract(dim, "trigram_rep")
    diffs = np.abs(np.diff(ys_rep))
    max_idx = diffs.argmax()
    if diffs[max_idx] < 0.1:
        continue
    
    before_val = xs[max_idx]
    after_val = xs[max_idx + 1]
    
    print(f"\n--- {dim}: cliff at {before_val:+.1f} → {after_val:+.1f} ---")
    points = DATA["sweeps"][dim]
    p_before = points[max_idx]
    p_after = points[max_idx + 1]
    
    for qid in ["factual", "news", "encourage"]:
        txt_before = p_before["queries"][qid]["metrics"]["cleaned_text"]
        txt_after = p_after["queries"][qid]["metrics"]["cleaned_text"]
        print(f"  [{qid}] α={before_val:+.1f}: {txt_before[:100]}")
        print(f"  [{qid}] α={after_val:+.1f}: {txt_after[:100]}")
        print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
