#!/usr/bin/env python3
"""
Cross-Model Terrain Comparison Analysis

Reads terrain_data.json from each model's directory and produces:
  1. Cliff detection per model per dimension
  2. Cross-model cliff position comparison
  3. Topology classification comparison
  4. Universality assessment
"""

import json, os, sys
import numpy as np
from pathlib import Path

BASE_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model")
CLIFF_THRESHOLD_Z = 2.0


def load_terrain(tag):
    path = BASE_DIR / tag / "terrain_data.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def detect_cliffs(sweep_data, metric="trigram_rep"):
    """Detect cliffs in a dimension's sweep data."""
    results = {}
    for dim, points in sweep_data.items():
        values = [p["value"] for p in points]
        metric_vals = []
        for p in points:
            qmetrics = [p["queries"][qid]["metrics"][metric] for qid in p["queries"]]
            metric_vals.append(float(np.mean(qmetrics)))

        steps = []
        for i in range(1, len(metric_vals)):
            steps.append(abs(metric_vals[i] - metric_vals[i-1]))

        mean_step = np.mean(steps) if steps else 0
        std_step = np.std(steps) if steps else 1

        cliffs = []
        for i, s in enumerate(steps):
            if std_step > 0:
                z = (s - mean_step) / std_step
            else:
                z = 0
            if z > CLIFF_THRESHOLD_Z:
                cliffs.append({
                    "from": values[i],
                    "to": values[i+1],
                    "delta": round(s, 4),
                    "z_score": round(z, 2),
                    "direction": "positive" if values[i+1] > 0 else "negative",
                })

        max_val = max(metric_vals)
        min_val = min(metric_vals)
        volatility = np.std(metric_vals)
        topology = "CLIFF" if cliffs else ("ROUGH" if volatility > 0.03 else "SMOOTH")

        results[dim] = {
            "topology": topology,
            "cliffs": cliffs,
            "max_value": round(max_val, 4),
            "min_value": round(min_val, 4),
            "volatility": round(volatility, 4),
            "range": round(max_val - min_val, 4),
        }
    return results


def main():
    tags = [d.name for d in BASE_DIR.iterdir()
            if d.is_dir() and (d / "terrain_data.json").exists()]
    tags.sort()

    if not tags:
        print("No terrain data found. Run experiments first.")
        print(f"Looking in: {BASE_DIR}")
        sys.exit(1)

    print(f"Found {len(tags)} model results: {', '.join(tags)}")
    print("=" * 80)

    all_results = {}
    model_meta = {}
    for tag in tags:
        data = load_terrain(tag)
        if data is None:
            continue
        meta = data.get("metadata", {})
        model_meta[tag] = {
            "model": meta.get("model", "?"),
            "type": meta.get("model_type", "?"),
            "layers": meta.get("n_layers", "?"),
            "hidden": meta.get("hidden_size", "?"),
        }
        cliffs = detect_cliffs(data["sweeps"])
        all_results[tag] = cliffs

    # ── Print model info ──
    print("\nModel Summary:")
    print(f"{'Tag':<25} {'Model Type':<15} {'Layers':<8} {'Hidden':<8}")
    print("-" * 60)
    for tag, meta in model_meta.items():
        print(f"{tag:<25} {meta['type']:<15} {str(meta['layers']):<8} {str(meta['hidden']):<8}")

    # ── Topology comparison ──
    dims = list(next(iter(all_results.values())).keys()) if all_results else []

    print(f"\n{'='*80}")
    print("TOPOLOGY CLASSIFICATION (per dimension per model)")
    print(f"{'='*80}")
    header = f"{'Dimension':<20}" + "".join(f"{t:<20}" for t in tags)
    print(header)
    print("-" * len(header))
    for dim in dims:
        row = f"{dim:<20}"
        for tag in tags:
            topo = all_results[tag][dim]["topology"]
            vol = all_results[tag][dim]["volatility"]
            row += f"{topo} (v={vol:.3f}){'':<5}"
        print(row)

    # ── Cliff positions ──
    print(f"\n{'='*80}")
    print("CLIFF LOCATIONS")
    print(f"{'='*80}")
    for dim in dims:
        has_cliff = any(all_results[t][dim]["cliffs"] for t in tags)
        if not has_cliff:
            continue
        print(f"\n  {dim}:")
        for tag in tags:
            cliffs = all_results[tag][dim]["cliffs"]
            if cliffs:
                for c in cliffs:
                    print(f"    {tag}: a={c['from']:+.1f}→{c['to']:+.1f}  "
                          f"Δ={c['delta']:.3f}  z={c['z_score']:.1f}σ  "
                          f"({c['direction']})")
            else:
                print(f"    {tag}: no cliff detected")

    # ── Universality assessment ──
    print(f"\n{'='*80}")
    print("UNIVERSALITY ASSESSMENT")
    print(f"{'='*80}")

    for dim in dims:
        topos = [all_results[t][dim]["topology"] for t in tags]
        unique_topos = set(topos)
        cliff_positions = []
        for tag in tags:
            for c in all_results[tag][dim]["cliffs"]:
                cliff_positions.append((tag, c["from"], c["to"]))

        if len(unique_topos) == 1:
            print(f"  {dim}: UNIVERSAL — all models show {topos[0]}")
        else:
            parts = ", ".join(f"{t}={all_results[t][dim]['topology']}" for t in tags)
            print(f"  {dim}: DIVERGENT — {parts}")

        if cliff_positions:
            positions = [(cp[1] + cp[2]) / 2 for cp in cliff_positions]
            if len(positions) > 1:
                spread = max(positions) - min(positions)
                print(f"    Cliff center spread: {spread:.1f} (positions: {[f'{p:.1f}' for p in positions]})")

    # ── Save comparison ──
    comparison = {
        "models": model_meta,
        "topologies": {dim: {tag: all_results[tag][dim] for tag in tags} for dim in dims},
        "universality": {},
    }
    for dim in dims:
        topos = [all_results[t][dim]["topology"] for t in tags]
        comparison["universality"][dim] = {
            "is_universal": len(set(topos)) == 1,
            "topologies": dict(zip(tags, topos)),
        }

    out_path = BASE_DIR / "cross_model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to {out_path}")


if __name__ == "__main__":
    main()
