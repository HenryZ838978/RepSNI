#!/usr/bin/env python3
"""
SNI Pipeline — Semantic Nebula Imaging (语义星云成像)

End-to-end pipeline:
  vectors (GGUF) + terrain_data.json → point cloud → compact JSON → standalone HTML

Usage:
  # Single model
  python sni_pipeline.py --tag qwen3-8b-bf16

  # Batch all models with vectors
  python sni_pipeline.py --batch

  # Dual comparison
  python sni_pipeline.py --compare qwen3-8b-bf16 minicpm41-8b

  # Gallery: all models on one page
  python sni_pipeline.py --gallery
"""

import argparse, json, os, random, sys
import numpy as np
from pathlib import Path
from itertools import combinations

from repeng import ControlVector

# Configure these paths for your environment:
#   BASE = directory containing model subdirectories, each with vectors/ and terrain_data.json
#   OUT_DIR = output directory for generated HTML and JSON files
BASE = Path(os.environ.get("SNI_DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "cross_model"))
OUT_DIR = Path(os.environ.get("SNI_OUTPUT_DIR", Path(__file__).resolve().parent.parent / "sni_output"))
DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
DIM_SHORT = ["emo", "form", "crea", "conf", "empa"]
DIM_LABELS = ["Emotion", "Formality", "Creativity", "Confidence", "Empathy"]
DIM_COLORS_CSS = ["#ff4d4d", "#4db3ff", "#ffcc33", "#33ff66", "#cc55ff"]

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Generate point cloud from vectors
# ═══════════════════════════════════════════════════════════════════

def danger_color(rep, dominant_dim=-1, coeff_norm=0.0):
    if rep < 0.02:
        base = np.array([0.05, 0.15, 0.55])
    elif rep < 0.035:
        t = (rep - 0.02) / 0.015
        base = np.array([0.05+0.05*t, 0.15+0.25*t, 0.55+0.15*t])
    elif rep < 0.05:
        t = (rep - 0.035) / 0.015
        base = np.array([0.1+0.15*t, 0.4+0.3*t, 0.7-0.2*t])
    elif rep < 0.065:
        t = (rep - 0.05) / 0.015
        base = np.array([0.25+0.35*t, 0.7+0.1*t, 0.5-0.3*t])
    elif rep < 0.08:
        t = (rep - 0.065) / 0.015
        base = np.array([0.6+0.3*t, 0.8-0.1*t, 0.2-0.05*t])
    elif rep < 0.12:
        t = (rep - 0.08) / 0.04
        base = np.array([0.9+0.1*t, 0.7-0.4*t, 0.15-0.1*t])
    else:
        t = min(1.0, (rep - 0.12) / 0.15)
        base = np.array([1.0, 0.3-0.25*t, 0.05])

    dim_tints = [
        np.array([0.15, -0.05, -0.05]),
        np.array([-0.05, 0.0, 0.15]),
        np.array([0.1, 0.1, -0.05]),
        np.array([-0.05, 0.12, -0.02]),
        np.array([0.08, -0.02, 0.12]),
    ]
    if 0 <= dominant_dim < 5:
        tint_strength = min(0.5, coeff_norm / 4.0)
        base = base + dim_tints[dominant_dim] * tint_strength
    return np.clip(base, 0, 1).tolist()


def generate_pointcloud(tag, n_points=50000):
    """Generate SNI data from pre-trained vectors."""
    tag_dir = BASE / tag
    vec_dir = tag_dir / "vectors"
    if not vec_dir.exists():
        print(f"  [SKIP] {tag}: no vectors directory")
        return None

    gguf_files = list(vec_dir.glob("*.gguf"))
    if len(gguf_files) < 5:
        print(f"  [SKIP] {tag}: only {len(gguf_files)} vectors (need 5)")
        return None

    # Load vectors
    vectors = {}
    for dim in DIMS:
        vpath = vec_dir / f"{dim}.gguf"
        if not vpath.exists():
            print(f"  [SKIP] {tag}: missing {dim}.gguf")
            return None
        vectors[dim] = ControlVector.import_gguf(str(vpath))

    shared_layers = sorted(set.intersection(*[set(v.directions.keys()) for v in vectors.values()]))
    mid_layer = shared_layers[len(shared_layers) // 2]

    dim_vecs = {}
    for dim in DIMS:
        dim_vecs[dim] = vectors[dim].directions[mid_layer].astype(np.float32)
    hidden_size = len(dim_vecs[DIMS[0]])

    # Sample 5D space
    np.random.seed(42)
    points_5d = []
    N = n_points
    for _ in range(int(N * 0.4)):
        points_5d.append(np.random.uniform(-3.0, 3.0, 5))
    for _ in range(int(N * 0.2)):
        x = np.random.randn(5) * 1.2
        points_5d.append(np.clip(x, -3.0, 3.0))
    for radius in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for _ in range(int(N * 0.05)):
            x = np.random.randn(5)
            x = x / np.linalg.norm(x) * radius
            points_5d.append(np.clip(x, -3.0, 3.0))
    for _ in range(int(N * 0.1)):
        x = np.zeros(5)
        i, j = np.random.choice(5, 2, replace=False)
        x[i] = np.random.uniform(-3.0, 3.0)
        x[j] = np.random.uniform(-3.0, 3.0)
        points_5d.append(x)
    for dim_idx in range(5):
        for val in np.arange(-3.0, 3.01, 0.05):
            x = np.zeros(5)
            x[dim_idx] = val
            points_5d.append(x)
    points_5d = np.array(points_5d)

    # PCA projection
    combined = np.zeros((len(points_5d), hidden_size), dtype=np.float32)
    for i, dim in enumerate(DIMS):
        combined += np.outer(points_5d[:, i], dim_vecs[dim])
    mean_vec = combined.mean(axis=0)
    centered = combined - mean_vec
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    top3 = eigenvectors[:, -3:][:, ::-1]
    proj_3d = centered @ top3

    for ax in range(3):
        mn, mx = proj_3d[:, ax].min(), proj_3d[:, ax].max()
        if mx - mn > 0:
            proj_3d[:, ax] = 2 * (proj_3d[:, ax] - mn) / (mx - mn) - 1

    var_explained = eigenvalues[-3:][::-1] / eigenvalues.sum()

    # Load manifold model if available
    beta = None
    manifold_path = BASE / "manifold" / "manifold_model.json"
    terrain_path = tag_dir / "terrain_data.json"
    if manifold_path.exists() and "qwen3-8b" in tag:
        model_data = json.load(open(manifold_path))
        beta = np.array(model_data["beta"])
    elif terrain_path.exists():
        terrain_data = json.load(open(terrain_path))
        X_list, y_list = [], []
        for dim_idx, dim in enumerate(DIMS):
            for pt in terrain_data.get("sweeps", {}).get(dim, []):
                x5 = [0.0] * 5
                x5[dim_idx] = pt["value"]
                features = [1.0] + x5 + [xi**2 for xi in x5]
                for i, j in combinations(range(5), 2):
                    features.append(x5[i] * x5[j])
                X_list.append(features)
                # Handle both formats: direct metrics vs nested queries
                if "metrics" in pt:
                    y_list.append(pt["metrics"]["trigram_rep"])
                elif "queries" in pt:
                    reps = [qd["metrics"]["trigram_rep"] for qd in pt["queries"].values()
                            if isinstance(qd, dict) and "metrics" in qd]
                    y_list.append(np.mean(reps) if reps else 0.0)
                else:
                    y_list.append(0.0)
        if len(X_list) > 10:
            X = np.array(X_list)
            y = np.array(y_list)
            beta = np.linalg.solve(X.T @ X + 0.01 * np.eye(X.shape[1]), X.T @ y)

    # Compute danger colors
    cloud = []
    for idx in range(len(points_5d)):
        if beta is not None:
            x5 = list(points_5d[idx])
            features = [1.0] + x5 + [xi**2 for xi in x5]
            for i, j in combinations(range(5), 2):
                features.append(x5[i] * x5[j])
            rep = max(0, float(np.dot(beta, features)))
        else:
            rep = 0.04
        abs_c = np.abs(points_5d[idx])
        dominant = int(np.argmax(abs_c))
        cn = float(np.linalg.norm(points_5d[idx]))
        color = danger_color(rep, dominant, cn)
        cloud.append({
            "x": round(float(proj_3d[idx, 0]), 3),
            "y": round(float(proj_3d[idx, 1]), 3),
            "z": round(float(proj_3d[idx, 2]), 3),
            "r": round(color[0], 2), "g": round(color[1], 2), "b": round(color[2], 2),
            "rep": round(float(rep), 4), "dom": dominant,
        })

    # Build paths from terrain data
    paths = {}
    terrain_data = json.load(open(terrain_path)) if terrain_path.exists() else {}
    for dim_idx, dim in enumerate(DIMS):
        path_pts = []
        for coeff in np.arange(-3.0, 3.01, 0.2):
            x5 = np.zeros(5); x5[dim_idx] = coeff
            c_single = np.zeros(hidden_size, dtype=np.float32)
            for i, d in enumerate(DIMS):
                c_single += x5[i] * dim_vecs[d]
            p3d = (c_single - mean_vec) @ top3
            for ax in range(3):
                mn, mx_ = proj_3d[:, ax].min(), proj_3d[:, ax].max()
                rng = mx_ - mn if mx_ - mn > 0 else 1
                p3d[ax] = 2 * (p3d[ax] - mn) / rng - 1
            if beta is not None:
                x5l = list(x5)
                f = [1.0] + x5l + [xi**2 for xi in x5l]
                for i, j in combinations(range(5), 2):
                    f.append(x5l[i] * x5l[j])
                rep = max(0, float(np.dot(beta, f)))
            else:
                rep = 0.04
            path_pts.append({"x": round(float(p3d[0]),3), "y": round(float(p3d[1]),3),
                             "z": round(float(p3d[2]),3), "rep": round(float(rep),4)})
        paths[dim] = {"up": path_pts}

    # Build markers from terrain sweep
    markers = []
    for dim_idx, dim in enumerate(DIMS):
        for pt in terrain_data.get("sweeps", {}).get(dim, []):
            if "metrics" in pt:
                rep = pt["metrics"]["trigram_rep"]
            elif "queries" in pt:
                reps = [qd["metrics"]["trigram_rep"] for qd in pt["queries"].values()
                        if isinstance(qd, dict) and "metrics" in qd]
                rep = float(np.mean(reps)) if reps else 0.0
            else:
                rep = 0.0
            x5 = np.zeros(5); x5[dim_idx] = pt["value"]
            c_single = np.zeros(hidden_size, dtype=np.float32)
            for i, d in enumerate(DIMS):
                c_single += x5[i] * dim_vecs[d]
            p3d = (c_single - mean_vec) @ top3
            for ax in range(3):
                mn, mx_ = proj_3d[:, ax].min(), proj_3d[:, ax].max()
                rng = mx_ - mn if mx_ - mn > 0 else 1
                p3d[ax] = 2 * (p3d[ax] - mn) / rng - 1
            markers.append({
                "x": round(float(p3d[0]),3), "y": round(float(p3d[1]),3),
                "z": round(float(p3d[2]),3), "rep": round(float(rep),3),
                "dim": dim, "coeff": round(float(pt["value"]),1),
                "cliff": bool(rep > 0.08),
            })

    # Compute structural metrics
    cosine_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            ci = dim_vecs[DIMS[i]] / np.linalg.norm(dim_vecs[DIMS[i]])
            cj = dim_vecs[DIMS[j]] / np.linalg.norm(dim_vecs[DIMS[j]])
            cosine_matrix[i, j] = float(np.dot(ci, cj))

    pc1_ratio = float(var_explained[0] / var_explained[1]) if var_explained[1] > 0 else 999

    # Structural assessment
    if pc1_ratio > 5:
        structure = "Compressed · Collapse-prone"
        health = "fragile"
    elif pc1_ratio > 3:
        structure = "Moderately concentrated"
        health = "caution"
    else:
        structure = "Multi-dimensional · Stable"
        health = "healthy"

    n_cliffs = sum(1 for m in markers if m["cliff"])
    avg_rep = np.mean([c["rep"] for c in cloud])
    max_rep = max(c["rep"] for c in cloud)

    metadata = {
        "model": tag,
        "hidden_size": hidden_size,
        "projection_layer": mid_layer,
        "n_layers": len(shared_layers),
        "variance_explained": [round(float(v), 4) for v in var_explained],
        "variance_total": round(float(var_explained.sum()), 4),
        "pc1_ratio": round(pc1_ratio, 1),
        "n_cloud_points": len(cloud),
        "n_markers": len(markers),
        "n_cliffs": n_cliffs,
        "avg_rep": round(float(avg_rep), 4),
        "max_rep": round(float(max_rep), 4),
        "structure": structure,
        "health": health,
        "dims": DIMS,
        "cosine_matrix": [[round(cosine_matrix[i,j], 3) for j in range(5)] for i in range(5)],
    }

    return {
        "metadata": metadata,
        "cloud": cloud,
        "markers": markers,
        "paths": paths,
    }


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Compact JSON for inline embedding
# ═══════════════════════════════════════════════════════════════════

def compact_data(full_data, max_points=8000):
    """Stratified downsample + array-of-struct → struct-of-array."""
    cloud = full_data["cloud"]
    random.seed(42)

    bins = [
        ("deep_safe",  lambda p: p["rep"] < 0.03, 2000),
        ("safe",       lambda p: 0.03 <= p["rep"] < 0.05, 2500),
        ("caution",    lambda p: 0.05 <= p["rep"] < 0.065, 2500),
        ("warning",    lambda p: 0.065 <= p["rep"] < 0.08, 1500),
        ("danger",     lambda p: p["rep"] >= 0.08, 99999),
    ]
    sampled = []
    for _, pred, limit in bins:
        group = [p for p in cloud if pred(p)]
        sampled.extend(random.sample(group, min(limit, len(group))))
    random.shuffle(sampled)
    if len(sampled) > max_points:
        sampled = random.sample(sampled, max_points)

    cx = [p["x"] for p in sampled]
    cy = [p["y"] for p in sampled]
    cz = [p["z"] for p in sampled]
    cr = [p["r"] for p in sampled]
    cg = [p["g"] for p in sampled]
    cb = [p["b"] for p in sampled]
    crep = [p["rep"] for p in sampled]

    cpaths = {}
    for dim, pd in full_data.get("paths", {}).items():
        pts = pd["up"]
        cpaths[dim] = {
            "x": [p["x"] for p in pts],
            "y": [p["y"] for p in pts],
            "z": [p["z"] for p in pts],
            "rep": [p["rep"] for p in pts],
        }

    markers = full_data.get("markers", [])
    mx = [m["x"] for m in markers]
    my = [m["y"] for m in markers]
    mz = [m["z"] for m in markers]
    mrep = [m["rep"] for m in markers]
    mdim = [m["dim"][:3] for m in markers]
    mcliff = [1 if m.get("cliff") else 0 for m in markers]

    return {
        "meta": full_data["metadata"],
        "c": {"x": cx, "y": cy, "z": cz, "r": cr, "g": cg, "b": cb, "rep": crep},
        "paths": cpaths,
        "m": {"x": mx, "y": my, "z": mz, "rep": mrep, "dim": mdim, "cliff": mcliff},
        "env": {"x": [], "y": [], "z": []},
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════
# STEP 3: Build HTML from template + data
# ═══════════════════════════════════════════════════════════════════

SINGLE_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SNI — {TITLE}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Instrument+Sans:wght@400;600;700&display=swap');
:root {
  --bg: #060610; --fg: #c8ccd0; --muted: #556; --accent: #4df;
  --safe: #4d8; --warn: #f84; --danger: #f44;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font-family:'Instrument Sans',sans-serif; overflow:hidden; height:100vh; }
#scene { position:fixed; inset:0; z-index:1; }

.panel { position:fixed; z-index:20; pointer-events:none; }
.panel-tl { top:20px; left:20px; max-width:320px; }
.panel-tr { top:20px; right:20px; max-width:260px; text-align:right; }
.panel-bl { bottom:20px; left:20px; }
.panel-br { bottom:20px; right:20px; text-align:right; }

.title {
  font-family:'JetBrains Mono',monospace;
  font-size:24px; font-weight:700; letter-spacing:3px;
}
.subtitle { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted); margin-top:4px; letter-spacing:2px; }
.subtitle-cn { font-size:12px; color:#667; margin-top:6px; }

.stat-box { margin-top:14px; }
.stat-row { font-family:'JetBrains Mono',monospace; font-size:11px; color:#889; margin-top:5px; }
.stat-val { color:var(--accent); font-weight:700; }
.stat-safe { color:var(--safe); }
.stat-warn { color:var(--warn); }
.stat-danger { color:var(--danger); }

.analysis-box {
  margin-top:16px; padding:12px 14px; border-radius:8px;
  background:rgba(10,10,20,0.85); border:1px solid #1a1a2e;
}
.analysis-title { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:2px; color:var(--muted); margin-bottom:8px; }
.analysis-row { font-size:11px; color:#99a; margin-top:4px; line-height:1.5; }
.analysis-row .hl { color:var(--accent); font-weight:600; }

.cosine-grid {
  display:grid; grid-template-columns:repeat(6,1fr); gap:1px; margin-top:8px;
  font-family:'JetBrains Mono',monospace; font-size:8px; text-align:center;
}
.cosine-grid .hdr { color:var(--muted); font-weight:700; }
.cosine-grid .cell { padding:2px 0; border-radius:2px; }

.grad-bar { width:180px; height:6px; border-radius:3px;
  background:linear-gradient(90deg,#0a1a44,#1a4488,#2a8866,#99cc33,#cc8822,#cc3322); margin-top:4px; }
.grad-labels { display:flex; justify-content:space-between; font-size:8px; color:#445; margin-top:2px; width:180px;
  font-family:'JetBrains Mono',monospace; }

.dim-legend { margin-top:12px; }
.dim-item { display:inline-flex; align-items:center; margin-right:10px; margin-top:4px; font-size:10px; color:#889; }
.dim-dot { width:8px; height:8px; border-radius:50%; margin-right:4px; }
</style>
</head><body>
<div id="scene"></div>

<div class="panel panel-tl">
  <div class="title">{MODEL_NAME}</div>
  <div class="subtitle">SEMANTIC NEBULA IMAGING · SNI</div>
  <div class="subtitle-cn">语义星云成像</div>
  <div class="stat-box">
    <div class="stat-row">Layers: <span class="stat-val">{N_LAYERS}</span> · Hidden: <span class="stat-val">{HIDDEN_SIZE}</span> · Proj layer: <span class="stat-val">{PROJ_LAYER}</span></div>
    <div class="stat-row">PCA variance: <span class="stat-val">{PCA_TOTAL}%</span> ({PCA_DETAIL})</div>
    <div class="stat-row">PC1:PC2 ratio: <span class="stat-val">{PC_RATIO}</span> · Structure: <span class="{HEALTH_CLASS}">{STRUCTURE}</span></div>
    <div class="stat-row">Points: <span class="stat-val">{N_POINTS}</span> · Markers: <span class="stat-val">{N_MARKERS}</span> · Cliffs: <span class="{CLIFF_CLASS}">{N_CLIFFS}</span></div>
  </div>
</div>

<div class="panel panel-tr">
  <div class="analysis-box">
    <div class="analysis-title">▸ STRUCTURAL ANALYSIS</div>
    {ANALYSIS_ROWS}
  </div>
  <div class="analysis-box" style="margin-top:8px">
    <div class="analysis-title">▸ DIMENSION COSINE SIMILARITY</div>
    <div class="cosine-grid">
      <div class="hdr"></div>{COSINE_HEADERS}
      {COSINE_ROWS}
    </div>
  </div>
</div>

<div class="panel panel-bl">
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted)">DANGER LEVEL (trigram_rep)</div>
  <div class="grad-bar"></div>
  <div class="grad-labels"><span>0.00</span><span>0.03</span><span>0.06</span><span>0.10</span><span>0.25+</span></div>
  <div class="dim-legend">{DIM_LEGEND}</div>
</div>

<div class="panel panel-br">
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted)">
    Projection: PCA(hidden_states[{PROJ_LAYER}]) → 3D<br>
    5D personality coefficient space [-3, +3]
  </div>
</div>

<script type="importmap">{"imports":{"three":"https://esm.sh/three@0.162.0","three/addons/":"https://esm.sh/three@0.162.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const D = {INLINE_DATA};
const c = D.c, n = c.x.length;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060610);
scene.fog = new THREE.FogExp2(0x060610, 0.06);

const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.01, 100);
camera.position.set(1.8, 1.2, 1.8);

const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.getElementById('scene').appendChild(renderer.domElement);

const ctrl = new OrbitControls(camera, renderer.domElement);
ctrl.enableDamping = true; ctrl.dampingFactor = 0.05;
ctrl.autoRotate = true; ctrl.autoRotateSpeed = 0.4;
ctrl.maxDistance = 5; ctrl.minDistance = 0.3;

scene.add(new THREE.AmbientLight(0x223344, 0.3));
const grid = new THREE.GridHelper(3, 15, 0x0a0a1a, 0x0a0a1a);
grid.position.y = -1.1; scene.add(grid);

const pos = new Float32Array(n*3), col = new Float32Array(n*3), siz = new Float32Array(n);
for (let i=0;i<n;i++) {
  pos[i*3]=c.x[i]; pos[i*3+1]=c.y[i]; pos[i*3+2]=c.z[i];
  col[i*3]=c.r[i]; col[i*3+1]=c.g[i]; col[i*3+2]=c.b[i];
  siz[i] = c.rep[i]>0.08?2.0:(c.rep[i]>0.05?1.2:0.8);
}
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
geo.setAttribute('color', new THREE.BufferAttribute(col,3));
geo.setAttribute('size', new THREE.BufferAttribute(siz,1));

const mat = new THREE.ShaderMaterial({
  uniforms:{uTime:{value:0}},
  vertexShader:`attribute float size; varying vec3 vColor; varying float vAlpha; uniform float uTime;
    void main(){vColor=color;vAlpha=1.0;vec4 mv=modelViewMatrix*vec4(position,1.0);
    gl_PointSize=size*(100.0/-mv.z);gl_Position=projectionMatrix*mv;}`,
  fragmentShader:`varying vec3 vColor;varying float vAlpha;
    void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;
    float core=smoothstep(0.5,0.05,d);float glow=exp(-d*10.0)*0.5+core*0.5;
    gl_FragColor=vec4(vColor*0.65,vAlpha*glow*0.3);}`,
  vertexColors:true, transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
});
scene.add(new THREE.Points(geo,mat));

const dimCols=[0xff4d4d,0x4db3ff,0xffcc33,0x33ff66,0xcc55ff];
let di=0;
for(const[dim,pd]of Object.entries(D.paths||{})){
  const pts=pd.x.map((x,i)=>new THREE.Vector3(x,pd.y[i],pd.z[i]));
  if(pts.length<2){di++;continue;}
  const lg=new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(lg,new THREE.LineBasicMaterial({color:dimCols[di%5],transparent:true,opacity:0.5})));
  di++;
}

// Markers
if(D.m&&D.m.x.length>0){
  const mn=D.m.x.length,mp=new Float32Array(mn*3),mc=new Float32Array(mn*3),ms=new Float32Array(mn);
  const dimMap={emo:0,for:1,cre:2,con:3,emp:4};
  for(let i=0;i<mn;i++){
    mp[i*3]=D.m.x[i];mp[i*3+1]=D.m.y[i];mp[i*3+2]=D.m.z[i];
    if(D.m.cliff[i]){mc[i*3]=1;mc[i*3+1]=0.15;mc[i*3+2]=0.1;ms[i]=5;}
    else{const ci=dimCols[dimMap[D.m.dim[i]]||0];const tc=new THREE.Color(ci);mc[i*3]=tc.r;mc[i*3+1]=tc.g;mc[i*3+2]=tc.b;ms[i]=3;}
  }
  const mg=new THREE.BufferGeometry();
  mg.setAttribute('position',new THREE.BufferAttribute(mp,3));
  mg.setAttribute('color',new THREE.BufferAttribute(mc,3));
  mg.setAttribute('size',new THREE.BufferAttribute(ms,1));
  const mm=new THREE.ShaderMaterial({
    vertexShader:`attribute float size;varying vec3 vC;void main(){vC=color;vec4 mv=modelViewMatrix*vec4(position,1.0);
      gl_PointSize=size*(120.0/-mv.z);gl_Position=projectionMatrix*mv;}`,
    fragmentShader:`varying vec3 vC;void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;
      float g=exp(-d*6.0);gl_FragColor=vec4(vC*g,g*0.8);}`,
    vertexColors:true,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,
  });
  scene.add(new THREE.Points(mg,mm));
}

const clock=new THREE.Clock();
(function animate(){requestAnimationFrame(animate);ctrl.update();
  mat.uniforms.uTime.value=clock.getElapsedTime();renderer.render(scene,camera);})();

addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
</script></body></html>'''


def build_analysis_rows(meta):
    """Generate structural analysis text from metadata."""
    rows = []
    ve = meta["variance_explained"]
    total = meta["variance_total"]
    pc_ratio = meta["pc1_ratio"]

    if pc_ratio > 5:
        rows.append(f'<div class="analysis-row">PC1 dominates at <span class="hl">{ve[0]*100:.1f}%</span> — representation space is highly compressed along one axis</div>')
        rows.append(f'<div class="analysis-row">PC1:PC2 = <span class="hl">{pc_ratio:.1f}:1</span> — personality dimensions are nearly collinear → <span style="color:var(--danger)">fragile</span></div>')
    elif pc_ratio > 3:
        rows.append(f'<div class="analysis-row">PC1 at <span class="hl">{ve[0]*100:.1f}%</span> with moderate concentration. Space is somewhat flattened.</div>')
        rows.append(f'<div class="analysis-row">PC1:PC2 = <span class="hl">{pc_ratio:.1f}:1</span> — moderate risk of dimensional interference</div>')
    else:
        rows.append(f'<div class="analysis-row">Balanced distribution: PC1 <span class="hl">{ve[0]*100:.1f}%</span> · PC2 <span class="hl">{ve[1]*100:.1f}%</span> · PC3 <span class="hl">{ve[2]*100:.1f}%</span></div>')
        rows.append(f'<div class="analysis-row">PC1:PC2 = <span class="hl">{pc_ratio:.1f}:1</span> — dimensions are well-separated → <span style="color:var(--safe)">stable</span></div>')

    rows.append(f'<div class="analysis-row">Total 3D capture: <span class="hl">{total*100:.1f}%</span> of representation variance</div>')

    if meta["n_cliffs"] > 0:
        rows.append(f'<div class="analysis-row" style="color:var(--danger)">⚠ {meta["n_cliffs"]} cliff points detected (trigram_rep > 0.08)</div>')
    else:
        rows.append(f'<div class="analysis-row">Mean danger: <span class="hl">{meta["avg_rep"]:.3f}</span> · Max: <span class="hl">{meta["max_rep"]:.3f}</span></div>')

    return "\n    ".join(rows)


def build_cosine_html(meta):
    """Build cosine similarity grid HTML."""
    matrix = meta.get("cosine_matrix", [[1]*5]*5)
    headers = "".join(f'<div class="hdr">{s}</div>' for s in DIM_SHORT)
    rows_html = ""
    for i in range(5):
        rows_html += f'<div class="hdr">{DIM_SHORT[i]}</div>'
        for j in range(5):
            v = matrix[i][j]
            if i == j:
                bg = "rgba(77,221,255,0.15)"
            elif abs(v) > 0.3:
                bg = f"rgba(255,136,68,{min(abs(v), 0.5)})"
            elif abs(v) > 0.1:
                bg = f"rgba(255,204,51,{min(abs(v), 0.3)})"
            else:
                bg = "rgba(40,40,60,0.5)"
            rows_html += f'<div class="cell" style="background:{bg}">{v:.2f}</div>'
    return headers, rows_html


def build_single_html(tag, compact, metadata):
    """Build self-contained HTML for a single model."""
    ve = metadata["variance_explained"]
    pca_detail = " + ".join(f"{v*100:.1f}" for v in ve)

    health_class = {"healthy": "stat-safe", "caution": "stat-warn", "fragile": "stat-danger"}.get(metadata["health"], "stat-val")
    cliff_class = "stat-danger" if metadata["n_cliffs"] > 0 else "stat-safe"

    analysis = build_analysis_rows(metadata)
    cos_headers, cos_rows = build_cosine_html(metadata)

    dim_legend = ""
    for i, label in enumerate(DIM_LABELS):
        dim_legend += f'<span class="dim-item"><span class="dim-dot" style="background:{DIM_COLORS_CSS[i]}"></span>{label}</span>'

    inline = json.dumps(compact, separators=(",", ":"), cls=NumpyEncoder)

    html = SINGLE_HTML_TEMPLATE
    replacements = {
        "{TITLE}": tag,
        "{MODEL_NAME}": tag.upper().replace("-", " · "),
        "{N_LAYERS}": str(metadata["n_layers"]),
        "{HIDDEN_SIZE}": str(metadata["hidden_size"]),
        "{PROJ_LAYER}": str(metadata["projection_layer"]),
        "{PCA_TOTAL}": f"{metadata['variance_total']*100:.1f}",
        "{PCA_DETAIL}": pca_detail,
        "{PC_RATIO}": f"{metadata['pc1_ratio']:.1f}:1",
        "{STRUCTURE}": metadata["structure"],
        "{HEALTH_CLASS}": health_class,
        "{N_POINTS}": f"{len(compact['c']['x']):,}",
        "{N_MARKERS}": str(metadata["n_markers"]),
        "{N_CLIFFS}": str(metadata["n_cliffs"]),
        "{CLIFF_CLASS}": cliff_class,
        "{ANALYSIS_ROWS}": analysis,
        "{COSINE_HEADERS}": cos_headers,
        "{COSINE_ROWS}": cos_rows,
        "{DIM_LEGEND}": dim_legend,
        "{INLINE_DATA}": inline,
    }
    for k, v in replacements.items():
        html = html.replace(k, v)
    return html


# ═══════════════════════════════════════════════════════════════════
# MAIN: CLI
# ═══════════════════════════════════════════════════════════════════

def process_tag(tag, n_points=50000, force=False):
    """Full pipeline for one tag."""
    out_html = OUT_DIR / f"sni_{tag}.html"
    out_json = OUT_DIR / f"sni_{tag}.json"

    if out_html.exists() and not force:
        print(f"  [{tag}] already exists, skip (use --force to rebuild)")
        return out_html

    print(f"  [{tag}] Generating point cloud...")
    data = generate_pointcloud(tag, n_points)
    if data is None:
        return None

    print(f"  [{tag}] Compacting...")
    compact = compact_data(data, max_points=8000)

    print(f"  [{tag}] Building HTML...")
    html = build_single_html(tag, compact, data["metadata"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_html, "w") as f:
        f.write(html)
    with open(out_json, "w") as f:
        json.dump(compact, f, separators=(",", ":"), cls=NumpyEncoder)

    meta = data["metadata"]
    print(f"  [{tag}] ✓ PCA={meta['variance_total']*100:.1f}%  PC1:PC2={meta['pc1_ratio']:.1f}:1  cliffs={meta['n_cliffs']}  → {out_html.name} ({os.path.getsize(out_html)/1024:.0f}KB)")
    return out_html


def main():
    parser = argparse.ArgumentParser(description="SNI Pipeline — Semantic Nebula Imaging")
    parser.add_argument("--tag", nargs="*", help="Model tag(s) to process")
    parser.add_argument("--batch", action="store_true", help="Process all models with vectors")
    parser.add_argument("--gallery", action="store_true", help="Build gallery page of all models")
    parser.add_argument("--compare", nargs=2, metavar="TAG", help="Build dual comparison page")
    parser.add_argument("--points", type=int, default=50000)
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover processable tags
    all_tags = []
    for d in sorted(BASE.iterdir()):
        if d.is_dir() and (d / "vectors").exists():
            gguf_count = len(list((d / "vectors").glob("*.gguf")))
            if gguf_count >= 5:
                all_tags.append(d.name)

    if args.batch:
        print(f"Batch processing {len(all_tags)} models:\n  " + "\n  ".join(all_tags))
        results = {}
        for tag in all_tags:
            result = process_tag(tag, args.points, args.force)
            if result:
                results[tag] = result
        print(f"\nDone: {len(results)}/{len(all_tags)} models processed")
        if args.gallery:
            build_gallery(results, args)

    elif args.compare:
        t1, t2 = args.compare
        print(f"Comparing {t1} vs {t2}")
        for t in [t1, t2]:
            process_tag(t, args.points, args.force)
        build_comparison(t1, t2)

    elif args.tag:
        for t in args.tag:
            process_tag(t, args.points, args.force)

    else:
        print(f"Available tags ({len(all_tags)}):")
        for t in all_tags:
            print(f"  {t}")
        print("\nUsage: --tag TAG [TAG...] | --batch | --compare TAG1 TAG2")


def build_comparison(t1, t2):
    """Build dual comparison HTML."""
    j1 = OUT_DIR / f"sni_{t1}.json"
    j2 = OUT_DIR / f"sni_{t2}.json"
    if not j1.exists() or not j2.exists():
        print(f"Missing JSON for comparison: {j1.exists()=} {j2.exists()=}")
        return

    d1 = json.load(open(j1))
    d2 = json.load(open(j2))
    m1, m2 = d1["meta"], d2["meta"]

    inline1 = json.dumps(d1, separators=(",", ":"))
    inline2 = json.dumps(d2, separators=(",", ":"))

    health_class = lambda h: {"healthy":"healthy","caution":"caution","fragile":"fragile"}.get(h,"")

    html = f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SNI — {m1["model"]} vs {m2["model"]}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Instrument+Sans:wght@400;600;700&display=swap');
:root {{ --bg:#060610; --fg:#c8ccd0; --muted:#556; --accent:#4df; }}
* {{ margin:0;padding:0;box-sizing:border-box; }}
body {{ background:var(--bg);color:var(--fg);font-family:'Instrument Sans',sans-serif;overflow:hidden;height:100vh; }}
.split {{ position:fixed;top:0;bottom:0; }}
.split-left {{ left:0;width:50%; }}
.split-right {{ left:50%;width:50%; }}
.label-bar {{
  position:absolute;top:0;left:0;right:0;z-index:20;pointer-events:none;
  background:linear-gradient(180deg,rgba(6,6,16,0.95),transparent);
  padding:18px 24px 40px;
}}
.model-name {{ font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;letter-spacing:2px; }}
.model-sub {{ font-family:'JetBrains Mono',monospace;font-size:11px;color:#667;margin-top:4px; }}
.stat-row {{ font-size:11px;color:#889;margin-top:5px;font-family:'JetBrains Mono',monospace; }}
.stat-row .v {{ color:var(--accent);font-weight:700; }}
.healthy {{ color:#4d8; }} .caution {{ color:#fd8; }} .fragile {{ color:#f84; }}
.center-title {{
  position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:30;text-align:center;pointer-events:none;
}}
.center-title .main {{ font-family:'JetBrains Mono',monospace;font-size:13px;letter-spacing:5px;color:#556; }}
.center-title .cn {{ font-size:11px;color:#445;margin-top:3px; }}
.divider {{
  position:fixed;left:50%;top:0;bottom:0;width:1px;z-index:25;
  background:linear-gradient(180deg,transparent 10%,#1a1a3a 30%,#1a1a3a 70%,transparent 90%);
}}
.bottom-bar {{
  position:fixed;bottom:0;left:0;right:0;z-index:20;pointer-events:none;
  background:linear-gradient(0deg,rgba(6,6,16,0.95),transparent);
  padding:30px 24px 16px;display:flex;justify-content:space-between;align-items:flex-end;
}}
.legend {{ font-family:'JetBrains Mono',monospace;font-size:10px;color:#556; }}
.grad-bar {{ width:200px;height:8px;border-radius:4px;
  background:linear-gradient(90deg,#0a1a44,#1a4488,#2a8866,#99cc33,#cc8822,#cc3322);margin-top:4px; }}
.grad-labels {{ display:flex;justify-content:space-between;font-size:9px;color:#445;margin-top:2px;width:200px; }}
</style>
</head><body>
<div class="center-title">
  <div class="main">SEMANTIC NEBULA IMAGING · SNI</div>
  <div class="cn">语义星云成像 — LLM人格流形对比</div>
</div>
<div class="divider"></div>

<div class="split split-left" id="left-pane">
  <div class="label-bar">
    <div class="model-name {health_class(m1["health"])}">{m1["model"]}</div>
    <div class="model-sub">BF16 · {m1["n_layers"]} layers · 5D personality space</div>
    <div class="stat-row">PCA variance: <span class="v">{m1["variance_total"]*100:.1f}%</span> ({" + ".join(f"{v*100:.1f}" for v in m1["variance_explained"])})</div>
    <div class="stat-row">PC1:PC2 = <span class="v">{m1["pc1_ratio"]:.1f}:1</span> · Structure: <span class="{health_class(m1["health"])}">{m1["structure"]}</span></div>
    <div class="stat-row">Cliffs: <span class="v">{m1["n_cliffs"]}</span> · Avg danger: <span class="v">{m1["avg_rep"]:.3f}</span></div>
  </div>
  <div id="c-left"></div>
</div>

<div class="split split-right" id="right-pane">
  <div class="label-bar">
    <div class="model-name {health_class(m2["health"])}">{m2["model"]}</div>
    <div class="model-sub">BF16 · {m2["n_layers"]} layers · 5D personality space</div>
    <div class="stat-row">PCA variance: <span class="v">{m2["variance_total"]*100:.1f}%</span> ({" + ".join(f"{v*100:.1f}" for v in m2["variance_explained"])})</div>
    <div class="stat-row">PC1:PC2 = <span class="v">{m2["pc1_ratio"]:.1f}:1</span> · Structure: <span class="{health_class(m2["health"])}">{m2["structure"]}</span></div>
    <div class="stat-row">Cliffs: <span class="v">{m2["n_cliffs"]}</span> · Avg danger: <span class="v">{m2["avg_rep"]:.3f}</span></div>
  </div>
  <div id="c-right"></div>
</div>

<div class="bottom-bar">
  <div class="legend">
    <div>DANGER LEVEL (trigram_rep)</div>
    <div class="grad-bar"></div>
    <div class="grad-labels"><span>0.00 safe</span><span>0.03</span><span>0.06</span><span>0.10</span><span>0.25+</span></div>
  </div>
  <div class="legend" style="text-align:right">
    <div>5 dims: emotion · formality · creativity · confidence · empathy</div>
  </div>
</div>

<script type="importmap">{{"imports":{{"three":"https://esm.sh/three@0.162.0","three/addons/":"https://esm.sh/three@0.162.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';

const DATA_L={inline1};
const DATA_R={inline2};

function createNebula(id,data){{
  const el=document.getElementById(id);
  const w=el.clientWidth||innerWidth/2,h=el.clientHeight||innerHeight;
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x060610);scene.fog=new THREE.FogExp2(0x060610,0.06);
  const camera=new THREE.PerspectiveCamera(55,w/h,0.01,100);camera.position.set(1.8,1.2,1.8);
  const renderer=new THREE.WebGLRenderer({{antialias:true}});renderer.setSize(w,h);renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  el.appendChild(renderer.domElement);
  const ctrl=new OrbitControls(camera,renderer.domElement);ctrl.enableDamping=true;ctrl.dampingFactor=0.05;
  ctrl.autoRotate=true;ctrl.autoRotateSpeed=0.4;ctrl.maxDistance=5;ctrl.minDistance=0.3;
  scene.add(new THREE.AmbientLight(0x223344,0.3));
  const grid=new THREE.GridHelper(3,15,0x0a0a1a,0x0a0a1a);grid.position.y=-1.1;scene.add(grid);
  const c=data.c,n=c.x.length;
  const pos=new Float32Array(n*3),col=new Float32Array(n*3),siz=new Float32Array(n);
  for(let i=0;i<n;i++){{pos[i*3]=c.x[i];pos[i*3+1]=c.y[i];pos[i*3+2]=c.z[i];
    col[i*3]=c.r[i];col[i*3+1]=c.g[i];col[i*3+2]=c.b[i];
    siz[i]=c.rep[i]>0.08?2.0:(c.rep[i]>0.05?1.2:0.8);}}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  geo.setAttribute('size',new THREE.BufferAttribute(siz,1));
  const mat=new THREE.ShaderMaterial({{
    uniforms:{{uTime:{{value:0}}}},
    vertexShader:`attribute float size;varying vec3 vC;void main(){{vC=color;vec4 mv=modelViewMatrix*vec4(position,1.0);
      gl_PointSize=size*(100.0/-mv.z);gl_Position=projectionMatrix*mv;}}`,
    fragmentShader:`varying vec3 vC;void main(){{float d=length(gl_PointCoord-0.5);if(d>0.5)discard;
      float core=smoothstep(0.5,0.05,d);float glow=exp(-d*10.0)*0.5+core*0.5;
      gl_FragColor=vec4(vC*0.65,glow*0.3);}}`,
    vertexColors:true,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,
  }});
  scene.add(new THREE.Points(geo,mat));
  const dimCols=[0xff4d4d,0x4db3ff,0xffcc33,0x33ff66,0xcc55ff];let di=0;
  for(const[dim,pd] of Object.entries(data.paths||{{}})){{
    const pts=pd.x.map((x,i)=>new THREE.Vector3(x,pd.y[i],pd.z[i]));
    if(pts.length<2){{di++;continue;}}
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({{color:dimCols[di%5],transparent:true,opacity:0.5}})));di++;
  }}
  const clock=new THREE.Clock();
  (function anim(){{requestAnimationFrame(anim);ctrl.update();mat.uniforms.uTime.value=clock.getElapsedTime();renderer.render(scene,camera);}})();
  return {{scene,camera,renderer,ctrl}};
}}

const L=createNebula('c-left',DATA_L),R=createNebula('c-right',DATA_R);
(function sync(){{requestAnimationFrame(sync);
  const a=L.camera.position.clone().lerp(R.camera.position,0.5);
  L.camera.position.lerp(a,0.02);R.camera.position.lerp(a,0.02);}})();
addEventListener('resize',()=>{{const w=innerWidth/2,h=innerHeight;
  [L,R].forEach(s=>{{s.camera.aspect=w/h;s.camera.updateProjectionMatrix();s.renderer.setSize(w,h);}});}});
</script></body></html>'''

    out_path = OUT_DIR / f"sni_compare_{t1}_vs_{t2}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"  Comparison → {out_path} ({os.path.getsize(out_path)/1024:.0f}KB)")


if __name__ == "__main__":
    main()
