import os
import html
import json
import re
import tempfile
import textwrap
import time
from datetime import datetime
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for server stability
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.tri as mtri
import plotly.graph_objects as go
try:
    import cv2
except ImportError:
    cv2 = None
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LOAD_TYPES = ["Tensile", "Torsion", "Compressive"]
DEFORM_SCALE = 1.0

FAILURE_HTML = """
<div class="failure-alert" style="background:linear-gradient(135deg,rgba(239,68,68,0.2),rgba(239,68,68,0.05));border:2px solid #ef4444;border-radius:12px;padding:16px;animation:pulse 2s infinite;">
    <h3 style="color:#ef4444;margin-top:0;">âš ï¸ FAILURE PREDICTED</h3>
    <p>Reduce load or upgrade material to ensure structural integrity.</p>
</div>
<style>
@keyframes pulse{0%,100%{box-shadow:0 0 20px rgba(239,68,68,0.3);}50%{box-shadow:0 0 40px rgba(239,68,68,0.6);}}
</style>
"""

# -----------------------------------------------------------------------------
# Colormap: Turbo (blue -> red)
# -----------------------------------------------------------------------------
def turbo_colormap(t: float) -> Tuple[float, float, float]:
    """Accurate Turbo colormap implementation. t in [0,1]."""
    # Turbo colormap polynomial approximation
    r = np.clip(0.1357 + 2.8473 * t - 5.4422 * t**2 + 6.9961 * t**3 - 3.0735 * t**4, 0, 1)
    if t > 0.8:
        r = np.clip(0.8936 + 1.3262 * (t - 0.8) - 5.8923 * (t - 0.8)**2, 0, 1)
    if t > 0.95:
        r = np.clip(0.9932 - 0.3297 * (t - 0.95), 0, 1)
    
    g = np.clip(0.0914 + 1.5584 * t - 0.2789 * t**2 - 3.9138 * t**3 + 4.9994 * t**4 - 1.7797 * t**5, 0, 1)
    if t > 0.6:
        g = np.clip(0.6512 + 0.7347 * (t - 0.6) - 2.8525 * (t - 0.6)**2 + 2.9311 * (t - 0.6)**3, 0, 1)
    if t > 0.9:
        g = np.clip(0.9400 - 0.9400 * (t - 0.9), 0, 1)
    
    b = np.clip(0.2511 + 2.0347 * t - 10.8794 * t**2 + 22.1419 * t**3 - 18.5707 * t**4 + 5.0225 * t**5, 0, 1)
    if t > 0.4:
        b = np.clip(0.6162 - 0.9132 * (t - 0.4) + 0.3447 * (t - 0.4)**2, 0, 1)
    if t > 0.7:
        b = np.clip(0.3086 - 1.0293 * (t - 0.7) + 1.0293 * (t - 0.7)**2, 0, 1)
    if t > 0.9:
        b = np.clip(0.0515 - 0.5150 * (t - 0.9), 0, 1)
    
    return (float(r), float(g), float(b))


# -----------------------------------------------------------------------------
# Mesh Loader & Preprocessor
# -----------------------------------------------------------------------------
def load_stl_as_buffergeometry(stl_path: str):
    """Load STL using trimesh and extract vertices, faces, normals."""
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        # Combine all geometries
        meshes = []
        for geom in mesh.geometry.values():
            meshes.append(geom)
        mesh = trimesh.util.concatenate(meshes)
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    normals = mesh.vertex_normals.astype(np.float32)
    return vertices, faces, normals


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, max_vertices: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce mesh complexity for interactive simulation stability."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    if len(vertices) <= max_vertices or len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    # Try trimesh simplification first (best quality when available).
    try:
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        target_faces = max(int(len(faces) * (max_vertices / max(len(vertices), 1))), max_vertices // 2)
        simplified = tm.simplify_quadric_decimation(target_faces)
        if simplified is not None and len(simplified.vertices) > 0 and len(simplified.faces) > 0:
            out_v = np.asarray(simplified.vertices, dtype=np.float32)
            out_f = np.asarray(simplified.faces, dtype=np.uint32)
            if len(out_v) <= max_vertices:
                return out_v, out_f
    except Exception:
        pass

    # Fallback: sample faces and rebuild compact vertex indices.
    ratio = max_vertices / float(len(vertices))
    target_faces = max(int(len(faces) * ratio), max_vertices // 2)
    target_faces = min(target_faces, len(faces))
    if target_faces <= 0:
        return vertices[:max_vertices], np.zeros((0, 3), dtype=np.uint32)

    # Evenly sample faces to preserve shape coverage.
    pick = np.linspace(0, len(faces) - 1, num=target_faces, dtype=np.int64)
    sampled_faces = faces[pick]
    used_vertices = np.unique(sampled_faces.reshape(-1))

    if len(used_vertices) > max_vertices:
        used_vertices = used_vertices[:max_vertices]
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)

    valid_mask = np.all(remap[sampled_faces] >= 0, axis=1)
    sampled_faces = sampled_faces[valid_mask]
    if len(sampled_faces) == 0:
        return vertices[:max_vertices], np.zeros((0, 3), dtype=np.uint32)

    out_faces = remap[sampled_faces].astype(np.uint32)
    out_vertices = vertices[used_vertices].astype(np.float32)
    return out_vertices, out_faces


# -----------------------------------------------------------------------------
# Physics Engine (GNN-inspired surrogate kernel)
# -----------------------------------------------------------------------------
def _build_vertex_adjacency(num_vertices: int, faces: np.ndarray) -> List[List[int]]:
    """Build sparse vertex adjacency from triangle faces."""
    neighbors = [set() for _ in range(num_vertices)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    return [list(nset) for nset in neighbors]


def predict_physics(
    mesh_data: Dict[str, Any],
    load_type: str,
    load_n: float = 1000.0,
    temperature_c: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GNN-inspired surrogate that predicts displacement and Von Mises-like stress per vertex.
    Returns:
        von_mises: (N,) normalized in [0, 1]
        displacement: (N, 3)
    """
    vertices = np.asarray(mesh_data.get("vertices"), dtype=np.float32)
    faces = np.asarray(mesh_data.get("faces"), dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Invalid mesh_data.vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Invalid mesh_data.faces")

    n = len(vertices)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-6)
    center = (bbox_min + bbox_max) * 0.5

    xn = (vertices[:, 0] - center[0]) / span[0]
    yn = (vertices[:, 1] - center[1]) / span[1]
    zn = (vertices[:, 2] - center[2]) / span[2]
    radial = np.sqrt(xn * xn + zn * zn)

    displacement = np.zeros((n, 3), dtype=np.float32)
    # Load + temperature affect deformation/stress distribution (not only magnitude).
    # This is a surrogate, so we intentionally introduce mild non-linear localization
    # as load increases, and thermal softening as temperature rises.
    load_n = float(load_n)
    temperature_c = float(temperature_c)
    load_ratio = np.clip(load_n / 5000.0, 0.05, 5.0)  # 5000N ~ baseline
    temp_soften = np.clip(1.0 - 0.0012 * (temperature_c - 25.0), 0.55, 1.10)
    # Use a more linear/sensitive amplitude so deformation magnitude is dynamic.
    amp = (0.25 + 0.95 * load_ratio) / max(temp_soften, 1e-6)
    abs_yn = np.abs(yn)
    # Load-dependent spatial localization so stress patterns vary with load.
    # Higher load concentrates deformation towards the ends / outer radius.
    p_end = 1.15 + 0.95 * np.clip(load_ratio, 0.1, 2.5)
    p_rad = 0.90 + 0.80 * np.clip(load_ratio, 0.1, 2.5)
    local = (0.55 + 0.45 * (abs_yn ** p_end)) * (0.30 + 0.70 * (radial ** p_rad))

    if load_type == "Tensile":
        displacement[:, 1] = yn * (1.0 + 0.2 * radial)
        displacement[:, 0] = -0.12 * xn * np.abs(yn)
        displacement[:, 2] = -0.12 * zn * np.abs(yn)
    elif load_type == "Torsion":
        twist = yn * np.pi * (0.75 + 0.55 * np.clip(load_ratio, 0.1, 2.5))
        cos_t = np.cos(twist)
        sin_t = np.sin(twist)
        rx = vertices[:, 0] - center[0]
        rz = vertices[:, 2] - center[2]
        tx = rx * cos_t - rz * sin_t
        tz = rx * sin_t + rz * cos_t
        displacement[:, 0] = (tx - rx) / span[0]
        displacement[:, 2] = (tz - rz) / span[2]
        displacement[:, 1] = 0.08 * np.sin(twist) * radial
    elif load_type == "Compressive":
        # Axial compression with center-focused bulging for buckling-like behavior.
        buckle = 1.0 + 0.55 * np.clip(load_ratio - 0.5, 0.0, 2.2)
        neck = np.clip(1.0 - np.abs(yn), 0.0, 1.0)
        displacement[:, 1] = -0.85 * yn * buckle
        displacement[:, 0] = 0.24 * xn * (neck ** 1.4) * buckle
        displacement[:, 2] = 0.24 * zn * (neck ** 1.4) * buckle
    else:
        raise ValueError(f"Unsupported load type: {load_type}")

    # Apply global + spatial scaling (keeps patterns load-dependent).
    displacement *= (np.float32(amp) * local.astype(np.float32))[:, None]

    # Message-passing style smoothing (3 rounds) to emulate graph aggregation.
    adjacency = _build_vertex_adjacency(n, faces)
    for _ in range(3):
        updated = displacement.copy()
        for idx, nbrs in enumerate(adjacency):
            if not nbrs:
                continue
            updated[idx] = 0.55 * displacement[idx] + 0.45 * displacement[nbrs].mean(axis=0)
        displacement = updated

    # Approximate strain energy from neighbor displacement gradients.
    grad_mag = np.zeros(n, dtype=np.float32)
    for idx, nbrs in enumerate(adjacency):
        if not nbrs:
            continue
        diffs = displacement[nbrs] - displacement[idx]
        grad_mag[idx] = np.sqrt((diffs * diffs).sum(axis=1)).mean()

    # Nonlinear localization: at higher load, high-gradient regions amplify more.
    gmax = float(np.max(grad_mag)) if np.max(grad_mag) > 1e-12 else 1.0
    gnorm = grad_mag / gmax
    nonlin = 1.0 + 1.35 * (np.clip(load_ratio - 0.6, 0.0, 2.5) / 2.5) * (gnorm ** 2.4)
    stress_proxy = (grad_mag * nonlin.astype(np.float32)).astype(np.float32)

    # IMPORTANT: return an unnormalized stress proxy so changing load can change the map,
    # and let the caller scale it into MPa for visualization.
    return stress_proxy, displacement.astype(np.float32)


# -----------------------------------------------------------------------------
# HTML Template Generator
# -----------------------------------------------------------------------------
def generate_simulation_html(
    vertices: np.ndarray,
    faces: np.ndarray,
    displacements: np.ndarray,
    stresses: np.ndarray,
    load_type: str,
    deform_scale: float = 1.0,
    temperature: float = 25.0,
) -> str:
    """Create a self-contained HTML payload with a guarded Three.js 4s workflow."""
    verts_list = vertices.tolist()
    faces_list = faces.tolist()
    disp_list = displacements.tolist()
    stress_list = stresses.tolist()

    # Dynamic background for Three.js view
    t_norm = np.clip((float(temperature) + 50) / 450, 0, 1)
    bg_r = int(15 + 55 * t_norm)
    bg_g = int(19 + 12 * t_norm)
    bg_b = int(25 - 15 * t_norm)
    bg_hex = '#%02x%02x%02x' % (bg_r, bg_g, bg_b)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1.0" />
<style>
html, body {{ margin: 0; padding: 0; background: {bg_hex}; color: #dbe6f5; font-family: Segoe UI, sans-serif; }}
#sim-wrap {{ position: relative; width: 100%; height: 420px; overflow: hidden; border: 1px solid #1f2a3d; border-radius: 10px; }}
#sim-root {{ position: absolute; inset: 0; }}
#status {{ position: absolute; top: 12px; left: 12px; background: rgba(8,12,20,0.88); border: 1px solid #1f2a3d; border-radius: 8px; padding: 8px 12px; z-index: 4; font-size: 13px; }}
#progress-wrap {{ position: absolute; top: 58px; left: 12px; width: 260px; height: 12px; background: #1a2231; border-radius: 20px; border: 1px solid #2c3f5a; overflow: hidden; z-index: 4; }}
#progress-bar {{ width: 0%; height: 100%; background: linear-gradient(90deg,#1d4ed8,#06b6d4,#eab308,#ef4444); }}
#legend {{ position: absolute; right: 12px; bottom: 12px; z-index: 4; background: rgba(8,12,20,0.88); border: 1px solid #1f2a3d; border-radius: 8px; padding: 8px 10px; font-size: 12px; }}
#legend-bar {{ width: 180px; height: 10px; margin: 6px 0; background: linear-gradient(90deg,#1d4ed8,#06b6d4,#eab308,#ef4444); border-radius: 12px; }}
#overlay {{ position: absolute; inset: 0; z-index: 3; pointer-events: none; display: flex; align-items: center; justify-content: center; color: #dbe6f5; font-size: 28px; font-weight: 700; background: rgba(6,9,15,0.18); }}
</style>
<script type="importmap">
{{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}
</script>
</head>
<body>
<div id="sim-wrap">
  <div id="sim-root"></div>
  <div id="status">Preparing surrogate simulation...</div>
  <div id="progress-wrap"><div id="progress-bar"></div></div>
  <div id="overlay">0%</div>
  <div id="legend"><div>Turbo Stress Scale</div><div id="legend-bar"></div><div style="display:flex;justify-content:space-between;"><span>Low</span><span>High</span></div></div>
</div>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

const payload = {{
  vertices: {json.dumps(verts_list)},
  faces: {json.dumps(faces_list)},
  displacements: {json.dumps(disp_list)},
  stresses: {json.dumps(stress_list)},
  loadType: "{load_type}",
  deformScale: {deform_scale},
  temperature: {temperature}
}};

const state = {{
  renderer: null, scene: null, camera: null, controls: null,
  mesh: null, wire: null, geometry: null,
  animationId: null, disposed: false
}};

function component(id) {{
  const el = document.getElementById(id);
  if (!el) throw new Error(`Component Not Found: #${{id}}`);
  return el;
}}

function setStatus(text) {{ try {{ component("status").textContent = text; }} catch (_) {{}} }}
function setProgress(p) {{
  const val = Math.max(0, Math.min(100, p));
  try {{ component("progress-bar").style.width = `${{val}}%`; component("overlay").textContent = `${{Math.round(val)}}%`; }} catch (_) {{}}
}}

function turbo(t) {{
  const x = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 0.1357 + 2.8473*x - 5.4422*x*x + 6.9961*x*x*x - 3.0735*x*x*x*x));
  const g = Math.max(0, Math.min(1, 0.0914 + 1.5584*x - 0.2789*x*x - 3.9138*x*x*x + 4.9994*x*x*x*x - 1.7797*x*x*x*x*x));
  const b = Math.max(0, Math.min(1, 0.2511 + 2.0347*x - 10.8794*x*x + 22.1419*x*x*x - 18.5707*x*x*x*x + 5.0225*x*x*x*x*x));
  return [r, g, b];
}}

function cleanup() {{
  if (state.animationId) {{ cancelAnimationFrame(state.animationId); state.animationId = null; }}
  if (state.controls) state.controls.dispose();
  if (state.scene) {{
    state.scene.traverse((obj) => {{
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {{
        const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
        materials.forEach((mat) => {{
          if (mat.map) mat.map.dispose();
          if (mat.lightMap) mat.lightMap.dispose();
          if (mat.normalMap) mat.normalMap.dispose();
          if (mat.roughnessMap) mat.roughnessMap.dispose();
          if (mat.metalnessMap) mat.metalnessMap.dispose();
          if (mat.emissiveMap) mat.emissiveMap.dispose();
          mat.dispose();
        }});
      }}
    }});
  }}
  if (state.renderer) {{
    state.renderer.dispose();
    const root = document.getElementById("sim-root");
    if (root && state.renderer.domElement && root.contains(state.renderer.domElement)) root.removeChild(state.renderer.domElement);
  }}
  state.disposed = true;
}}

async function boot() {{
  if (window.__mechAiCleanup) {{
    try {{ window.__mechAiCleanup(); }} catch (_) {{}}
  }}
  cleanup();
  state.disposed = false;
  const root = component("sim-root");
  const wrap = component("sim-wrap");

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("{bg_hex}");
  const camera = new THREE.PerspectiveCamera(
    45,
    Math.max(wrap.clientWidth, 1) / Math.max(wrap.clientHeight, 1),
    0.01,
    100
  );
  camera.position.set(2.6, 2.0, 2.8);

  const renderer = new THREE.WebGLRenderer({{ antialias: false, alpha: false }});
  renderer.setSize(Math.max(wrap.clientWidth, 1), Math.max(wrap.clientHeight, 1));
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.shadowMap.enabled = false;
  root.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const ambient = new THREE.AmbientLight(0xffffff, 0.85);
  const dir = new THREE.DirectionalLight(0xffffff, 0.65);
  dir.position.set(3, 4, 2);
  scene.add(ambient, dir);

  const geometry = new THREE.BufferGeometry();
  const verts = new Float32Array(payload.vertices.flat());
  const indices = new Uint32Array(payload.faces.flat());
  geometry.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();

  const colors = new Float32Array(payload.stresses.length * 3);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const basePositions = new Float32Array(verts);

  const solidMat = new THREE.MeshStandardMaterial({{ vertexColors: true, roughness: 0.45, metalness: 0.08 }});
  const mesh = new THREE.Mesh(geometry, solidMat);
  scene.add(mesh);

  const wireMat = new THREE.LineBasicMaterial({{ color: 0x5eead4, transparent: true, opacity: 0.9 }});
  let wireGeom = new THREE.WireframeGeometry(geometry);
  const wire = new THREE.LineSegments(wireGeom, wireMat);
  scene.add(wire);

  geometry.computeBoundingBox();
  const bbox = geometry.boundingBox;
  const center = new THREE.Vector3();
  bbox.getCenter(center);
  mesh.position.sub(center);
  wire.position.copy(mesh.position);
  const size = new THREE.Vector3(); bbox.getSize(size);
  const s = 1.8 / Math.max(size.x, size.y, size.z, 1e-6);
  mesh.scale.setScalar(s); wire.scale.copy(mesh.scale);

  state.renderer = renderer; state.scene = scene; state.camera = camera; state.controls = controls;
  state.mesh = mesh; state.wire = wire; state.geometry = geometry;

  const durationMs = 4000;
  const fpsInterval = 1000 / 30;
  let t0 = performance.now();
  let lastFrame = 0;

  function renderStep(now) {{
    if (state.disposed) return;
    state.animationId = requestAnimationFrame(renderStep);
    if (now - lastFrame < fpsInterval) return;
    lastFrame = now;

    const elapsed = Math.max(0, now - t0);
    const p = Math.min(1, elapsed / durationMs);
    setProgress(p * 100);
    if (p < 0.3) setStatus("Pre-processing wireframe...");
    else if (p < 0.6) setStatus("Solving surrogate kernel...");
    else setStatus("Post-processing heatmap...");

    const pos = geometry.attributes.position.array;
    const col = geometry.attributes.color.array;
    const deform = payload.deformScale * p;
    // Thermal baseline: hot parts glow even at low stress.
    const thermalBase = Math.max(0, (payload.temperature - 25) / 400) * 0.22;

    for (let i = 0; i < payload.vertices.length; i++) {{
      pos[i*3] = (payload.vertices[i][0] + payload.displacements[i][0] * deform);
      pos[i*3+1] = (payload.vertices[i][1] + payload.displacements[i][1] * deform);
      pos[i*3+2] = (payload.vertices[i][2] + payload.displacements[i][2] * deform);
      
      // Combine stress (Turbo) with thermal glow.
      const s = payload.stresses[i] * p;
      const [r, g, b] = turbo(s + thermalBase);
      col[i*3] = r; col[i*3+1] = g; col[i*3+2] = b;
    }}
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
    geometry.computeVertexNormals();

    if (wire.geometry) wire.geometry.dispose();
    wire.geometry = new THREE.WireframeGeometry(geometry);
    wire.material.opacity = Math.max(0, 1.0 - p * 2.2);

    controls.update();
    renderer.render(scene, camera);

    if (p >= 1) {{
      setStatus(`Complete (${{payload.loadType}})`);
      const ov = document.getElementById("overlay");
      if (ov) ov.style.display = "none";
    }}
  }}

  const onResize = () => {{
    if (!state.camera || !state.renderer) return;
    const w = Math.max(wrap.clientWidth, 1);
    const h = Math.max(wrap.clientHeight, 1);
    state.camera.aspect = w / h;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(w, h);
  }};
  window.addEventListener("resize", onResize);
  onResize();

  setStatus("Running 4-second simulation...");
  setProgress(0);
  renderStep(performance.now());
}}

try {{
  boot();
  window.__mechAiCleanup = cleanup;
}} catch (err) {{
  setStatus(`Simulation init error: ${{err.message}}`);
  console.error(err);
}}
</script>
</body>
</html>"""
    return html


# -----------------------------------------------------------------------------
# Materials Database
# -----------------------------------------------------------------------------
MATERIALS = {
    "steel": {"label": "Steel", "yield_mpa": 250, "elastic_modulus_gpa": 200},
    "aluminum": {"label": "Aluminum", "yield_mpa": 270, "elastic_modulus_gpa": 70},
    "titanium": {"label": "Titanium", "yield_mpa": 880, "elastic_modulus_gpa": 116},
    "plastic": {"label": "Plastic", "yield_mpa": 40, "elastic_modulus_gpa": 3},
    "brass": {"label": "Brass", "yield_mpa": 200, "elastic_modulus_gpa": 100},
    "copper": {"label": "Copper", "yield_mpa": 70, "elastic_modulus_gpa": 120},
}


# -----------------------------------------------------------------------------
# Simulation State Management
# -----------------------------------------------------------------------------
class SimState:
    """Global simulation state for tracking last detected object and dimensions."""
    def __init__(self):
        self.last_detected_object = "bolt"
        self.last_dims = {}
        self.last_stl_path = None

# Global state instance
sim_state = SimState()

def _sanitize_predictions(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp unstable ML outputs to physically plausible ranges."""
    out = dict(predictions)
    out["yield_mpa"] = float(np.clip(out.get("yield_mpa", 250.0), 20.0, 2500.0))
    out["stress_mpa"] = float(np.clip(out.get("stress_mpa", 1.0), 0.01, out["yield_mpa"] * 2.0))
    out["strain"] = float(np.clip(out.get("strain", 1e-5), 1e-7, 0.2))
    out["safety_factor"] = float(np.clip(out.get("safety_factor", out["yield_mpa"] / max(out["stress_mpa"], 1e-6)), 0.01, 500.0))
    out["stress_ratio"] = float(np.clip(out["stress_mpa"] / max(out["yield_mpa"], 1e-6), 0.0, 5.0))
    out["material_label"] = out.get("material_label", "Steel")
    return out


def _apply_operating_conditions(predictions: Dict[str, Any], load: float, temperature: float) -> Dict[str, Any]:
    """
    Convert ML baseline outputs into operating-condition-aware values.
    Keeps ML as baseline while making extreme load/temp cases fail realistically.
    """
    out = dict(predictions)
    load = float(load)
    temperature = float(temperature)

    load_factor = np.clip((max(load, 100.0) / 3000.0) ** 2.2, 0.25, 45.0)
    thermal_factor = np.clip(1.0 + max(0.0, temperature - 25.0) / 90.0, 1.0, 6.0)
    stress_factor = float(load_factor * thermal_factor)
    yield_softening = np.clip(1.0 - 0.0022 * max(0.0, temperature - 25.0), 0.18, 1.0)

    out["stress_mpa"] = float(np.clip(out["stress_mpa"] * stress_factor, 0.01, 5000.0))
    out["yield_mpa"] = float(np.clip(out["yield_mpa"] * yield_softening, 20.0, 2500.0))
    out["safety_factor"] = float(np.clip(out["yield_mpa"] / max(out["stress_mpa"], 1e-6), 0.01, 500.0))
    out["stress_ratio"] = float(np.clip(out["stress_mpa"] / max(out["yield_mpa"], 1e-6), 0.0, 5.0))
    return out


def _pick_stress_color_max(yield_mpa: float, max_stress_mpa: float) -> float:
    """
    Choose a colorscale max that preserves hotspot contrast at low loads
    while still keeping a meaningful reference near yield at higher loads.
    """
    y = float(max(yield_mpa, 1e-6))
    smax = float(max(max_stress_mpa, 1e-6))
    # Favor visible color contrast on the model.
    return max(1e-6, min(smax * 0.85, max(smax, 0.03 * y)))


# -----------------------------------------------------------------------------
# Main Simulation Functions
# -----------------------------------------------------------------------------
def run_simulation(component_type: str, temperature: float, load: float,
                   load_type: str = "Tensile", deform_scale: float = 1.0):
    """Integrated simulation pipeline."""
    try:
        print("\n" + "=" * 50)
        print(f"[MECHAI] SIMULATION START: {component_type} | Load={load}N | Temp={temperature}Â°C")
        print("=" * 50)
        
        # Access the globally tracked STL_PATH
        from Modelling import STL_PATH as stl_path
        print(f"[Sim] Checking STL path: {stl_path}")
        
        # Strict requirement: do not simulate dummy geometry in simulation tab.
        if not stl_path or not os.path.exists(stl_path):
            print("[Sim] ERROR: No STL found.")
            msg = (
                "âš ï¸ No designed model found for simulation.\n\n"
                "Please generate a model first in the **3D Modelling** tab, then run simulation."
            )
            return (
                0.0, 0.0, 0.0, 0.0,
                None, None, None, None,
                msg,
                {"visible": False, "value": ""},
                None
            )

        print("[Sim] Loading mesh...")
        from ml import _predict_from_ml_models
        # Get ML predictions
        predictions = _sanitize_predictions(_predict_from_ml_models(load, temperature, component_type))
        predictions = _apply_operating_conditions(predictions, load, temperature)
        
        # Load mesh from the designed STL only.
        vertices, faces, normals = load_stl_as_buffergeometry(stl_path)
        
        print(f"[Sim] Mesh loaded: {len(vertices)} vertices. Decimating...")
        # Reduce geometric complexity to keep browser render stable.
        vertices, faces = decimate_mesh(vertices, faces, max_vertices=5000)

        print("[Sim] Predicting physics...")
        # Compute surrogate GNN-style simulation
        stress_proxy, displacements = predict_physics(
            {"vertices": vertices, "faces": faces},
            load_type,
            load_n=load,
            temperature_c=temperature,
        )
        
        print("[Sim] Scaling stress fields...")
        # Scale surrogate stress field into MPa while preserving spatial distribution.
        smax = float(np.max(stress_proxy)) if len(stress_proxy) else 0.0
        if smax > 1e-9:
            stresses = (stress_proxy / smax) * float(predictions["stress_mpa"])
        else:
            stresses = np.zeros_like(stress_proxy, dtype=np.float32)
        smax_curr = float(np.max(stresses)) if len(stresses) else 0.0
        q90 = float(np.quantile(stresses, 0.90)) if len(stresses) else 0.0
        stress_cmax = max(
            _pick_stress_color_max(yield_mpa=float(predictions["yield_mpa"]), max_stress_mpa=smax_curr),
            q90,
            1e-6,
        )
        
        print("[Sim] Generating interactive 3D plot...")
        # Use HTML/JS embed path for reliable client-side 4s animation controls.
        interactive_plot = _generate_interactive_3d_html(
            vertices=vertices,
            faces=faces,
            stresses=stresses,
            displacements=displacements,
            deform_scale=deform_scale,
            load_type=load_type,
            temperature=temperature,
        )
        
        print("[Sim] Generating curves and reports...")

        # Generate standard plots & reports
        sf_curve, deform_curve = _generate_response_curves(
            predictions,
            component_type,
            temperature,
            current_load=load,
            load_type=load_type,
        )
        heatmap_plot = _generate_heatmap_plot(
            predictions,
            vertices,
            faces,
            displacements,
            stresses,
            temperature=temperature,
            stress_cmax=stress_cmax,
        )
        compliance_gauge = _generate_compliance_gauge(predictions)
        ai_report = _generate_ai_report(predictions, load, temperature, load_type, deform_scale)

        # Update failure panel visibility and content
        is_failure = predictions["safety_factor"] < 1.0
        failure_panel_update = {"visible": is_failure, "value": FAILURE_HTML if is_failure else ""}
        
        return (
            predictions["safety_factor"],
            predictions["stress_mpa"],
            predictions["strain"],
            predictions["yield_mpa"],
            sf_curve,
            deform_curve,
            heatmap_plot,
            compliance_gauge,
            ai_report,
            failure_panel_update,
            interactive_plot
        )
    except Exception as e:
        # Return default values on error (11 items)
        return (2.0, 0.0, 0.0, 0.0, None, None, None, None, 
                f"âš ï¸ Simulation error: {str(e)}", {"visible": False, "value": ""}, None)


def find_failure(component_type: str, temperature: float, load: float,
                 load_type: str = "Tensile", deform_scale: float = 1.0):
    """Find failure point by increasing load until safety factor < 1."""
    try:
        from ml import _predict_from_ml_models
        
        current_load = load
        max_load = 10000
        step = 100
        
        if getattr(sim_state, "last_stl_path", None):
            component_type = getattr(sim_state, "last_detected_object", component_type)

        while current_load <= max_load:
            predictions = _sanitize_predictions(_predict_from_ml_models(current_load, temperature, component_type))
            predictions = _apply_operating_conditions(predictions, current_load, temperature)
            if predictions["safety_factor"] < 1.0:
                break
            current_load += step
        
        # Generate curve for the report
        sf_curve, _ = _generate_response_curves(
            predictions,
            component_type,
            temperature,
            current_load=current_load,
            load_type=load_type,
        )
        
        ai_report = f"ðŸ’¥ **Failure Predicted**\n\nComponent fails at **{current_load}N** load under {load_type} stress.\nSafety Factor: {predictions['safety_factor']:.2f}\nStress: {predictions['stress_mpa']:.1f} MPa\n\n**Recommendation:** Reduce load below {current_load - step}N or select higher yield material."
        
        failure_panel_update = {"visible": True, "value": FAILURE_HTML}
        
        return (
            current_load,
            predictions["safety_factor"],
            predictions["stress_mpa"],
            ai_report,
            failure_panel_update,
            sf_curve
        )
    except Exception as e:
        return (load, 1.0, 0.0, f"âš ï¸ Error finding failure: {str(e)}", {"visible": False, "value": ""}, None)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _create_dummy_mesh(component_type: str):
    """Create a simple mesh for visualization when no STL is available."""
    if component_type in ["bolt", "screw", "shaft"]:
        # Create cylinder
        height = 1.0
        radius = 0.2
        segments = 32
        
        vertices = []
        faces = []
        
        # Top and bottom centers
        vertices.append([0, height/2, 0])  # top center
        vertices.append([0, -height/2, 0])  # bottom center
        
        # Cylinder vertices
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            vertices.append([x, height/2, z])  # top ring
            vertices.append([x, -height/2, z])  # bottom ring
        
        # Faces
        for i in range(segments):
            next_i = (i + 1) % segments
            # Top face
            faces.append([0, 2 + i*2, 2 + next_i*2])
            # Bottom face
            faces.append([1, 3 + next_i*2, 3 + i*2])
            # Side faces
            faces.append([2 + i*2, 3 + i*2, 2 + next_i*2])
            faces.append([3 + i*2, 3 + next_i*2, 2 + next_i*2])
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.uint32)
    
    elif component_type in ["nut", "washer", "bushing"]:
        # Create hollow cylinder
        height = 0.3
        outer_r = 0.4
        inner_r = 0.2
        segments = 32
        
        vertices = []
        faces = []
        
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            # Outer ring top
            vertices.append([outer_r * cos_a, height/2, outer_r * sin_a])
            # Outer ring bottom
            vertices.append([outer_r * cos_a, -height/2, outer_r * sin_a])
            # Inner ring top
            vertices.append([inner_r * cos_a, height/2, inner_r * sin_a])
            # Inner ring bottom
            vertices.append([inner_r * cos_a, -height/2, inner_r * sin_a])
        
        for i in range(segments):
            next_i = (i + 1) % segments
            base = i * 4
            nb = next_i * 4 # Next base
            
            # Outer side
            faces.append([base, base + 1, nb])
            faces.append([base + 1, nb + 1, nb])
            # Inner side
            faces.append([base + 2, nb + 2, base + 3])
            faces.append([base + 3, nb + 2, nb + 3])
            # Top face
            faces.append([base, nb, base + 2])
            faces.append([nb, nb + 2, base + 2])
            # Bottom face
            faces.append([base + 1, base + 3, nb + 1])
            faces.append([base + 3, nb + 3, nb + 1])
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.uint32)
    
    else:
        # Default cube with more detail
        subdivisions = 4
        vertices = []
        faces = []
        
        # Generate a subdivided cube for better deformation visualization
        step = 1.0 / subdivisions
        for i in range(subdivisions + 1):
            for j in range(subdivisions + 1):
                for k in range(subdivisions + 1):
                    x = -0.5 + i * step
                    y = -0.5 + j * step
                    z = -0.5 + k * step
                    vertices.append([x, y, z])
        
        # Create faces only on the surface
        def get_idx(i, j, k):
            return i * (subdivisions + 1) * (subdivisions + 1) + j * (subdivisions + 1) + k
        
        # Front and back faces (z = Â±0.5)
        for i in range(subdivisions):
            for j in range(subdivisions):
                # Front
                faces.append([get_idx(i, j, 0), get_idx(i+1, j, 0), get_idx(i+1, j+1, 0)])
                faces.append([get_idx(i, j, 0), get_idx(i+1, j+1, 0), get_idx(i, j+1, 0)])
                # Back
                faces.append([get_idx(i, j, subdivisions), get_idx(i+1, j+1, subdivisions), get_idx(i+1, j, subdivisions)])
                faces.append([get_idx(i, j, subdivisions), get_idx(i, j+1, subdivisions), get_idx(i+1, j+1, subdivisions)])
        
        # Left and right faces (x = Â±0.5)
        for j in range(subdivisions):
            for k in range(subdivisions):
                # Left
                faces.append([get_idx(0, j, k), get_idx(0, j+1, k+1), get_idx(0, j+1, k)])
                faces.append([get_idx(0, j, k), get_idx(0, j, k+1), get_idx(0, j+1, k+1)])
                # Right
                faces.append([get_idx(subdivisions, j, k), get_idx(subdivisions, j+1, k), get_idx(subdivisions, j+1, k+1)])
                faces.append([get_idx(subdivisions, j, k), get_idx(subdivisions, j+1, k+1), get_idx(subdivisions, j, k+1)])
        
        # Top and bottom faces (y = Â±0.5)
        for i in range(subdivisions):
            for k in range(subdivisions):
                # Bottom
                faces.append([get_idx(i, 0, k), get_idx(i+1, 0, k), get_idx(i+1, 0, k+1)])
                faces.append([get_idx(i, 0, k), get_idx(i+1, 0, k+1), get_idx(i, 0, k+1)])
                # Top
                faces.append([get_idx(i, subdivisions, k), get_idx(i+1, subdivisions, k+1), get_idx(i+1, subdivisions, k)])
                faces.append([get_idx(i, subdivisions, k), get_idx(i, subdivisions, k+1), get_idx(i+1, subdivisions, k+1)])
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.uint32)


def _generate_response_curves(predictions, component_type="bolt", temperature=25, current_load: float = 1000.0, load_type: str = "Tensile"):
    """Generate smooth, physically consistent response curves anchored to the current operating point."""
    loads = np.linspace(100, 10000, 120)
    p = _sanitize_predictions(predictions)
    base_load = float(np.clip(current_load, 100.0, 10000.0))
    base_stress = max(0.05, float(p["stress_mpa"]))
    yield_mpa = max(20.0, p["yield_mpa"])
    current_sf = float(p["safety_factor"])
    current_strain = float(p["strain"])

    # Apparent modulus inferred from current point, bounded to prevent unstable spikes.
    apparent_E_mpa = np.clip(base_stress / max(p["strain"], 1e-6), 2_000.0, 350_000.0)
    temp_factor = np.clip(1.0 - 0.0009 * max(0.0, temperature - 25.0), 0.65, 1.05)

    # Load-type dependent nonlinearity so curve shape changes meaningfully.
    exp_map = {"Tensile": 1.03, "Torsion": 1.10, "Compressive": 1.18}
    exponent = float(exp_map.get(str(load_type), 1.06))

    # Stress curve goes through (base_load, base_stress) and responds to temperature.
    stress_curve = base_stress * ((loads / max(base_load, 1e-6)) ** exponent) / temp_factor
    stress_curve = np.clip(stress_curve, 0.01, yield_mpa * 2.0)
    safety_factors = np.clip(yield_mpa / np.maximum(stress_curve, 1e-6), 0.01, 500.0)
    elastic_strain = stress_curve / max(apparent_E_mpa, 1e-6)
    plastic_knee = np.clip(stress_curve / max(yield_mpa, 1e-6) - 0.85, 0.0, None)
    strains = np.clip(elastic_strain + 0.0025 * (plastic_knee ** 2), 0.0, 0.2)
    
    # Safety factor plot
    fig1, ax1 = plt.subplots(figsize=(7, 4), facecolor='#111820')
    ax1.set_facecolor('#111820')
    ax1.plot(loads, safety_factors, color='#22d3a0', linewidth=2.5, alpha=0.6, label='Safety Factor')
    
    # Highlight current point
    ax1.scatter([base_load], [current_sf], color='#ffffff', s=100, edgecolors='#22d3a0', linewidths=2, zorder=5, label=f'Current: {current_sf:.2f}')
    ax1.annotate(f"  {current_sf:.2f} SF", (base_load, current_sf), color='#ffffff', fontsize=10, fontweight='bold')

    ax1.axhline(y=1.0, color='#ef4444', linestyle='--', linewidth=2, label='Failure Threshold (SF=1)')
    ax1.axhline(y=2.0, color='#f59e0b', linestyle=':', linewidth=1.5, label='Recommended Min (SF=2)')
    ax1.set_xlabel('Load (N)', color='#e0e8f0', fontsize=11)
    ax1.set_ylabel('Safety Factor', color='#e0e8f0', fontsize=11)
    ax1.set_title(f'Safety Factor vs Load ({load_type})', color='#e0e8f0', fontsize=13, fontweight='bold')
    ax1.tick_params(colors='#8899aa')
    ax1.grid(True, alpha=0.2, color='#2a3a4a')
    ax1.legend(facecolor='#1a2332', edgecolor='#2a3a4a', labelcolor='#e0e8f0', loc='upper right')
    ax1.spines['bottom'].set_color('#2a3a4a')
    ax1.spines['top'].set_color('#2a3a4a')
    ax1.spines['left'].set_color('#2a3a4a')
    ax1.spines['right'].set_color('#2a3a4a')
    ax1.set_ylim(0, max(3.5, float(np.max(safety_factors) * 1.1)))
    
    # Strain plot
    fig2, ax2 = plt.subplots(figsize=(7, 4), facecolor='#111820')
    ax2.set_facecolor('#111820')
    ax2.plot(loads, strains, color='#3b82f6', linewidth=2.5, alpha=0.6, label='Strain')
    
    # Highlight current point
    ax2.scatter([base_load], [current_strain], color='#ffffff', s=100, edgecolors='#3b82f6', linewidths=2, zorder=5, label=f'Current: {current_strain:.4f}')
    ax2.annotate(f"  {current_strain:.4f}", (base_load, current_strain), color='#ffffff', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Load (N)', color='#e0e8f0', fontsize=11)
    ax2.set_ylabel('Strain (mm/mm)', color='#e0e8f0', fontsize=11)
    ax2.set_title(f'Strain vs Load ({load_type})', color='#e0e8f0', fontsize=13, fontweight='bold')
    ax2.tick_params(colors='#8899aa')
    ax2.grid(True, alpha=0.2, color='#2a3a4a')
    ax2.legend(facecolor='#1a2332', edgecolor='#2a3a4a', labelcolor='#e0e8f0')
    ax2.spines['bottom'].set_color('#2a3a4a')
    ax2.spines['top'].set_color('#2a3a4a')
    ax2.spines['left'].set_color('#2a3a4a')
    ax2.spines['right'].set_color('#2a3a4a')
    
    return fig1, fig2


def _generate_heatmap_plot(
    predictions,
    vertices=None,
    faces=None,
    displacements=None,
    stresses=None,
    temperature=25.0,
    stress_cmax: float | None = None,
):
    """Generate high-contrast stress heatmap with standard 2D projection."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#111820')
    ax.set_facecolor('#111820')
    
    if vertices is not None and stresses is not None and len(vertices) > 0:
        # Use simple X-Y projection for static heatmap stability
        px, py = vertices[:, 0], vertices[:, 1]
        
        yield_mpa = float(predictions.get("yield_mpa", 250.0))
        vmax = float(stress_cmax) if stress_cmax is not None else max(float(np.max(stresses)), 0.1 * yield_mpa)

        try:
            triangulation = mtri.Triangulation(px, py)
            cnt = ax.tricontourf(
                triangulation,
                stresses,
                levels=20,
                cmap='turbo',
                vmin=0.0,
                vmax=vmax,
                alpha=0.9,
            )
            ax.triplot(triangulation, color='#ffffff', linewidth=0.1, alpha=0.1)
            
            cbar = fig.colorbar(cnt, ax=ax)
            cbar.set_label("Stress (MPa)", color='white')
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        except:
            # Fallback to scatter if triangulation fails
            sc = ax.scatter(px, py, c=stresses, cmap='turbo', s=10, alpha=0.7, vmin=0.0, vmax=vmax)
            fig.colorbar(sc, ax=ax, label="Stress (MPa)")

    ax.set_title(f"Structural Stress Heatmap ({temperature}Â°C)", color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#8899aa')
    for spine in ax.spines.values():
        spine.set_color('#2a3a4a')
        
    plt.tight_layout()
    return fig


def _generate_interactive_3d_plot(
    vertices,
    faces,
    stresses,
    displacements=None,
    deform_scale=1.0,
    stress_cmax: float | None = None,
    title_suffix: str = "",
    temperature: float = 25.0,
):
    """Generate Plotly Mesh3d with a reliable 4-second deformation + heat ramp."""
    import plotly.graph_objects as go

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    stresses = np.asarray(stresses, dtype=np.float32)
    if displacements is None:
        displacements = np.zeros_like(vertices, dtype=np.float32)
    else:
        displacements = np.asarray(displacements, dtype=np.float32)

    if len(vertices) == 0 or len(faces) == 0:
        return go.Figure()

    # Absolute MPa colorscale to make color meaning consistent across runs.
    cmin = 0.0
    cmax = float(stress_cmax) if stress_cmax is not None else max(float(np.max(stresses)), 1e-6)

    # Center model for stable view framing.
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    bbox = centered.max(axis=0) - centered.min(axis=0)
    model_span = float(np.max(bbox))
    # Use a fixed reference for visual deformation so magnitude is dynamic with load.
    # We expect 'typical' displacements to be around 0.1-0.2 span at max load.
    disp_vis = displacements * (0.28 * model_span)

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return 3.0 * t * t - 2.0 * t * t * t

    frame_count = 40  # 4 seconds @ 10 fps
    duration_ms = int(4000 / frame_count)
    frames = []
    stress_colorscale = [
        [0.0, "#1d4ed8"],
        [0.5, "#ffffff"],
        [1.0, "#dc2626"],
    ]

    for idx in range(frame_count + 1):
        t = idx / frame_count
        eased = _smoothstep(t)
        v = centered + disp_vis * (float(deform_scale) * eased)
        # Ramp stress colors from blue -> white -> red over time.
        intensity = stresses * eased
        frames.append(
            go.Frame(
                name=f"f{idx}",
                data=[go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=i, j=j, k=k,
                    intensity=intensity,
                    cmin=cmin, cmax=cmax,
                    colorscale=stress_colorscale,
                    flatshading=False,
                    opacity=0.35 + 0.61 * eased,
                    hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>stress=%{intensity:.2f} MPa<extra></extra>",
                    lighting=dict(ambient=0.55, diffuse=0.9, fresnel=0.25, specular=1.0, roughness=0.18),
                    lightposition=dict(x=110, y=180, z=120),
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Stress (MPa)", font=dict(color="#8899aa")),
                        tickfont=dict(color="#8899aa"),
                    ),
                )]
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)
    frame_names = [f"f{idx}" for idx in range(frame_count + 1)]
    # Keep background fixed; only model colors should animate.
    bg_color = "rgb(17,24,32)"

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=bg_color,
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.25, z=1.05)),
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=0, r=0, b=0, t=40),
        font=dict(color="#e0e8f0"),
        title=dict(
            text=(
                f"Surrogate 3D Stress Animation (4s) | Deform: {float(deform_scale):.2f}x"
                + (f" | {title_suffix}" if title_suffix else "")
            ),
            x=0.5,
            font=dict(size=13),
        ),
        # Autoplay configuration
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05,
            "y": 1.15,
            "buttons": [{
                "label": "â–¶ Play 4s Sim",
                "method": "animate",
                "args": [None, {"frame": {"duration": duration_ms, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": False, "mode": "immediate"}]
            }, {
                "label": "â†º Reset",
                "method": "animate",
                "args": [["f0"], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}]
            }]
        }],
        sliders=[{
            "active": 0,
            "x": 0.12,
            "len": 0.76,
            "y": 0.0,
            "steps": [
                {
                    "label": f"{int(100 * idx / frame_count)}%",
                    "method": "animate",
                    "args": [[f"f{idx}"], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}],
                }
                for idx in range(frame_count + 1)
            ],
        }],
        uirevision="mechai-sim",
    )
    fig.layout.update(template=None)
    return fig


def _generate_interactive_3d_html(vertices, faces, stresses, displacements, deform_scale: float, load_type: str, temperature: float = 25.0) -> str:
    """HTML/JS Plotly embed with reliable autoplay animation (4 seconds)."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    stresses = np.asarray(stresses, dtype=np.float32)
    displacements = np.asarray(displacements, dtype=np.float32)

    if len(vertices) == 0 or len(faces) == 0:
        return "<div style='padding:12px;color:#ef4444;'>No mesh to render.</div>"

    # Normalize and stretch stress for visible color contrast.
    smax = max(float(np.max(stresses)), 1e-8)
    sn = stresses / smax
    s_lo, s_hi = np.quantile(sn, [0.05, 0.95]) if len(sn) > 10 else (float(sn.min()), float(sn.max()))
    if abs(s_hi - s_lo) < 1e-8:
        s_lo, s_hi = 0.0, 1.0

    centered = vertices - vertices.mean(axis=0, keepdims=True)
    bbox = centered.max(axis=0) - centered.min(axis=0)
    model_span = float(np.max(bbox))
    # Scale displacement into a visible deformation envelope (magnitude remains dynamic).
    disp_vis = displacements * (0.32 * model_span)

    payload = {
        "v": centered.astype(np.float32).tolist(),
        "f": faces.astype(np.uint32).tolist(),
        "d": disp_vis.astype(np.float32).tolist(),
        "s": sn.astype(np.float32).tolist(),
        "sLo": float(s_lo),
        "sHi": float(s_hi),
        "deformScale": float(deform_scale),
        "loadType": str(load_type),
    }

    # Unique id to avoid collisions on re-render.
    uid = f"mechai_plot_{int(datetime.now().timestamp() * 1000)}"
    payload_json = json.dumps(payload)

    # Keep a fixed dark background; only model colors should change.
    bg_hex = "#111820"

    inner_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:{bg_hex}; font-family:Segoe UI,Arial,sans-serif; }}
button:hover {{ border-color:#22d3a0 !important; color:#22d3a0 !important; }}
</style>
</head>
<body>
<div style="position:relative;width:100%;height:520px;border:1px solid #1e2a3a;border-radius:12px;overflow:hidden;background:{bg_hex};box-sizing:border-box;">  <div id="{uid}" style="width:100%;height:100%;"></div>
  <div id="{uid}_overlay" style="position:absolute;left:14px;bottom:14px;background:rgba(8,12,20,0.78);border:1px solid #1f2a3d;border-radius:10px;padding:6px 10px;color:#dbe6f5;font-family:monospace;font-size:12px;z-index:5;">
    Load: <span id="{uid}_pct">0%</span> | {load_type} | Deform: {deform_scale:.2f}x
  </div>
  <div style="position:absolute;left:14px;top:14px;z-index:6;display:flex;gap:8px;">
    <button id="{uid}_play" style="padding:8px 12px;border-radius:10px;border:1px solid #2a3a4a;background:#0f1319;color:#e0e8f0;font-weight:700;cursor:pointer;">
      Run 4s Animation
    </button>
    <button id="{uid}_reset" style="padding:8px 12px;border-radius:10px;border:1px solid #2a3a4a;background:#0f1319;color:#e0e8f0;font-weight:700;cursor:pointer;">
      Reset
    </button>
  </div>
</div>

<script>
(function() {{
  const payload = {payload_json};
  const divId = "{uid}";
  const pctEl = document.getElementById("{uid}_pct");
  const overlay = document.getElementById("{uid}_overlay");
  let graphDiv = null;

  function smoothstep(t) {{ return t * t * (3 - 2 * t); }}
  function clamp01(x) {{ return Math.max(0, Math.min(1, x)); }}
  function setError(msg) {{
    if (!overlay) return;
    overlay.style.background = "rgba(127,29,29,0.75)";
    overlay.style.borderColor = "#ef4444";
    overlay.textContent = "Plot error: " + msg;
  }}

  function ensurePlotly(cb) {{
    if (window.Plotly) return cb();
    const s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.30.0.min.js";
    s.onload = cb;
    s.onerror = () => setError("Failed to load Plotly");
    document.head.appendChild(s);
  }}

  function buildFrame(t) {{
    const eased = smoothstep(t);
    const ds = payload.deformScale * eased;
    const v = payload.v;
    const d = payload.d;
    const n = v.length;
    const x = new Array(n), y = new Array(n), z = new Array(n), intensity = new Array(n);
    const sLo = payload.sLo, sHi = payload.sHi;
    for (let i = 0; i < n; i++) {{
      x[i] = v[i][0] + d[i][0] * ds;
      y[i] = v[i][1] + d[i][1] * ds;
      z[i] = v[i][2] + d[i][2] * ds;
      const s = clamp01((payload.s[i] - sLo) / Math.max(sHi - sLo, 1e-9));
      intensity[i] = clamp01(s * (0.15 + 0.85 * eased));
    }}
    return {{ x, y, z, intensity }};
  }}

  function renderAt(t) {{
    try {{
      const fr = buildFrame(t);
      if (window.Plotly && graphDiv) {{
        window.Plotly.restyle(graphDiv, {{
          x: [fr.x], y: [fr.y], z: [fr.z], intensity: [fr.intensity]
        }}, [0]);
      }}
      if (pctEl) pctEl.textContent = Math.round(t * 100) + "%";
    }} catch (e) {{
      setError(e && e.message ? e.message : String(e));
    }}
  }}

  function startAnimation() {{
    const start = performance.now();
    const duration = 4000;
    function step(now) {{
      const t = Math.min(1, (now - start) / duration);
      renderAt(t);
      if (t < 1) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
  }}

  ensurePlotly(() => {{
    const div = document.getElementById(divId);
    if (!div) return;
    graphDiv = div;

    const fr0 = buildFrame(0);
    const faces = payload.f;
    const i = faces.map(f => f[0]);
    const j = faces.map(f => f[1]);
    const k = faces.map(f => f[2]);

    const data = [{{
      type: "mesh3d",
      x: fr0.x,
      y: fr0.y,
      z: fr0.z,
      i, j, k,
      intensity: fr0.intensity,
      cmin: 0, cmax: 1,
      colorscale: [[0, "#1d4ed8"], [0.5, "#ffffff"], [1, "#dc2626"]],
      opacity: 0.95,
      flatshading: false,
      lighting: {{ ambient: 0.55, diffuse: 0.9, fresnel: 0.25, specular: 1.0, roughness: 0.18 }},
      lightposition: {{ x: 110, y: 180, z: 120 }},
      colorbar: {{ title: "Stress (Low -> High)" }},
      hovertemplate: "x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<br>stress=%{{intensity:.3f}}<extra></extra>"
    }}];

    const layout = {{
      paper_bgcolor: "{bg_hex}",
      plot_bgcolor: "{bg_hex}",
      margin: {{ l: 0, r: 0, b: 0, t: 40 }},
      font: {{ color: "#e0e8f0" }},
      title: {{
        text: `Surrogate 3D Stress Animation (4s) | ${{payload.loadType}} | Deform: ${{payload.deformScale.toFixed(2)}}x`,
        x: 0.5,
        font: {{ size: 14 }}
      }},
      scene: {{
        xaxis: {{ visible: false }},
        yaxis: {{ visible: false }},
        zaxis: {{ visible: false }},
        bgcolor: "{bg_hex}",
        aspectmode: "data",
        camera: {{ eye: {{ x: 1.45, y: 1.25, z: 1.05 }} }}
      }}
    }};

    try {{
      window.Plotly.newPlot(div, data, layout, {{ displayModeBar: true, responsive: true }});
      renderAt(0);
    }} catch (e) {{
      setError(e && e.message ? e.message : String(e));
      return;
    }}

    const playBtn = document.getElementById("{uid}_play");
    const resetBtn = document.getElementById("{uid}_reset");
    if (playBtn) playBtn.onclick = startAnimation;
    if (resetBtn) resetBtn.onclick = () => renderAt(0);

    // Autoplay on render to match the requested workflow.
    startAnimation();
  }});
}})();
</script>
</body>
</html>"""

    srcdoc = html.escape(inner_html, quote=True)
    return (
        f'<iframe title="MECH-AI 3D Stress Animation" '
        f'style="width:100%;height:540px;border:0;border-radius:12px;background:{bg_hex};" '
        f'sandbox="allow-scripts allow-same-origin" srcdoc="{srcdoc}"></iframe>'
    )

def _generate_simulation_video(vertices, faces, displacements, stresses, load_type, temperature, deform_scale, title_suffix=""):
    """
    Generate a high-quality H.264 MP4 video with incremental load logic.
    Process: 5% to 100% load steps with 4s simulation gaps.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import time
    
    print(f"[Sim] Starting incremental simulation (5% to 100% load)...")
    
    # Video settings
    fps = 15
    total_duration = 4.0 # 4 second final video
    
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"mechai_sim_{int(time.time())}.mp4")
    
    centered = vertices - vertices.mean(axis=0)
    bbox = centered.max(axis=0) - centered.min(axis=0)
    span = np.max(bbox) if np.max(bbox) > 1e-6 else 1.0
    thermal_base = np.clip((float(temperature) - 25) / 400, 0, 1) * 0.22

    fig = plt.figure(figsize=(8, 6), facecolor='#111820')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    
    frames = []
    
    # Incremental logic: 5% to 100% in 20 steps (5% each)
    steps = np.linspace(0.05, 1.0, 20)
    
    try:
        for step_idx, load_factor in enumerate(steps):
            # Requirements: 4-second gap between each load increment
            if step_idx > 0:
                print(f"[Sim] Waiting 4s for load increment {int(load_factor*100)}%...")
                time.sleep(4)
            
            # Render frame for this load level
            # (We use 2 frames per step to reach ~40 frames / 4s at 10-15fps)
            for sub_frame in range(2):
                v_deform = centered + displacements * (float(deform_scale) * load_factor * 0.28 * span)
                intensity = stresses * load_factor + thermal_base
                face_intensity = np.mean(intensity[faces], axis=1)
                face_colors = [turbo_colormap(val) for val in np.clip(face_intensity, 0, 1)]
                
                ax.clear()
                ax.set_facecolor('#111820')
                ax.set_axis_off()
                
                poly = Poly3DCollection(v_deform[faces])
                poly.set_facecolors(face_colors)
                poly.set_edgecolor((0.4, 0.4, 0.4, 0.1))
                ax.add_collection3d(poly)
                
                lim = span * 0.7
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
                
                # Smooth rotation
                angle = 45 + (len(frames) * 1.5)
                ax.view_init(elev=22, azim=angle)
                ax.set_title(f"MECH-AI Incremental Load: {int(load_factor*100)}%", color='#e0e8f0', fontsize=10, pad=-10)

                canvas.draw()
                w, h = fig.canvas.get_width_height()
                img_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
                frames.append(img_rgb)

        # Write video using imageio-ffmpeg (H.264 / avc1)
        print("[Sim] Encoding incremental animation to H.264...")
        imageio.mimwrite(video_path, frames, fps=fps, quality=8, codec='libx264', pixelformat='yuv420p')
        print(f"[Sim] Video generated successfully: {video_path}")
        return video_path
                
    except Exception as e:
        print(f"[Sim] Incremental Video error: {str(e)}")
        # Fallback to GIF
        try:
            gif_path = video_path.replace(".mp4", ".gif")
            imageio.mimwrite(gif_path, frames, fps=fps)
            return gif_path
        except:
            return None
    finally:
        plt.close(fig)


def _generate_compliance_gauge(predictions):
    """Generate compliance gauge visualization."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#111820')
    ax.set_facecolor('#111820')
    
    safety = predictions["safety_factor"]
    
    # Determine color and status based on safety factor
    if safety >= 2.0:
        color = '#22d3a0'
        status = 'SAFE'
        substatus = 'Good safety margin'
    elif safety >= 1.5:
        color = '#3b82f6'
        status = 'ACCEPTABLE'
        substatus = 'Adequate margin'
    elif safety >= 1.0:
        color = '#f59e0b'
        status = 'CAUTION'
        substatus = 'Minimal margin'
    else:
        color = '#ef4444'
        status = 'FAIL'
        substatus = 'Will fail under load'
    
    # Create semi-circular gauge
    theta = np.linspace(0, np.pi, 100)
    r_outer = 1.0
    r_inner = 0.6
    
    # Background arc
    ax.fill_between(np.cos(theta), np.sin(theta), 0, alpha=0.1, color='#2a3a4a')
    
    # Value arc - map safety factor 0-3 to angle
    max_sf = 3.0
    sf_angle = min(safety / max_sf, 1.0) * np.pi
    
    if sf_angle > 0:
        theta_value = np.linspace(0, sf_angle, 50)
        ax.fill_between(np.cos(theta_value), np.sin(theta_value), 0, alpha=0.6, color=color)
    
    # Add tick marks
    for i in range(4):
        angle = i * np.pi / 3
        ax.plot([0.55 * np.cos(angle), 0.65 * np.cos(angle)], 
                [0.55 * np.sin(angle), 0.65 * np.sin(angle)], 
                color='#8899aa', linewidth=2)
        ax.text(0.75 * np.cos(angle), 0.75 * np.sin(angle), str(i), 
                ha='center', va='center', color='#8899aa', fontsize=10, fontweight='bold')
    
    # Center text
    ax.text(0, 0.2, f'{safety:.2f}', ha='center', va='center', 
            color=color, fontsize=28, fontweight='bold', fontfamily='monospace')
    ax.text(0, -0.15, status, ha='center', va='center', 
            color=color, fontsize=14, fontweight='bold')
    ax.text(0, -0.35, substatus, ha='center', va='center', 
            color='#8899aa', fontsize=9)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Safety Factor Gauge', color='#e0e8f0', fontsize=13, fontweight='bold', pad=20)
    
    return fig


def _generate_ai_report(predictions, load, temperature, load_type, deform_scale):
    """Generate AI analysis report."""
    predictions = _sanitize_predictions(predictions)
    safety = predictions["safety_factor"]
    stress = predictions["stress_mpa"]
    strain = predictions["strain"]
    material = predictions["material_label"]
    yield_mpa = predictions["yield_mpa"]
    stress_ratio = predictions.get("stress_ratio", stress / max(yield_mpa, 0.1))
    
    report = f"## AI Analysis Report\n\n"
    report += f"**Material:** {material} (Yield: {yield_mpa:.0f} MPa)\n"
    report += f"**Load Configuration:** {load_type} at {load} N\n"
    report += f"**Deformation Scale:** {deform_scale}x (visualization)\n"
    report += f"**Operating Temperature:** {temperature}Â°C\n\n"
    
    report += f"### Results\n"
    report += f"- **Safety Factor:** `{safety:.2f}`\n"
    report += f"- **Max Stress:** `{stress:.1f} MPa` ({stress_ratio*100:.1f}% of yield)\n"
    report += f"- **Strain:** `{strain:.6f}`\n\n"
    
    if safety >= 2.0:
        report += "âœ… **Design is safe** with excellent safety margin. No action required.\n"
        report += f"- Can withstand up to ~{int(load * safety)} N before reaching yield.\n"
    elif safety >= 1.5:
        report += "âœ“ **Design is acceptable** with adequate safety margin for static loads.\n"
        report += "- Consider increasing margin for cyclic/dynamic loading.\n"
    elif safety >= 1.0:
        report += "**Design meets minimum requirements** but has low safety margin.\n"
        report += "- **Recommendation:** Increase cross-section or reduce load.\n"
        report += "- Monitor closely under fatigue conditions.\n"
    else:
        report += "**Design will FAIL** under current conditions.\n"
        report += f"- Stress exceeds yield by {(1-safety)*100:.1f}%.\n"
        report += "- **Immediate action required:** Upgrade material or reduce load.\n"
    
    # Add load-specific notes
    if load_type == "Tensile":
        report += "\n**Tensile Note:** Check for necking at stress concentrations."
    elif load_type == "Torsion":
        report += "\n**Torsion Note:** Verify shear stress in thin-walled sections."
    elif load_type == "Compressive":
        report += "\n**Compression Note:** Check local buckling and crushing zones."
    
    return report


# -----------------------------------------------------------------------------
# Export functions for Frontend integration
# -----------------------------------------------------------------------------
def get_simulation_html(component_type: str, temperature: float, load: float,
                       load_type: str = "Tensile", deform_scale: float = 1.0,
                       stl_path: str = None) -> str:
    """Generate simulation HTML for embedding in Gradio HTML component."""
    try:
        if stl_path and os.path.exists(stl_path):
            vertices, faces, normals = load_stl_as_buffergeometry(stl_path)
        else:
            vertices, faces = _create_dummy_mesh(component_type)
        
        vertices, faces = decimate_mesh(vertices, faces, max_vertices=5000)
        stress_proxy, displacements = predict_physics(
            {"vertices": vertices, "faces": faces},
            load_type,
            load_n=load,
            temperature_c=temperature,
        )

        # Convert surrogate proxy -> normalized [0,1] for the lightweight Three.js HTML.
        # (This path is kept for compatibility; the main UI uses Plotly figures.)
        smax = float(np.max(stress_proxy)) if len(stress_proxy) else 0.0
        stresses = (stress_proxy / smax).astype(np.float32) if smax > 1e-9 else np.zeros_like(stress_proxy, dtype=np.float32)

        return generate_simulation_html(vertices, faces, displacements, stresses,
                                       load_type, deform_scale, temperature)
    except Exception as e:
        return f"<html><body style='background:#0f1319;color:#ef4444;padding:20px;'>Simulation Error: {str(e)}</body></html>"


def generate_simulation_report_pdf(
    component_type: str,
    temperature: float,
    load: float,
    load_type: str = "Tensile",
    deform_scale: float = 1.0,
    stl_path: str = None,
) -> str:
    """Generate a polished simulation report as a PDF file."""
    print(f"[Sim] Starting PDF Report generation for {component_type}...")
    try:
        plt.close("all")

        print("[Sim] Running simulation for report data...")
        result = run_simulation(
            component_type=component_type,
            temperature=temperature,
            load=load,
            load_type=load_type,
            deform_scale=deform_scale,
        )
        (
            safety_factor, stress_mpa, strain, yield_mpa,
            sf_curve, deform_curve, heatmap_plot, compliance_gauge,
            ai_report, _failure_panel, _video
        ) = result

        report_text = re.sub(r"[*`#_>-]", "", str(ai_report))
        report_text = report_text.encode("ascii", "ignore").decode("ascii")
        report_text = report_text.replace("AI Analysis Report", "AI Analysis")
        report_lines = [line.strip(" -") for line in report_text.splitlines() if line.strip()]

        fd, pdf_path = tempfile.mkstemp(prefix="mechai_report_", suffix=".pdf")
        os.close(fd)

        def footer(fig):
            fig.text(0.5, 0.025, "Made by MECH-AI", ha="center", va="center", fontsize=9, color="#617080", fontweight="bold")
            fig.text(0.88, 0.025, datetime.now().strftime("%Y-%m-%d %H:%M"), ha="right", va="center", fontsize=8, color="#8a96a3")

        def card(ax, x, y, w, h, title, lines, accent="#22d3a0"):
            ax.add_patch(plt.Rectangle((x, y - h), w, h, transform=ax.transAxes, facecolor="#f7fafc", edgecolor="#d5dee8", linewidth=1.0))
            ax.add_patch(plt.Rectangle((x, y - 0.035), w, 0.035, transform=ax.transAxes, facecolor=accent, edgecolor=accent, linewidth=0))
            ax.text(x + 0.018, y - 0.024, title.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold", color="#ffffff", va="center")
            cy = y - 0.065
            for line in lines:
                ax.text(x + 0.02, cy, line, transform=ax.transAxes, fontsize=10.5, color="#202a36", va="top")
                cy -= 0.034

        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), facecolor="#ffffff")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            ax.add_patch(plt.Rectangle((0, 0.875), 1, 0.125, transform=ax.transAxes, facecolor="#111820", edgecolor="none"))
            ax.add_patch(plt.Rectangle((0, 0.872), 1, 0.006, transform=ax.transAxes, facecolor="#22d3a0", edgecolor="none"))
            ax.text(0.055, 0.94, "MECH-AI Engineering Analysis", transform=ax.transAxes, fontsize=18, fontweight="bold", color="#e7eef7", va="center")
            ax.text(0.058, 0.902, "Structural simulation report", transform=ax.transAxes, fontsize=10, color="#9fb0c2", va="center")

            status = "SAFE" if float(safety_factor) >= 2 else "CAUTION" if float(safety_factor) >= 1 else "FAIL"
            status_color = "#22d3a0" if status == "SAFE" else "#f59e0b" if status == "CAUTION" else "#ef4444"
            ax.add_patch(plt.Rectangle((0.79, 0.92), 0.16, 0.042, transform=ax.transAxes, facecolor=status_color, edgecolor="none"))
            ax.text(0.87, 0.941, status, transform=ax.transAxes, ha="center", va="center", fontsize=11, fontweight="bold", color="#ffffff")
            ax.text(0.95, 0.898, datetime.now().strftime("%Y-%m-%d %H:%M"), transform=ax.transAxes, fontsize=9, color="#9fb0c2", ha="right", va="center")

            card(
                ax, 0.055, 0.825, 0.41, 0.18, "System Configuration",
                [
                    f"Component: {str(component_type).upper()}",
                    f"Load: {float(load):,.0f} N",
                    f"Load type: {load_type}",
                    f"Temperature: {float(temperature):.1f} C",
                ],
                accent="#3b82f6",
            )
            card(
                ax, 0.535, 0.825, 0.41, 0.18, "Performance Metrics",
                [
                    f"Safety factor: {float(safety_factor):.3f}",
                    f"Peak stress: {float(stress_mpa):.2f} MPa",
                    f"Strain: {float(strain):.6f}",
                    f"Material yield: {float(yield_mpa):.1f} MPa",
                ],
                accent=status_color,
            )

            ax.text(0.055, 0.585, "AI Assessment", transform=ax.transAxes, fontsize=14, fontweight="bold", color="#111820")
            ax.add_patch(plt.Rectangle((0.055, 0.105), 0.89, 0.455, transform=ax.transAxes, facecolor="#fbfcfe", edgecolor="#d5dee8", linewidth=1.0))
            y = 0.532
            for line in report_lines[:22]:
                for wrapped in textwrap.wrap(line, width=92) or [""]:
                    if y < 0.13:
                        break
                    ax.text(0.082, y, wrapped, transform=ax.transAxes, fontsize=9.7, color="#293545", va="top")
                    y -= 0.026
                if y < 0.13:
                    break

            footer(fig)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print("[Sim] Appending visual plots to PDF...")
            for plot_fig in [sf_curve, deform_curve, heatmap_plot, compliance_gauge]:
                if plot_fig is not None:
                    try:
                        plot_fig.set_facecolor("#ffffff")
                        footer(plot_fig)
                        pdf.savefig(plot_fig, bbox_inches="tight")
                    except Exception as e:
                        print(f"[Sim] Plot save error: {e}")
                    finally:
                        plt.close(plot_fig)

        print("[Sim] Report generation complete.")
        return pdf_path

    except Exception as e:
        print(f"[Sim] CRITICAL REPORT ERROR: {str(e)}")
        fd, err_pdf = tempfile.mkstemp(prefix="mechai_error_", suffix=".pdf")
        os.close(fd)
        try:
            with PdfPages(err_pdf) as pdf:
                fig, ax = plt.subplots(figsize=(8, 6), facecolor="#ffffff")
                ax.axis("off")
                ax.text(0.08, 0.72, "MECH-AI Report Failed", fontsize=18, fontweight="bold", color="#ef4444")
                ax.text(0.08, 0.55, textwrap.fill(str(e), width=80), fontsize=11, color="#202a36")
                fig.text(0.5, 0.04, "Made by MECH-AI", ha="center", fontsize=9, color="#617080", fontweight="bold")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            return err_pdf
        except Exception:
            return None

def build_simulation_tab(default_component: str = "bolt"):
    """
    Standalone Gradio simulation tab.
    Keep this isolated from build123d design/modeling functions.
    """
    import gradio as gr

    def _render_simulation_html(load_type: str, deform_scale: float):
        stl_path = getattr(sim_state, "last_stl_path", None)
        return get_simulation_html(
            component_type=getattr(sim_state, "last_detected_object", default_component),
            temperature=25.0,
            load=1000.0,
            load_type=load_type,
            deform_scale=deform_scale,
            stl_path=stl_path
        )

    with gr.Tab("Simulation (Three.js Surrogate)"):
        gr.Markdown("Run a lightweight 4-second GNN-inspired surrogate simulation.")
        with gr.Row():
            load_type = gr.Dropdown(choices=LOAD_TYPES, value="Tensile", label="Load Type")
            deform_scale = gr.Slider(minimum=0.1, maximum=8.0, step=0.1, value=1.0, label="Deformation Scale")
            run_btn = gr.Button("Run Simulation", variant="primary")
        sim_html = gr.HTML(label="Simulation View")
        run_btn.click(fn=_render_simulation_html, inputs=[load_type, deform_scale], outputs=sim_html)

