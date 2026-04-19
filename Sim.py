"""Simulation: materials, engine, Plotly plots, app state, and ML-driven handlers."""
import os
import math

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION CONFIGURATION & CONSTANTS (from ok.py)
# ═══════════════════════════════════════════════════════════════════════════

MATERIALS: Dict[str, Dict] = {
    "steel": {
        "label": "Structural Steel",
        "density_g_cm3": 7.85,
        "yield_mpa": 250.0,
        "ultimate_mpa": 400.0,
        "youngs_gpa": 200.0,
        "poisson": 0.29,
        "color": "#8a9bb0",
        "standard": "ASME_VIII_D1"
    },
    "aluminum": {
        "label": "Aluminum 6061-T6",
        "density_g_cm3": 2.70,
        "yield_mpa": 276.0,
        "ultimate_mpa": 310.0,
        "youngs_gpa": 69.0,
        "poisson": 0.33,
        "color": "#c8d8e8",
        "standard": "ISO_9001"
    },
    "titanium": {
        "label": "Titanium Grade 5",
        "density_g_cm3": 4.43,
        "yield_mpa": 880.0,
        "ultimate_mpa": 950.0,
        "youngs_gpa": 114.0,
        "poisson": 0.31,
        "color": "#9ab0c8",
        "standard": "ASME_VIII_D2"
    },
    "abs_plastic": {
        "label": "ABS Plastic",
        "density_g_cm3": 1.05,
        "yield_mpa": 40.0,
        "ultimate_mpa": 45.0,
        "youngs_gpa": 2.3,
        "poisson": 0.35,
        "color": "#f0e8d0",
        "standard": "ISO_9001"
    },
    "nylon": {
        "label": "Nylon PA66",
        "density_g_cm3": 1.14,
        "yield_mpa": 82.0,
        "ultimate_mpa": 95.0,
        "youngs_gpa": 3.0,
        "poisson": 0.39,
        "color": "#e8f0d8",
        "standard": "ISO_9001"
    },
    "copper": {
        "label": "Copper C101",
        "density_g_cm3": 8.96,
        "yield_mpa": 70.0,
        "ultimate_mpa": 220.0,
        "youngs_gpa": 117.0,
        "poisson": 0.33,
        "color": "#d4956a",
        "standard": "ASME_VIII_D1"
    },
}

SAFETY_STANDARDS = {
    "ASME_VIII_D1": {"tensile_sf": 3.5, "yield_sf": 1.5, "min_sf": 2.0},
    "ASME_VIII_D2": {"tensile_sf": 2.4, "yield_sf": 1.5, "min_sf": 1.5},
    "ISO_9001": {"generic_sf": 2.0, "min_sf": 1.8},
    "AISC_LRFD": {"phi": 0.9, "min_sf": 1.67},
}


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION DATA CLASSES (from ok.py)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    object_type: str
    material_key: str
    material_label: str
    volume_cm3: float = 0.0
    mass_g: float = 0.0
    mass_kg: float = 0.0
    stress_mpa: float = 0.0
    max_stress_mpa: float = 0.0
    deformation_mm: float = 0.0
    safety_factor: float = 0.0
    standard: str = "ASME_VIII_D1"
    compliance_status: str = "UNKNOWN"
    compliance_color: str = "#808080"
    stress_ratio: float = 0.0
    yield_mpa: float = 250.0
    load_curve: List[Tuple[float, float]] = field(default_factory=list)
    displacement_curve: List[Tuple[float, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE (from ok.py)
# ═══════════════════════════════════════════════════════════════════════════

class SimulationEngine:
    def run(self, object_type: str, dims: dict, material: str = "steel",
            force_N: float = 500.0, include_optimised: bool = True) -> SimResult:
        mat = MATERIALS.get(material, MATERIALS["steel"])
        mat["_key"] = material
        result = self._kernel(object_type, dims, mat, force_N)
        if include_optimised:
            result.optimised = self._optimise(result, dims, mat, force_N)
        return result

    def _kernel(self, obj_type: str, dims: dict, material: dict, force_N: float) -> SimResult:
        mat_label = material["label"]
        density = material["density_g_cm3"]
        yield_mpa = material["yield_mpa"]
        youngs_gpa = material["youngs_gpa"]

        vol_mm3 = 0.0
        area_mm2 = 1.0
        char_len = 10.0
        notes = []

        if obj_type == "nut":
            inner_r = dims.get("inner_radius", 5.0)
            outer_r = dims.get("outer_radius", 9.0)
            t = dims.get("thickness", 7.0)
            is_hex = dims.get("is_hex", True)
            if is_hex:
                side = outer_r
                area = (3 * math.sqrt(3) / 2) * side ** 2
            else:
                area = math.pi * outer_r ** 2
            vol_mm3 = area * t - math.pi * inner_r ** 2 * t
            area_mm2 = math.pi * (outer_r ** 2 - inner_r ** 2)
            char_len = t
            notes.append(f"{'Hex' if is_hex else 'Circular'} nut: inner={inner_r*2}mm, outer={outer_r*2}mm, t={t}mm")

        elif obj_type == "bolt":
            shaft_r = dims.get("shaft_radius", 5.0)
            shaft_len = dims.get("shaft_length", 30.0)
            head_r = dims.get("head_radius", 9.0)
            head_h = dims.get("head_height", 5.0)
            vol_mm3 = math.pi * shaft_r ** 2 * shaft_len + math.pi * head_r ** 2 * head_h
            area_mm2 = math.pi * shaft_r ** 2
            char_len = shaft_len
            notes.append(f"Bolt: shaft={shaft_r*2}mm×{shaft_len}mm, head={head_r*2}mm×{head_h}mm")

        elif obj_type == "washer":
            inner_r = dims.get("inner_radius", 4.0)
            outer_r = dims.get("outer_radius", 10.0)
            t = dims.get("thickness", 2.0)
            vol_mm3 = math.pi * (outer_r ** 2 - inner_r ** 2) * t
            area_mm2 = math.pi * (outer_r ** 2 - inner_r ** 2)
            char_len = t
            notes.append(f"Washer: inner={inner_r*2}mm, outer={outer_r*2}mm, t={t}mm")

        elif obj_type == "bracket":
            base_l = dims.get("base_length", 50.0)
            base_w = dims.get("base_width", 30.0)
            wall_h = dims.get("wall_height", 40.0)
            t = dims.get("thickness", 5.0)
            vol_mm3 = base_l * base_w * t + base_l * t * wall_h
            area_mm2 = base_l * t
            char_len = wall_h
            notes.append(f"L-bracket: base={base_l}×{base_w}mm, wall={wall_h}mm, t={t}mm")

        elif obj_type == "cylinder":
            r = dims.get("radius", 15.0)
            h = dims.get("height", 40.0)
            vol_mm3 = math.pi * r ** 2 * h
            area_mm2 = math.pi * r ** 2
            char_len = h
            notes.append(f"Cylinder: r={r}mm, h={h}mm")

        elif obj_type == "screw":
            shaft_r = dims.get("shaft_radius", 3.0)
            shaft_len = dims.get("shaft_length", 20.0)
            head_r = dims.get("head_radius", 5.0)
            head_h = dims.get("head_height", 3.0)
            vol_mm3 = math.pi * shaft_r ** 2 * shaft_len + math.pi * head_r ** 2 * head_h
            area_mm2 = math.pi * shaft_r ** 2
            char_len = shaft_len
            notes.append(f"Screw: shaft={shaft_r*2}mm×{shaft_len}mm, head={head_r*2}mm×{head_h}mm")

        elif obj_type == "plate":
            l = dims.get("length", 80.0)
            w = dims.get("width", 60.0)
            t = dims.get("thickness", 8.0)
            vol_mm3 = l * w * t
            area_mm2 = l * w
            char_len = t
            notes.append(f"Plate: {l}×{w}×{t}mm")

        else:
            r = dims.get("radius", 10.0)
            h = dims.get("height", 20.0)
            vol_mm3 = math.pi * r ** 2 * h
            area_mm2 = math.pi * r ** 2
            char_len = h
            notes.append(f"Generic cylinder: r={r}mm, h={h}mm")

        area_mm2 = max(area_mm2, 0.1)
        vol_cm3 = vol_mm3 / 1000.0
        mass_g = vol_cm3 * density
        stress_mpa = force_N / area_mm2
        E_mpa = youngs_gpa * 1000.0
        deform_mm = (stress_mpa / E_mpa) * char_len if E_mpa > 0 else 0
        sf = yield_mpa / stress_mpa if stress_mpa > 0 else 99.0

        load_curve = []
        displacement_curve = []
        max_load = force_N * 2.5

        for i in range(21):
            load = (max_load / 20) * i
            s = load / area_mm2
            d = (s / E_mpa) * char_len if E_mpa > 0 else 0
            sf_i = yield_mpa / s if s > 0 else 99.0
            load_curve.append((load, sf_i))
            displacement_curve.append((load, d))

        standard_key = material.get("standard", "ASME_VIII_D1")
        standard = SAFETY_STANDARDS.get(standard_key, SAFETY_STANDARDS["ASME_VIII_D1"])
        min_sf = standard.get("min_sf", 2.0)

        if sf >= min_sf:
            status, color = "✅ PASS", "#22d3a0"
        elif sf >= 1.0:
            status, color = "⚠️ WARNING", "#f59e0b"
        else:
            status, color = "❌ FAIL", "#ef4444"

        notes.extend([
            f"Applied force: {force_N:.0f} N | Cross-section: {area_mm2:.1f} mm²",
            f"Yield strength: {yield_mpa} MPa | E = {youngs_gpa} GPa"
        ])

        return SimResult(
            object_type=obj_type,
            material_key=material.get("_key", "unknown"),
            material_label=mat_label,
            volume_cm3=round(vol_cm3, 3),
            mass_g=round(mass_g, 2),
            mass_kg=round(mass_g / 1000, 4),
            stress_mpa=round(stress_mpa, 3),
            max_stress_mpa=round(stress_mpa * 1.5, 3),
            deformation_mm=round(deform_mm, 4),
            safety_factor=round(sf, 2),
            standard=standard_key,
            compliance_status=status,
            compliance_color=color,
            stress_ratio=min(stress_mpa / yield_mpa, 1.0),
            yield_mpa=yield_mpa,
            load_curve=load_curve,
            displacement_curve=displacement_curve,
            notes=notes
        )

    def _optimise(self, original: SimResult, dims: dict, material: dict, force_N: float):
        opt_dims = {k: v * 1.2 if isinstance(v, (int, float)) else v for k, v in dims.items()}
        return self._kernel(original.object_type, opt_dims, material, force_N)

    def summary_to_dims(self, object_type: str, raw_dims: List[float]) -> dict:
        d = raw_dims
        get = lambda i, default=10.0: d[i] if i < len(d) else default

        mapping = {
            "nut": {
                "inner_radius": get(0, 10) / 2,
                "outer_radius": get(1, 20) / 2,
                "thickness": get(2, 8),
                "is_hex": True
            },
            "bolt": {
                "shaft_radius": get(0, 10) / 2,
                "shaft_length": get(1, 30),
                "head_radius": get(2, 18) / 2,
                "head_height": get(3, 5)
            },
            "screw": {
                "shaft_radius": get(0, 6) / 2,
                "shaft_length": get(1, 20),
                "head_radius": get(2, 10) / 2,
                "head_height": get(3, 3)
            },
            "washer": {
                "inner_radius": get(0, 10) / 2,
                "outer_radius": get(1, 20) / 2,
                "thickness": get(2, 2)
            },
            "bracket": {
                "base_length": get(0, 50),
                "base_width": get(1, 30),
                "wall_height": get(2, 40),
                "thickness": get(3, 5)
            },
            "cylinder": {
                "radius": get(0, 20) / 2,
                "height": get(1, 40)
            },
            "plate": {
                "length": get(0, 80),
                "width": get(1, 60),
                "thickness": get(2, 8)
            }
        }
        return mapping.get(object_type, {"radius": get(0, 10), "height": get(1, 20)})


# ═══════════════════════════════════════════════════════════════════════════
# AI EXPLAINER (from ok.py)
# ═══════════════════════════════════════════════════════════════════════════

class AIExplainer:
    def __init__(self):
        self.material_explanations = {
            "steel": "Steel provides excellent strength ({yield_strength} MPa yield) and fatigue resistance. Ideal for high-load structural applications.",
            "aluminum": "Aluminum 6061-T6 offers superior strength-to-weight ratio ({yield_strength} MPa). Perfect for aerospace and automotive where weight matters.",
            "titanium": "Titanium Grade 5 delivers exceptional strength ({yield_strength} MPa) with corrosion resistance. Premium choice for critical applications.",
            "abs_plastic": "ABS plastic is cost-effective for low loads ({yield_strength} MPa). Great for housings and prototypes.",
            "nylon": "Nylon is self-lubricating with {yield_strength} MPa strength. Excellent for wear parts and bushings.",
            "copper": "Copper provides thermal conductivity with {yield_strength} MPa strength. Used for heat dissipation components."
        }
        self.stress_explanations = {
            "nut": "🔴 **High Stress Zone**: Thread roots and hex corners. The sharp geometry creates stress concentration (Kt≈3.5).",
            "bolt": "🔴 **Critical Sections**: Thread roots (Kt≈2.8) and head-shoulder fillet. Sudden section change causes stress spike.",
            "screw": "🔴 **Critical Sections**: Thread roots and head fillet carry peak stress. Add generous fillets for fatigue life.",
            "washer": "🟢 **Uniform Distribution**: Simple geometry spreads load evenly. Minor edge effects at inner bore.",
            "bracket": "🔴 **Bending Zone**: Fillet between base and wall carries maximum bending moment. Add generous fillets here.",
            "cylinder": "🟡 **Surface Stress**: Outer surface in compression. Any scratches act as stress risers.",
            "plate": "🟡 **Bending Stress**: Distributed load creates bending at supports. Thickness drives stiffness."
        }

    def explain_material(self, material_key: str, yield_mpa: float) -> str:
        template = self.material_explanations.get(material_key, "Material selected for mechanical properties.")
        return template.format(yield_strength=yield_mpa)

    def explain_stress(self, object_type: str) -> str:
        return self.stress_explanations.get(object_type, "Stress follows elastic theory for this geometry.")

    def generate_report(self, params: dict, result: SimResult, failure_point: float = None,
                        failure_mode: str = None) -> str:
        lines = []
        lines.append(f"## 🤖 AI Analysis Report: {result.object_type.upper()}")
        lines.append("")
        lines.append(f"### 🧪 Material: {result.material_label}")
        lines.append(self.explain_material(result.material_key, result.yield_mpa))
        lines.append(f"- **Yield Strength**: {result.yield_mpa} MPa")
        lines.append(f"- **Young's Modulus**: {MATERIALS[result.material_key]['youngs_gpa']} GPa")
        lines.append(f"- **Density**: {MATERIALS[result.material_key]['density_g_cm3']} g/cm³")
        lines.append("")
        lines.append(f"### 📊 Current State (Load: {params.get('load', 500)} N)")
        lines.append(f"| Metric | Value | Status |")
        lines.append(f"|--------|-------|--------|")
        lines.append(f"| Stress | {result.stress_mpa:.2f} MPa | {self._stress_status(result)} |")
        lines.append(f"| Safety Factor | {result.safety_factor:.2f} | {self._sf_status(result)} |")
        lines.append(f"| Strain | {result.deformation_mm:.6f} | Elastic |")
        lines.append(f"| Yield Point | {result.mass_g:.2f} MPa | Reference |")
        lines.append("")
        lines.append("### 🔍 Stress Concentration Analysis")
        lines.append(self.explain_stress(result.object_type))
        lines.append("")
        if failure_point:
            lines.append("### ⚠️ FAILURE PREDICTION")
            lines.append(f"**Predicted Failure Load**: `{failure_point:.1f} N`")
            lines.append(f"**Failure Mode**: `{failure_mode}`")
            lines.append(f"**Current Margin**: `{((failure_point - params.get('load', 500)) / failure_point * 100):.1f}%`")
            lines.append("")
            lines.append("### 💡 AI Optimization Recommendations")
            if result.safety_factor < 2.0:
                lines.append("1. **⬆️ Increase outer diameter by 15%** → Reduces stress by ~30%")
                lines.append(f"2. **🔄 Upgrade material** → Switch to {self._suggest_upgrade(result.material_key)}")
                lines.append("3. **📐 Add fillets** → 2mm radius at corners reduces Kt by 40%")
            lines.append("")
        return "\n".join(lines)

    def _stress_status(self, result: SimResult) -> str:
        ratio = result.stress_mpa / result.yield_mpa
        if ratio < 0.5: return "🟢 Low"
        elif ratio < 0.8: return "🟡 Moderate"
        else: return "🔴 High"

    def _sf_status(self, result: SimResult) -> str:
        if result.safety_factor >= 2.0: return "🟢 Safe"
        elif result.safety_factor >= 1.5: return "🟡 Marginal"
        else: return "🔴 Critical"

    def _suggest_upgrade(self, current: str) -> str:
        upgrades = {"abs_plastic": "nylon", "nylon": "aluminum", "aluminum": "steel",
                    "steel": "titanium", "copper": "steel"}
        return upgrades.get(current, "higher grade alloy")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION VISUALIZATION HELPERS (from ok.py)
# ═══════════════════════════════════════════════════════════════════════════

def create_sf_plot(history: List[Dict], standard: str) -> go.Figure:
    fig = go.Figure()
    if history:
        loads = [h["load"] for h in history]
        sfs = [h["sf"] for h in history]
        fig.add_trace(go.Scatter(x=loads, y=sfs, mode='lines+markers',
                                 line=dict(color='#22d3a0', width=3), name='SF'))
        fig.add_trace(go.Scatter(x=[loads[-1]], y=[sfs[-1]], mode='markers',
                                 marker=dict(size=15, color='#ef4444', symbol='star'), name='Current'))
    min_sf = SAFETY_STANDARDS.get(standard, {}).get('min_sf', 2.0)
    fig.add_hline(y=min_sf, line_dash="dash", line_color="#f59e0b", annotation_text=f"Min SF ({min_sf})")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#ef4444", annotation_text="Yield")
    fig.update_layout(template="plotly_dark", title="Load vs Safety Factor",
                      xaxis_title="Load (N)", yaxis_title="Safety Factor", height=300,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def create_deform_plot(history: List[Dict]) -> go.Figure:
    fig = go.Figure()
    if history:
        loads = [h["load"] for h in history]
        deforms = [h["deform"] for h in history]
        fig.add_trace(go.Scatter(x=loads, y=deforms, mode='lines+markers', fill='tozeroy',
                                 fillcolor='rgba(59,130,246,0.2)', line=dict(color='#3b82f6', width=3)))
    fig.update_layout(template="plotly_dark", title="Load vs Strain",
                      xaxis_title="Load (N)", yaxis_title="Strain", height=300,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def create_heatmap(result: SimResult, obj_type: str) -> go.Figure:
    obj = str(obj_type).strip().lower()
    ratio = float(max(0.01, result.stress_ratio))
    colors = [[0, '#0088ff'], [0.25, '#00ddcc'], [0.5, '#00ff66'], [0.75, '#ffcc00'], [1, '#ff2222']]
    cmax = 1.0
    fig = go.Figure()

    def _normalize_stress(s):
        arr = np.array(s, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return arr
        smin = np.nanmin(arr[finite])
        smax = np.nanmax(arr[finite])
        if abs(smax - smin) < 1e-9:
            arr[finite] = 0.35
            return arr
        arr[finite] = (arr[finite] - smin) / (smax - smin)
        return np.clip(arr, 0.0, 1.0)

    def _add_surface(x, y, z, s, showscale=False):
        s = _normalize_stress(s)
        surface_kwargs = dict(
            x=x, y=y, z=z, surfacecolor=s,
            colorscale=colors, cmin=0.0, cmax=cmax,
            showscale=showscale,
            hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<br>Stress/Yield=%{surfacecolor:.3f}<extra></extra>",
        )
        if showscale:
            surface_kwargs["colorbar"] = dict(title='Stress/Yield', thickness=15)
        fig.add_trace(go.Surface(**surface_kwargs))

    def _cyl_shell(radius=8.0, length=26.0, z0=0.0, n_theta=84, n_z=42, amp=None):
        theta = np.linspace(0, 2 * np.pi, n_theta)
        z = np.linspace(z0, z0 + length, n_z)
        theta, z = np.meshgrid(theta, z)
        r = radius if amp is None else (radius + amp(theta, z))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z, theta

    def _disc(radius=8.0, z0=0.0, n_theta=96, n_r=34):
        theta = np.linspace(0, 2 * np.pi, n_theta)
        r = np.linspace(0.0, radius, n_r)
        theta, r = np.meshgrid(theta, r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(x, z0)
        return x, y, z, r

    def _ring_disc(r_in=4.0, r_out=10.0, z0=0.0, n_theta=96, n_r=32):
        theta = np.linspace(0, 2 * np.pi, n_theta)
        r = np.linspace(r_in, r_out, n_r)
        theta, r = np.meshgrid(theta, r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(x, z0)
        return x, y, z, r, theta

    def _add_cylinder(radius, length, hotspot=0.65, z0=0.0, threaded=False):
        x, y, z, theta = _cyl_shell(radius=radius, length=length, z0=z0, amp=(lambda t, zz: 0.45 * np.sin(6.0 * t + 0.9 * zz)) if threaded else None)
        axial = z / max(1e-6, z0 + length)
        s_shell = ratio * (0.20 + 0.55 * axial + 0.35 * np.exp(-((axial - hotspot) ** 2) / 0.02))
        if threaded:
            s_shell *= (1.0 + 0.18 * np.sin(6.0 * theta + 0.9 * z))
        _add_surface(x, y, z, s_shell, showscale=True)
        x1, y1, z1, r1 = _disc(radius, z0=z0)
        s1 = ratio * (0.15 + 0.40 * (r1 / radius))
        _add_surface(x1, y1, z1, s1)
        x2, y2, z2, r2 = _disc(radius, z0=z0 + length)
        s2 = ratio * (0.30 + 0.50 * (r2 / radius))
        _add_surface(x2, y2, z2, s2)

    try:
        if obj in ["cylinder", "shaft", "bushing", "coupling"]:
            _add_cylinder(radius=8.0, length=28.0, hotspot=0.7)
        elif obj in ["bolt", "rivet"]:
            _add_cylinder(radius=4.2, length=28.0, hotspot=0.2 if obj == "rivet" else 0.35)
            _add_cylinder(radius=8.0, length=6.0, hotspot=0.5, z0=28.0)  # head
        elif obj == "screw":
            _add_cylinder(radius=3.8, length=30.0, hotspot=0.35, threaded=True)
            _add_cylinder(radius=6.8, length=5.0, hotspot=0.5, z0=30.0)  # head
        elif obj == "nut":
            # Hex nut: outer hex prism + inner cylindrical bore + top/bottom annular faces.
            r_in, r_out, h = 4.5, 10.0, 6.5

            def _hex_radius(theta_vals, flat_radius):
                # Polar radius of regular hexagon with flats at 0, 60, ...
                a = (theta_vals + np.pi / 6) % (np.pi / 3) - np.pi / 6
                return flat_radius * np.cos(np.pi / 6) / np.maximum(np.cos(a), 1e-6)

            theta = np.linspace(0, 2 * np.pi, 180)
            zz = np.linspace(0.0, h, 40)
            theta, zz = np.meshgrid(theta, zz)
            r_hex = _hex_radius(theta, r_out)
            xh = r_hex * np.cos(theta)
            yh = r_hex * np.sin(theta)
            zh = zz

            corner_hotspot = np.clip(np.cos(3 * theta) ** 8, 0.0, 1.0)
            axial = zz / h
            s_outer = ratio * (0.22 + 0.55 * corner_hotspot + 0.18 * axial)
            _add_surface(xh, yh, zh, s_outer, showscale=True)

            xi, yi, zi, _ = _cyl_shell(radius=r_in, length=h, z0=0.0)
            # Higher stress around bore where threads/load transfer happen.
            s_inner = ratio * (0.55 + 0.35 * np.exp(-((zi / h - 0.5) ** 2) / 0.08))
            _add_surface(xi, yi, zi, s_inner)

            rt = np.linspace(0.0, 1.0, 46)
            tt = np.linspace(0, 2 * np.pi, 180)
            rt, tt = np.meshgrid(rt, tt)
            rhex_top = _hex_radius(tt, r_out)
            rr = r_in + (rhex_top - r_in) * rt
            xt = rr * np.cos(tt)
            yt = rr * np.sin(tt)
            zt = np.full_like(xt, h)
            radial_top = (rr - r_in) / np.maximum(rhex_top - r_in, 1e-6)
            s_top = ratio * (0.35 + 0.60 * (1.0 - radial_top))  # redder near bore
            _add_surface(xt, yt, zt, s_top)

            xb = rr * np.cos(tt)
            yb = rr * np.sin(tt)
            zb = np.full_like(xb, 0.0)
            s_bot = ratio * (0.28 + 0.45 * (1.0 - radial_top))
            _add_surface(xb, yb, zb, s_bot)
        elif obj in ["washer", "bearing", "pulley", "sprocket", "gear"]:
            # Dedicated geometries so pulley/bearing always render and gear looks like spur gear.
            if obj == "washer":
                r_in, r_out, h = 5.5, 11.0, 2.4
                x, y, z, _ = _cyl_shell(radius=r_out, length=h, z0=0.0)
                _add_surface(x, y, z, ratio * (0.22 + 0.55 * (z / h)), showscale=True)
                xi, yi, zi, _ = _cyl_shell(radius=r_in, length=h, z0=0.0)
                _add_surface(xi, yi, zi, ratio * (0.65 - 0.20 * (zi / h)))
                xt, yt, zt, rt, _ = _ring_disc(r_in, r_out, z0=h)
                _add_surface(xt, yt, zt, ratio * (0.25 + 0.65 * ((r_out - rt) / (r_out - r_in))))
                xb, yb, zb, rb, _ = _ring_disc(r_in, r_out, z0=0.0)
                _add_surface(xb, yb, zb, ratio * (0.18 + 0.45 * (rb / r_out)))

            elif obj == "bearing":
                # Outer race, inner race and raceway groove profile.
                r_bore, r_inner, r_outer, h = 4.5, 8.0, 13.0, 7.0
                xo, yo, zo, _ = _cyl_shell(radius=r_outer, length=h, z0=0.0)
                _add_surface(xo, yo, zo, ratio * (0.25 + 0.55 * np.exp(-((zo / h - 0.5) ** 2) / 0.04)), showscale=True)
                xi, yi, zi, _ = _cyl_shell(radius=r_inner, length=h, z0=0.0)
                _add_surface(xi, yi, zi, ratio * (0.30 + 0.45 * np.exp(-((zi / h - 0.5) ** 2) / 0.03)))
                xb, yb, zb, _ = _cyl_shell(radius=r_bore, length=h, z0=0.0)
                _add_surface(xb, yb, zb, ratio * (0.65 - 0.20 * (zb / h)))
                xt, yt, zt, rt, _ = _ring_disc(r_bore, r_outer, z0=h)
                race = np.exp(-((rt - r_inner) ** 2) / 1.2)
                _add_surface(xt, yt, zt, ratio * (0.20 + 0.65 * race))
                xbb, ybb, zbb, rb, _ = _ring_disc(r_bore, r_outer, z0=0.0)
                race_b = np.exp(-((rb - r_inner) ** 2) / 1.2)
                _add_surface(xbb, ybb, zbb, ratio * (0.18 + 0.55 * race_b))

            elif obj == "pulley":
                # Hub + rim + central groove.
                r_bore, r_hub, r_rim, h = 4.5, 7.0, 13.5, 8.0
                xh, yh, zh, _ = _cyl_shell(radius=r_hub, length=h, z0=0.0)
                _add_surface(xh, yh, zh, ratio * (0.28 + 0.40 * (zh / h)), showscale=True)
                xr, yr, zr, theta = _cyl_shell(radius=r_rim, length=h, z0=0.0, amp=lambda t, zz: -0.9 * np.exp(-((zz / h - 0.5) ** 2) / 0.03))
                rim_hot = 0.35 + 0.45 * np.exp(-((zr / h - 0.5) ** 2) / 0.03) + 0.15 * np.cos(theta) ** 2
                _add_surface(xr, yr, zr, ratio * rim_hot)
                xb, yb, zb, _ = _cyl_shell(radius=r_bore, length=h, z0=0.0)
                _add_surface(xb, yb, zb, ratio * (0.62 - 0.15 * (zb / h)))
                xt, yt, zt, rt, _ = _ring_disc(r_bore, r_rim, z0=h)
                _add_surface(xt, yt, zt, ratio * (0.2 + 0.6 * ((r_rim - rt) / (r_rim - r_bore))))
                xb2, yb2, zb2, rb, _ = _ring_disc(r_bore, r_rim, z0=0.0)
                _add_surface(xb2, yb2, zb2, ratio * (0.2 + 0.45 * (rb / r_rim)))

            elif obj in ["gear", "sprocket"]:
                if obj == "gear":
                    # Reference-style spur gear: solid center, blocky teeth, flat top face.
                    teeth = 20
                    r_core, r_root, r_tip, h = 7.2, 10.0, 12.6, 6.8

                    def _gear_tooth_radius(t):
                        c = np.cos(teeth * t)
                        tooth_sector = (c > 0.45).astype(float)
                        chamfer = np.clip((c - 0.35) / 0.10, 0.0, 1.0) * np.clip((0.55 - c) / 0.10, 0.0, 1.0)
                        return r_root + (r_tip - r_root) * np.clip(tooth_sector + 0.25 * chamfer, 0.0, 1.0)

                    tt = np.linspace(0, 2 * np.pi, 320)
                    zz = np.linspace(0.0, h, 42)
                    tt, zz = np.meshgrid(tt, zz)
                    rbnd = _gear_tooth_radius(tt)
                    xg = rbnd * np.cos(tt)
                    yg = rbnd * np.sin(tt)
                    zg = zz
                    tooth_hot = (rbnd - r_root) / max(1e-6, (r_tip - r_root))
                    s_outer = ratio * (0.20 + 0.62 * tooth_hot + 0.12 * np.exp(-((zz / h - 0.55) ** 2) / 0.08))
                    _add_surface(xg, yg, zg, s_outer, showscale=True)

                    xc, yc, zc, _ = _cyl_shell(radius=r_core, length=h, z0=0.0)
                    s_core = ratio * (0.18 + 0.20 * (zc / h))
                    _add_surface(xc, yc, zc, s_core)

                    uu = np.linspace(0.0, 1.0, 92)
                    t2 = np.linspace(0, 2 * np.pi, 320)
                    uu, t2 = np.meshgrid(uu, t2)
                    r_tooth = _gear_tooth_radius(t2)
                    rr = r_core + (r_tooth - r_core) * uu
                    xt = rr * np.cos(t2)
                    yt = rr * np.sin(t2)
                    rim_ratio = (rr - r_core) / np.maximum(r_tooth - r_core, 1e-6)

                    zt = np.full_like(xt, h)
                    s_top = ratio * (0.22 + 0.55 * (1.0 - rim_ratio) + 0.15 * np.clip(np.cos(teeth * t2), 0.0, 1.0))
                    _add_surface(xt, yt, zt, s_top)

                    zb = np.full_like(xt, 0.0)
                    s_bot = ratio * (0.18 + 0.45 * rim_ratio)
                    _add_surface(xt, yt, zb, s_bot)
                else:
                    # Keep sprocket distinct with fewer/larger teeth.
                    teeth = 12
                    r_bore, r_root, r_tip, h = 4.2, 9.8, 12.8, 6.5
                    xg, yg, zg, theta = _cyl_shell(
                        radius=r_root,
                        length=h,
                        z0=0.0,
                        amp=lambda t, zz: (r_tip - r_root) * np.clip(np.cos(teeth * t), 0.0, 1.0) ** 1.6
                    )
                    tooth_peak = np.clip(np.cos(teeth * theta), 0.0, 1.0) ** 1.6
                    _add_surface(xg, yg, zg, ratio * (0.24 + 0.62 * tooth_peak + 0.12 * (zg / h)), showscale=True)
                    xbore, ybore, zbore, _ = _cyl_shell(radius=r_bore, length=h, z0=0.0)
                    _add_surface(xbore, ybore, zbore, ratio * (0.68 - 0.20 * (zbore / h)))
                    xcap, ycap, zcap, rcap, _ = _ring_disc(r_bore, r_tip, z0=h)
                    _add_surface(xcap, ycap, zcap, ratio * (0.2 + 0.7 * ((r_tip - rcap) / (r_tip - r_bore))))
                    xcap2, ycap2, zcap2, rcap2, _ = _ring_disc(r_bore, r_tip, z0=0.0)
                    _add_surface(xcap2, ycap2, zcap2, ratio * (0.18 + 0.5 * (rcap2 / r_tip)))
        elif obj == "spring":
            t = np.linspace(0, 8 * np.pi, 220)
            s = np.linspace(0, 2 * np.pi, 30)
            t, s = np.meshgrid(t, s)
            R, r = 7.0, 1.0
            x = (R + r * np.cos(s)) * np.cos(t)
            y = (R + r * np.cos(s)) * np.sin(t)
            z = 1.7 * t + r * np.sin(s)
            ss = ratio * (0.25 + 0.75 * np.abs(np.sin(t)))
            _add_surface(x, y, z, ss, showscale=True)
        elif obj == "hinge":
            xx = np.linspace(-13, 13, 52)
            yy = np.linspace(-8, 8, 34)
            xx, yy = np.meshgrid(xx, yy)
            z_left = np.full_like(xx, 0.0)
            z_right = np.full_like(xx, 4.0)
            s_left = ratio * (0.15 + 0.85 * np.exp(-((xx + 1.5) ** 2) / 4.0))
            s_right = ratio * (0.15 + 0.85 * np.exp(-((xx - 1.5) ** 2) / 4.0))
            mask_l = xx <= -1.0
            mask_r = xx >= 1.0
            _add_surface(np.where(mask_l, xx, np.nan), np.where(mask_l, yy, np.nan), np.where(mask_l, z_left, np.nan), np.where(mask_l, s_left, np.nan), showscale=True)
            _add_surface(np.where(mask_r, xx, np.nan), np.where(mask_r, yy, np.nan), np.where(mask_r, z_right, np.nan), np.where(mask_r, s_right, np.nan))
            _add_cylinder(radius=2.2, length=18.0, hotspot=0.5, z0=0.0)  # hinge pin
        elif obj in ["cube", "box", "cuboid", "bracket", "cone", "sphere"]:
            if obj == "sphere":
                th = np.linspace(0, 2 * np.pi, 88)
                ph = np.linspace(0.05, np.pi - 0.05, 50)
                th, ph = np.meshgrid(th, ph)
                r = 9.0
                x = r * np.sin(ph) * np.cos(th)
                y = r * np.sin(ph) * np.sin(th)
                z = r * np.cos(ph)
                s = ratio * (0.2 + 0.8 * (np.sin(ph) ** 2))
                _add_surface(x, y, z, s, showscale=True)
            elif obj == "cone":
                z = np.linspace(0, 16, 58)
                th = np.linspace(0, 2 * np.pi, 92)
                z, th = np.meshgrid(z, th)
                r = 9.0 * (1 - z / 16.0)
                x = r * np.cos(th)
                y = r * np.sin(th)
                s = ratio * (0.25 + 0.75 * (z / 16.0))
                _add_surface(x, y, z, s, showscale=True)
                xb, yb, zb, rb = _disc(radius=9.0, z0=0.0)
                _add_surface(xb, yb, zb, ratio * (0.2 + 0.5 * rb / 9.0))
            elif obj == "bracket":
                # L-bracket: base plate + vertical plate
                x = np.linspace(-12, 12, 36)
                y = np.linspace(-10, 10, 30)
                x, y = np.meshgrid(x, y)
                zb = np.full_like(x, 0.0)
                sb = ratio * (0.20 + 0.35 * (np.abs(x) / 12.0) + 0.60 * np.exp(-((x + 9) ** 2 + (y - 2) ** 2) / 18.0))
                _add_surface(x, y, zb, sb, showscale=True)

                y2 = np.linspace(-10, 10, 30)
                z2 = np.linspace(0, 14, 32)
                y2, z2 = np.meshgrid(y2, z2)
                x2 = np.full_like(y2, -10.0)
                s2 = ratio * (0.25 + 0.55 * (z2 / 14.0) + 0.35 * np.exp(-((z2 - 2.5) ** 2 + (y2 - 2) ** 2) / 10.0))
                _add_surface(x2, y2, z2, s2)
            else:
                # Box/cuboid body with side faces
                lx, ly, lz = (16.0, 16.0, 16.0) if obj == "cube" else (22.0, 14.0, 10.0)
                x = np.linspace(-lx / 2, lx / 2, 34)
                y = np.linspace(-ly / 2, ly / 2, 28)
                x, y = np.meshgrid(x, y)
                zt = np.full_like(x, lz / 2)
                zb = np.full_like(x, -lz / 2)
                edge = (np.abs(x) / (lx / 2)) ** 3 + (np.abs(y) / (ly / 2)) ** 3
                st = ratio * (0.18 + 0.82 * np.clip(edge / 2.0, 0.0, 1.0))
                sb = ratio * (0.15 + 0.45 * np.clip(edge / 2.0, 0.0, 1.0))
                _add_surface(x, y, zt, st, showscale=True)
                _add_surface(x, y, zb, sb)

                ys = np.linspace(-ly / 2, ly / 2, 28)
                zs = np.linspace(-lz / 2, lz / 2, 28)
                ys, zs = np.meshgrid(ys, zs)
                xr = np.full_like(ys, lx / 2)
                xl = np.full_like(ys, -lx / 2)
                sr = ratio * (0.22 + 0.55 * np.abs(zs) / (lz / 2))
                sl = ratio * (0.22 + 0.55 * np.abs(zs) / (lz / 2))
                _add_surface(xr, ys, zs, sr)
                _add_surface(xl, ys, zs, sl)

                xs = np.linspace(-lx / 2, lx / 2, 34)
                zs2 = np.linspace(-lz / 2, lz / 2, 28)
                xs, zs2 = np.meshgrid(xs, zs2)
                yf = np.full_like(xs, ly / 2)
                yb = np.full_like(xs, -ly / 2)
                sf = ratio * (0.24 + 0.50 * np.abs(zs2) / (lz / 2))
                sbk = ratio * (0.24 + 0.50 * np.abs(zs2) / (lz / 2))
                _add_surface(xs, yf, zs2, sf)
                _add_surface(xs, yb, zs2, sbk)
        else:
            _add_cylinder(radius=7.0, length=20.0, hotspot=0.6)
    except Exception:
        # Never break simulation UI because of plotting; fallback to safe 3D cylinder heatmap.
        fig = go.Figure()
        x, y, z, theta = _cyl_shell(radius=7.0, length=22.0, z0=0.0)
        s = ratio * (0.2 + 0.6 * (z / 22.0) + 0.2 * np.cos(theta) ** 2)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=s,
            colorscale=colors, cmin=0.0, cmax=cmax, showscale=True,
            colorbar=dict(title='Stress/Yield', thickness=15),
        ))

    fig.update_layout(
        template="plotly_dark", title="3D Stress Heatmap",
        scene=dict(aspectmode='data', xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
        height=400, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def create_compliance_gauge(result: SimResult) -> go.Figure:
    min_sf = SAFETY_STANDARDS.get(result.standard, {}).get('min_sf', 2.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result.safety_factor,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Safety Factor", 'font': {'size': 20, 'color': '#e0e8f0'}},
        delta={'reference': min_sf, 'increasing': {'color': "#22d3a0"}},
        gauge={
            'axis': {'range': [None, max(5, result.safety_factor * 1.2)]},
            'bar': {'color': result.compliance_color},
            'steps': [
                {'range': [0, 1], 'color': 'rgba(239,68,68,0.3)'},
                {'range': [1, min_sf], 'color': 'rgba(245,158,11,0.3)'},
                {'range': [min_sf, 10], 'color': 'rgba(34,211,160,0.3)'}
            ],
            'threshold': {'line': {'color': "white", 'width': 3}, 'value': min_sf}
        }
    ))
    fig.update_layout(template="plotly_dark", height=280,
                      paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL SIMULATION STATE
# ═══════════════════════════════════════════════════════════════════════════

class SimAppState:
    def __init__(self):
        self.last_detected_object = "nut"
        self.last_dims = [10, 20, 8]
        self.sim_history = []
        self.failure_point = None

sim_state = SimAppState()
sim_engine = SimulationEngine()
ai_explainer = AIExplainer()

def run_simulation(component_type, temperature, load):
    from ml import _predict_from_ml_models
    obj_type = str(component_type).strip().lower()
    ml = _predict_from_ml_models(load, temperature, obj_type)

    result = SimResult(
        object_type=obj_type,
        material_key=ml["material_key"],
        material_label=ml["material_label"],
        stress_mpa=ml["stress_mpa"],
        max_stress_mpa=ml["stress_mpa"],
        deformation_mm=ml["strain"],
        safety_factor=ml["safety_factor"],
        standard=MATERIALS[ml["material_key"]]["standard"],
        yield_mpa=ml["yield_mpa"],
        stress_ratio=ml["stress_ratio"],
        mass_g=ml["yield_mpa"],
    )
    if result.safety_factor >= 2.0:
        result.compliance_color = "#22d3a0"
    elif result.safety_factor >= 1.5:
        result.compliance_color = "#f59e0b"
    else:
        result.compliance_color = "#ef4444"

    sim_state.sim_history.append({
        "load": load, "sf": result.safety_factor,
        "stress": result.stress_mpa, "deform": result.deformation_mm
    })

    sf_fig      = create_sf_plot(sim_state.sim_history, result.standard)
    deform_fig  = create_deform_plot(sim_state.sim_history)
    heatmap_fig = create_heatmap(result, obj_type)
    gauge_fig   = create_compliance_gauge(result)

    params = {"load": load, "temperature": temperature, "component_type": obj_type}
    report = ai_explainer.generate_report(params, result, None, None)
    show_failure = result.safety_factor < 1.2

    return (
        result.safety_factor,
        result.stress_mpa,
        result.deformation_mm,
        result.mass_g,
        sf_fig,
        deform_fig,
        heatmap_fig,
        gauge_fig,
        report,
        gr.Group(visible=show_failure)
    )

def find_failure(component_type, temperature, load):
    from ml import _predict_from_ml_models
    obj_type = str(component_type).strip().lower()
    baseline_temp = float(temperature)
    applied_load  = float(load)
    ml_at_applied = _predict_from_ml_models(applied_load, baseline_temp, obj_type)
    failure_load  = applied_load * float(ml_at_applied["safety_factor"])
    sim_state.failure_point = failure_load

    ml_fail = _predict_from_ml_models(failure_load, baseline_temp, obj_type)
    result = SimResult(
        object_type=obj_type,
        material_key=ml_fail["material_key"],
        material_label=ml_fail["material_label"],
        stress_mpa=ml_fail["stress_mpa"],
        max_stress_mpa=ml_fail["stress_mpa"],
        deformation_mm=ml_fail["strain"],
        safety_factor=ml_fail["safety_factor"],
        standard=MATERIALS[ml_fail["material_key"]]["standard"],
        yield_mpa=ml_fail["yield_mpa"],
        stress_ratio=ml_fail["stress_ratio"],
    )
    failure_mode = "Yield Failure (Plastic Deformation)" if result.stress_mpa > result.yield_mpa else "Buckling/Instability"

    params = {"load": failure_load}
    report = ai_explainer.generate_report(params, result, failure_load, failure_mode)

    sf_fig = create_sf_plot(
        sim_state.sim_history + [{"load": failure_load, "sf": result.safety_factor}],
        result.standard
    )

    return (
        applied_load,
        result.safety_factor,
        result.stress_mpa,
        f"## 💥 FAILURE FOUND AT {failure_load:.0f}N\n\n{report}",
        gr.Group(visible=True),
        sf_fig
    )

