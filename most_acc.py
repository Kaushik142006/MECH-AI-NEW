
from build123d import (
    Axis, BuildPart, BuildSketch, BuildLine,
    Box, Cylinder, Plane, Sphere, RegularPolygon, Circle, Rectangle,
    extrude, revolve, loft, sweep, Shell,
    fillet, chamfer, Hole,
    Helix, PolarLocations, GridLocations, add, export_stl,
    Location, Vector, Edge, Wire, Face,
    Mode, GeomType, Select,
    LineType,
)
import gradio as gr
import os
import re
import shutil
import traceback
import tempfile
import subprocess
import sys
import uuid
import hashlib
from openai import OpenAI

# â”€â”€ Ollama client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
STL_PATH = os.path.join(MODEL_DIR, "model.stl").replace("\\", "/")

# Performance tuning (keeps geometry logic intact)
VERBOSE_LOGS = False
MAX_HISTORY_MESSAGES = 8
PROMPT_MAX_TOKENS = 500
CODER_MAX_TOKENS = 650
COLLECTOR_MAX_TOKENS = 220
MAX_AUTOFIX_RETRIES = 2
CODE_MIN_LENGTH = 120

_PROMPT_CACHE: dict[str, str] = {}
_CODER_CACHE: dict[str, str] = {}
_PIPELINE_CACHE: dict[str, str] = {}


def prepare_viewer_model(stl_path: str) -> str | None:
    if not stl_path or not os.path.exists(stl_path):
        return None

    viewer_path = os.path.join(MODEL_DIR, f"viewer_{uuid.uuid4().hex}.stl")
    shutil.copy2(stl_path, viewer_path)
    return viewer_path.replace("\\", "/")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  VERIFIED build123d CODE TEMPLATES
#  Export syntax: export_stl(b.part, 'path')  â€” standalone function
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def make_hex_nut(inner_r, outer_r, thickness, stl_path):
    return (
        "from build123d import *\n"
        f"inner_r = {inner_r}\n"
        f"outer_r = {outer_r}\n"
        f"thickness = {thickness}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        RegularPolygon(radius=outer_r, side_count=6)\n"
        "    extrude(amount=thickness)\n"
        "    Hole(radius=inner_r)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_circle_nut(inner_r, outer_r, thickness, stl_path):
    return (
        "from build123d import *\n"
        f"inner_r = {inner_r}\n"
        f"outer_r = {outer_r}\n"
        f"thickness = {thickness}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Circle(radius=outer_r)\n"
        "    extrude(amount=thickness)\n"
        "    Hole(radius=inner_r)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_hex_bolt(shaft_r, head_r, shaft_len, head_h, stl_path):
    """Hex bolt = hex head + cylindrical shaft below it."""
    return (
        "from build123d import *\n"
        f"shaft_r  = {shaft_r}\n"
        f"head_r   = {head_r}\n"
        f"shaft_len = {shaft_len}\n"
        f"head_h   = {head_h}\n"
        "with BuildPart() as b:\n"
        "    # Shaft\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Circle(radius=shaft_r)\n"
        "    extrude(amount=shaft_len)\n"
        "    # Hex head on top of shaft\n"
        "    with BuildSketch(Plane.XY.offset(shaft_len)):\n"
        "        RegularPolygon(radius=head_r, side_count=6)\n"
        "    extrude(amount=head_h)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_screw(shaft_r, head_r, shaft_len, head_h, stl_path):
    """Screw = round/flat head + cylindrical shaft."""
    return (
        "from build123d import *\n"
        f"shaft_r   = {shaft_r}\n"
        f"head_r    = {head_r}\n"
        f"shaft_len = {shaft_len}\n"
        f"head_h    = {head_h}\n"
        "with BuildPart() as b:\n"
        "    # Shaft\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Circle(radius=shaft_r)\n"
        "    extrude(amount=shaft_len)\n"
        "    # Round head on top\n"
        "    with BuildSketch(Plane.XY.offset(shaft_len)):\n"
        "        Circle(radius=head_r)\n"
        "    extrude(amount=head_h)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_cylinder(radius, height, stl_path):
    return (
        "from build123d import *\n"
        f"with BuildPart() as b:\n"
        f"    Cylinder(radius={radius}, height={height})\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_cylinder_with_hole(radius, height, hole_r, stl_path):
    return (
        "from build123d import *\n"
        f"radius = {radius}; height = {height}; hole_r = {hole_r}\n"
        "with BuildPart() as b:\n"
        "    Cylinder(radius=radius, height=height)\n"
        "    Hole(radius=hole_r, depth=height)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_box(length, width, height, stl_path):
    return (
        "from build123d import *\n"
        f"with BuildPart() as b:\n"
        f"    Box({length}, {width}, {height})\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_washer(inner_r, outer_r, thickness, stl_path):
    return make_circle_nut(inner_r, outer_r, thickness, stl_path)


def make_plate_with_holes(length, width, thickness, hole_r, hole_count, stl_path):
    """Rectangular plate with evenly spaced holes in a grid."""
    cols = int(max(round(max(hole_count, 1) ** 0.5), 2))
    rows = int(max((max(hole_count, 1) + cols - 1) // cols, 2))
    return (
        "from build123d import *\n"
        f"length = {length}; width = {width}; thickness = {thickness}\n"
        f"hole_r = {hole_r}; hole_count = {hole_count}; cols = {cols}; rows = {rows}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Rectangle(length, width)\n"
        "    extrude(amount=thickness)\n"
        "    x_spacing = max(length * 0.7 / max(cols - 1, 1), hole_r * 3.0)\n"
        "    y_spacing = max(width * 0.7 / max(rows - 1, 1), hole_r * 3.0)\n"
        "    with GridLocations(x_spacing, y_spacing, cols, rows):\n"
        "        Hole(radius=hole_r, depth=thickness)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_plate_center_hole(length, width, thickness, hole_r, stl_path):
    return (
        "from build123d import *\n"
        f"length = {length}; width = {width}; thickness = {thickness}; hole_r = {hole_r}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Rectangle(length, width)\n"
        "    extrude(amount=thickness)\n"
        "    Hole(radius=hole_r, depth=thickness)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_table(top_l, top_w, top_t, leg_w, leg_h, stl_path):
    return (
        "from build123d import *\n"
        f"top_l = {top_l}; top_w = {top_w}; top_t = {top_t}\n"
        f"leg_w = {leg_w}; leg_h = {leg_h}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY.offset(leg_h)):\n"
        "        Rectangle(top_l, top_w)\n"
        "    extrude(amount=top_t)\n"
        "    x_off = max((top_l - leg_w) / 2.0, leg_w * 0.6)\n"
        "    y_off = max((top_w - leg_w) / 2.0, leg_w * 0.6)\n"
        "    with BuildSketch(Plane.XY):\n"
        "        with Locations((-x_off, -y_off)):\n"
        "            Rectangle(leg_w, leg_w)\n"
        "    extrude(amount=leg_h)\n"
        "    with BuildSketch(Plane.XY):\n"
        "        with Locations((-x_off, y_off)):\n"
        "            Rectangle(leg_w, leg_w)\n"
        "    extrude(amount=leg_h)\n"
        "    with BuildSketch(Plane.XY):\n"
        "        with Locations((x_off, -y_off)):\n"
        "            Rectangle(leg_w, leg_w)\n"
        "    extrude(amount=leg_h)\n"
        "    with BuildSketch(Plane.XY):\n"
        "        with Locations((x_off, y_off)):\n"
        "            Rectangle(leg_w, leg_w)\n"
        "    extrude(amount=leg_h)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_bushing(inner_r, outer_r, height, stl_path):
    """Simple cylindrical bushing / sleeve."""
    return (
        "from build123d import *\n"
        f"inner_r = {inner_r}; outer_r = {outer_r}; height = {height}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Circle(radius=outer_r)\n"
        "    extrude(amount=height)\n"
        "    Hole(radius=inner_r, depth=height)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_stepped_shaft(section_diams, section_lengths, stl_path):
    """Solid stepped shaft with sections stacked along one axis."""
    return (
        "from build123d import *\n"
        f"section_diams = {list(section_diams)}\n"
        f"section_lengths = {list(section_lengths)}\n"
        "with BuildPart() as b:\n"
        "    z_offset = 0.0\n"
        "    for dia, seg_len in zip(section_diams, section_lengths):\n"
        "        with BuildSketch(Plane.XY.offset(z_offset)):\n"
        "            Circle(radius=dia / 2.0)\n"
        "        extrude(amount=seg_len)\n"
        "        z_offset += seg_len\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


def make_bracket(base_l, base_w, wall_h, thickness, stl_path):
    """L-shaped bracket: horizontal base + vertical wall."""
    return (
        "from build123d import *\n"
        f"base_l = {base_l}; base_w = {base_w}\n"
        f"wall_h = {wall_h}; thickness = {thickness}\n"
        "with BuildPart() as b:\n"
        "    # Horizontal base plate\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Rectangle(base_l, base_w)\n"
        "    extrude(amount=thickness)\n"
        "    # Vertical wall\n"
        "    with BuildSketch(Plane.XZ):\n"
        "        Rectangle(base_l, wall_h)\n"
        "    extrude(amount=thickness)\n"
        f"export_stl(b.part, '{stl_path}')\n"
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  OBJECT DETECTOR  â€” reads the summary and picks the right template
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _has_corner_feature_request(summary: str) -> bool:
    s = summary.lower()
    has_corner = "corner" in s
    has_feature = any(
        token in s for token in ("rectangle", "rectangles", "vertical", "post", "pillar", "boss", "feature")
    )
    explicit_four = bool(
        re.search(r"\b(?:4|four)\s*(?:x\s*)?corners?\b", s)
        or re.search(r"\b(?:at|in)\s+each\s+corner\b", s)
    )
    if "hole" in s and not has_feature:
        return False
    return has_corner and (has_feature or explicit_four)


def _requires_four_corners(summary: str) -> bool:
    s = summary.lower()
    return bool(
        re.search(r"\b(?:4|four)\s*(?:x\s*)?corners?\b", s)
        or re.search(r"\b(?:at|in)\s+each\s+corner\b", s)
        or "all corners" in s
    )


def detect_object(summary: str) -> str:
    s = summary.lower()
    if _has_corner_feature_request(s):
        return "plate"
    if any(w in s for w in ["screw", "machine screw", "wood screw", "self-tapping"]):
        return "screw"
    if any(w in s for w in ["bolt", "hex bolt", "carriage bolt", "stud bolt"]):
        return "bolt"
    if any(w in s for w in ["nut", "hex nut", "lock nut", "flange nut"]):
        return "nut"
    if "washer" in s:
        return "washer"
    if any(w in s for w in ["bracket", "angle bracket", "l-bracket", "l bracket"]):
        return "bracket"
    if any(w in s for w in ["plate", "flat plate", "base plate", "mounting plate"]):
        return "plate"
    if any(w in s for w in ["bushing", "sleeve", "bearing sleeve"]):
        return "bushing"
    if any(w in s for w in ["shaft", "axle", "spindle"]):
        return "shaft"
    if any(w in s for w in ["cylinder", "rod", "pipe", "tube"]):
        return "cylinder"
    if any(w in s for w in ["gear", "tooth", "teeth"]):
        return "gear"
    if any(w in s for w in ["helmet", "visor", "head shell"]):
        return "helmet"
    if any(w in s for w in ["frame", "chassis", "skeleton"]):
        return "frame"
    if any(w in s for w in ["table", "desk"]):
        return "table"
    if any(w in s for w in ["box", "cube", "rect", "block", "enclosure"]):
        return "box"
    return "unknown"


def parse_dims(summary: str):
    """Extract all positive numbers from summary string."""
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", summary)
    return [float(m) for m in matches if float(m) > 0]


def classify_object(summary: str) -> str:
    s = summary.lower()

    if _has_corner_feature_request(s):
        return "corner_feature_part"
    if any(w in s for w in ("screw", "bolt", "nut", "washer", "bushing", "fastener", "thread", "threaded")):
        return "fastener"
    if any(w in s for w in ("gear", "tooth", "teeth")):
        return "gear"
    if any(w in s for w in ("table", "frame", "assembly", "chassis", "multi-part", "multipart")):
        return "assembly"
    if any(w in s for w in ("hole", "bore", "slot", "center hole", "bolt circle", "hole pattern", "multi-hole")):
        return "hole_part"
    if any(w in s for w in ("helmet", "organic", "shell")):
        return "organic"
    return "generic"


def _extract_hole_count(summary: str, default: int = 4) -> int:
    match = re.search(r"(\d+)\s*(?:x\s*)?(?:holes|hole)\b", summary, flags=re.IGNORECASE)
    if match:
        return max(int(match.group(1)), 1)
    return max(default, 1)


def generate_gear(summary: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")
    dims = parse_dims(summary)
    outer_dia = dims[0] if len(dims) >= 1 else 60.0
    thickness = dims[1] if len(dims) >= 2 else 10.0
    tooth_count = int(round(dims[2])) if len(dims) >= 3 else 20
    tooth_depth = dims[3] if len(dims) >= 4 else max(outer_dia * 0.08, 2.0)
    tooth_count = max(tooth_count, 8)
    root_radius = max((outer_dia / 2.0) - tooth_depth, outer_dia * 0.2)
    pitch_circ = 2.0 * 3.14159265 * (outer_dia / 2.0)
    tooth_width = max((pitch_circ / tooth_count) * 0.45, tooth_depth * 0.8)
    return (
        "from build123d import *\n"
        f"root_radius = {root_radius}\n"
        f"thickness = {thickness}\n"
        f"tooth_count = {tooth_count}\n"
        f"tooth_depth = {tooth_depth}\n"
        f"tooth_width = {tooth_width}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Circle(radius=root_radius)\n"
        "    extrude(amount=thickness)\n"
        "    with BuildSketch(Plane.XY):\n"
        "        with PolarLocations(root_radius + tooth_depth * 0.5, tooth_count):\n"
        "            Rectangle(tooth_depth, tooth_width)\n"
        "    extrude(amount=thickness, mode=Mode.ADD)\n"
        f"export_stl(b.part, '{safe_path}')\n"
    )


def generate_table(summary: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")
    dims = parse_dims(summary)
    top_l = dims[0] if len(dims) >= 1 else 120.0
    top_w = dims[1] if len(dims) >= 2 else 70.0
    top_t = dims[2] if len(dims) >= 3 else 6.0
    leg_h = dims[3] if len(dims) >= 4 else 72.0
    leg_w = dims[4] if len(dims) >= 5 else max(min(top_l, top_w) * 0.08, 6.0)
    return make_table(top_l, top_w, top_t, leg_w, leg_h, safe_path)


def generate_hole_part(summary: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")
    dims = parse_dims(summary)
    lowered = summary.lower()
    obj = detect_object(summary)

    if obj == "cylinder":
        radius = (dims[0] / 2.0) if len(dims) >= 1 else 10.0
        height = dims[1] if len(dims) >= 2 else 20.0
        hole_r = (dims[2] / 2.0) if len(dims) >= 3 else max(radius * 0.35, 1.0)
        return make_cylinder_with_hole(radius, height, hole_r, safe_path)

    length = dims[0] if len(dims) >= 1 else 80.0
    width = dims[1] if len(dims) >= 2 else 60.0
    thickness = dims[2] if len(dims) >= 3 else 8.0
    hole_r = (dims[3] / 2.0) if len(dims) >= 4 else max(min(length, width) * 0.06, 2.0)
    hole_count = _extract_hole_count(summary, 4)
    wants_pattern = any(
        k in lowered for k in ("holes", "pattern", "grid", "bolt circle", "multi-hole", "multiple")
    ) or hole_count > 1

    if wants_pattern:
        cols = int(max(round(hole_count ** 0.5), 2))
        rows = int(max((hole_count + cols - 1) // cols, 2))
        return (
            "from build123d import *\n"
            f"length = {length}; width = {width}; thickness = {thickness}\n"
            f"hole_r = {hole_r}; hole_count = {hole_count}; cols = {cols}; rows = {rows}\n"
            "with BuildPart() as b:\n"
            "    with BuildSketch(Plane.XY):\n"
            "        Rectangle(length, width)\n"
            "    extrude(amount=thickness)\n"
            "    x_spacing = max(length * 0.7 / max(cols - 1, 1), hole_r * 3.0)\n"
            "    y_spacing = max(width * 0.7 / max(rows - 1, 1), hole_r * 3.0)\n"
            "    with GridLocations(x_spacing, y_spacing, cols, rows):\n"
            "        Hole(radius=hole_r, depth=thickness)\n"
            f"export_stl(b.part, '{safe_path}')\n"
        )

    return make_plate_center_hole(length, width, thickness, hole_r, safe_path)


def generate_corner_feature_part(summary: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")
    dims = parse_dims(summary)

    if dims and _requires_four_corners(summary) and dims[0] == 4.0:
        dims = dims[1:]

    length = _extract_labeled_value(summary, [r"Base\s+Length", r"Plate\s+Length", r"Length"])
    width = _extract_labeled_value(summary, [r"Base\s+Width", r"Plate\s+Width", r"Width"])
    base_t = _extract_labeled_value(summary, [r"Base\s+Thickness", r"Plate\s+Thickness", r"Thickness"])
    corner_l = _extract_labeled_value(summary, [r"Corner\s+Length", r"Corner\s+Size", r"Post\s+Length"])
    corner_w = _extract_labeled_value(summary, [r"Corner\s+Width", r"Post\s+Width"])
    corner_h = _extract_labeled_value(
        summary,
        [r"Corner\s+Height", r"Vertical\s+Height", r"Post\s+Height", r"Feature\s+Height"],
    )

    length = length if length is not None else (dims[0] if len(dims) >= 1 else 100.0)
    width = width if width is not None else (dims[1] if len(dims) >= 2 else 80.0)
    base_t = base_t if base_t is not None else (dims[2] if len(dims) >= 3 else 6.0)
    corner_l = corner_l if corner_l is not None else (dims[3] if len(dims) >= 4 else max(min(length, width) * 0.15, 5.0))
    corner_w = corner_w if corner_w is not None else (dims[4] if len(dims) >= 5 else corner_l)
    corner_h = corner_h if corner_h is not None else (dims[5] if len(dims) >= 6 else max(base_t * 2.5, 10.0))

    corner_l = min(max(corner_l, 1.0), max(length * 0.45, 1.0))
    corner_w = min(max(corner_w, 1.0), max(width * 0.45, 1.0))
    corner_h = max(corner_h, 1.0)

    return (
        "from build123d import *\n"
        f"length = {length}\n"
        f"width = {width}\n"
        f"base_thickness = {base_t}\n"
        f"corner_length = {corner_l}\n"
        f"corner_width = {corner_w}\n"
        f"corner_height = {corner_h}\n"
        "x_off = max(length / 2.0 - corner_length / 2.0, corner_length / 2.0)\n"
        "y_off = max(width / 2.0 - corner_width / 2.0, corner_width / 2.0)\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Rectangle(length, width)\n"
        "    extrude(amount=base_thickness)\n"
        "    with BuildSketch(Plane.XY.offset(base_thickness)):\n"
        "        with Locations(\n"
        "            (-x_off, -y_off),\n"
        "            (-x_off, y_off),\n"
        "            (x_off, -y_off),\n"
        "            (x_off, y_off),\n"
        "        ):\n"
        "            Rectangle(corner_length, corner_width)\n"
        "    extrude(amount=corner_height)\n"
        f"export_stl(b.part, '{safe_path}')\n"
    )


def generate_helmet(summary: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")
    dims = parse_dims(summary)
    outer_r = (dims[0] / 2.0) if len(dims) >= 1 else 60.0
    wall_t = dims[1] if len(dims) >= 2 else max(outer_r * 0.08, 2.0)
    wall_t = min(max(wall_t, 1.0), outer_r * 0.4)
    return (
        "from build123d import *\n"
        f"outer_r = {outer_r}\n"
        f"wall_t = {wall_t}\n"
        "with BuildPart() as b:\n"
        "    Sphere(radius=outer_r)\n"
        "    with BuildSketch(Plane.XY.offset(-outer_r * 0.55)):\n"
        "        Rectangle(outer_r * 2.8, outer_r * 2.8)\n"
        "    extrude(amount=outer_r * 2.0, mode=Mode.SUBTRACT)\n"
        "    with BuildSketch(Plane.YZ.offset(outer_r * 0.35)):\n"
        "        Rectangle(outer_r * 1.3, outer_r * 0.8)\n"
        "    extrude(amount=outer_r * 2.2, mode=Mode.SUBTRACT)\n"
        "    Shell(amount=-wall_t, openings=b.faces().sort_by(Axis.Z)[:1])\n"
        f"export_stl(b.part, '{safe_path}')\n"
    )


def generate_frame(summary: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")
    dims = parse_dims(summary)
    length = dims[0] if len(dims) >= 1 else 120.0
    width = dims[1] if len(dims) >= 2 else 80.0
    height = dims[2] if len(dims) >= 3 else 90.0
    thickness = dims[3] if len(dims) >= 4 else 8.0
    return (
        "from build123d import *\n"
        f"length = {length}; width = {width}; height = {height}; t = {thickness}\n"
        "with BuildPart() as b:\n"
        "    x = max(length / 2.0 - t / 2.0, t)\n"
        "    y = max(width / 2.0 - t / 2.0, t)\n"
        "    for sx in (-1, 1):\n"
        "        for sy in (-1, 1):\n"
        "            with BuildSketch(Plane.XY):\n"
        "                with Locations((sx * x, sy * y)):\n"
        "                    Rectangle(t, t)\n"
        "            extrude(amount=height)\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Rectangle(length, t)\n"
        "    extrude(amount=t)\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Rectangle(t, width)\n"
        "    extrude(amount=t)\n"
        "    with BuildSketch(Plane.XY.offset(height - t)):\n"
        "        Rectangle(length, t)\n"
        "    extrude(amount=t)\n"
        "    with BuildSketch(Plane.XY.offset(height - t)):\n"
        "        Rectangle(t, width)\n"
        "    extrude(amount=t)\n"
        f"export_stl(b.part, '{safe_path}')\n"
    )


def generate_specialized_code(summary: str, stl_path: str) -> str | None:
    obj = detect_object(summary)
    cls = classify_object(summary)
    if cls == "corner_feature_part":
        return generate_corner_feature_part(summary, stl_path)
    if obj == "table":
        return generate_table(summary, stl_path)
    if obj == "gear" or cls == "gear":
        return generate_gear(summary, stl_path)
    if cls == "hole_part":
        return generate_hole_part(summary, stl_path)
    if obj == "helmet" or cls == "organic":
        return generate_helmet(summary, stl_path)
    if obj == "frame":
        return generate_frame(summary, stl_path)
    return None


def generate_fallback(summary: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")
    obj   = detect_object(summary)
    dims  = parse_dims(summary)
    lowered = summary.lower()
    is_hex = "hex" in lowered

    print(f"[Fallback] Object='{obj}'  Dims={dims}  Hex={is_hex}")

    specialized = generate_specialized_code(summary, safe_path)
    summary_class = classify_object(summary)
    if specialized is not None and (
        obj in {"table", "gear", "helmet", "frame"}
        or summary_class in {"hole_part", "corner_feature_part"}
    ):
        print("[Fallback] Specialized code:\n", specialized)
        return execute_code(specialized, stl_path, summary)

    # â”€â”€ Assign dimensions with sensible defaults per object type â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if obj == "screw":
        shaft_dia = _extract_labeled_value(summary, [r"Shaft\s+Dia(?:meter)?", r"Diameter", r"Dia(?:meter)?"])
        head_dia  = _extract_labeled_value(summary, [r"Head\s+Dia(?:meter)?"])
        head_h_in = _extract_labeled_value(summary, [r"Head\s+Height", r"Head\s+Thickness"])
        shaft_r   = ((shaft_dia or (dims[0] if len(dims) >= 1 else 6.0)) / 2)
        shaft_len = _extract_labeled_value(summary, [r"Length", r"Shaft\s+Length"]) or (dims[1] if len(dims) >= 2 else 20.0)
        head_r    = ((head_dia or (dims[2] if len(dims) >= 3 else shaft_r * 3.6)) / 2)
        head_h    = head_h_in or (dims[3] if len(dims) >= 4 else max(shaft_r * 0.7, 3.0))
        code = make_screw(shaft_r, head_r, shaft_len, head_h, safe_path)

    elif obj == "bolt":
        shaft_dia = _extract_labeled_value(summary, [r"Shaft\s+Dia(?:meter)?", r"Diameter", r"Dia(?:meter)?"])
        head_dia  = _extract_labeled_value(summary, [r"Head\s+Dia(?:meter)?"])
        head_h_in = _extract_labeled_value(summary, [r"Head\s+Height", r"Head\s+Thickness"])
        shaft_r   = ((shaft_dia or (dims[0] if len(dims) >= 1 else 10.0)) / 2)
        shaft_len = _extract_labeled_value(summary, [r"Length", r"Shaft\s+Length"]) or (dims[1] if len(dims) >= 2 else 30.0)
        head_r    = ((head_dia or (dims[2] if len(dims) >= 3 else shaft_r * 3.6)) / 2)
        head_h    = head_h_in or (dims[3] if len(dims) >= 4 else max(shaft_r * 0.7, 4.0))
        code = make_hex_bolt(shaft_r, head_r, shaft_len, head_h, safe_path)

    elif obj in ("nut", "washer"):
        inner_r   = (dims[0] / 2) if len(dims) >= 1 else 5.0
        outer_r   = (dims[1] / 2) if len(dims) >= 2 else inner_r * 2.0
        thickness = dims[2]        if len(dims) >= 3 else 5.0
        if obj == "washer":
            code = make_washer(inner_r, outer_r, thickness, safe_path)
        elif is_hex:
            code = make_hex_nut(inner_r, outer_r, thickness, safe_path)
        else:
            code = make_circle_nut(inner_r, outer_r, thickness, safe_path)

    elif obj == "bracket":
        base_l    = dims[0] if len(dims) >= 1 else 40.0
        base_w    = dims[1] if len(dims) >= 2 else 30.0
        wall_h    = dims[2] if len(dims) >= 3 else 30.0
        thickness = dims[3] if len(dims) >= 4 else 5.0
        code = make_bracket(base_l, base_w, wall_h, thickness, safe_path)

    elif obj == "plate":
        length    = dims[0] if len(dims) >= 1 else 60.0
        width     = dims[1] if len(dims) >= 2 else 40.0
        thickness = dims[2] if len(dims) >= 3 else 5.0
        hole_r    = dims[3] / 2 if len(dims) >= 4 else 4.0
        wants_hole = "hole" in lowered or "bore" in lowered
        wants_pattern = any(k in lowered for k in ["holes", "pattern", "grid", "bolt circle", "multi-hole", "multiple"])
        if wants_hole and not wants_pattern:
            code = make_plate_center_hole(length, width, thickness, hole_r, safe_path)
        elif wants_hole:
            code = generate_hole_part(summary, safe_path)
        else:
            code = make_box(length, width, thickness, safe_path)

    elif obj == "bushing":
        outer_r = (dims[0] / 2) if len(dims) >= 1 else 10.0
        inner_r = (dims[1] / 2) if len(dims) >= 2 else outer_r * 0.5
        height  = dims[2]        if len(dims) >= 3 else 20.0
        code = make_bushing(inner_r, outer_r, height, safe_path)

    elif obj == "shaft":
        if len(dims) >= 4:
            usable_count = len(dims) - (len(dims) % 2)
            section_diams = dims[:usable_count:2]
            section_lengths = dims[1:usable_count:2]
            code = make_stepped_shaft(section_diams, section_lengths, safe_path)
        else:
            radius = (dims[0] / 2) if len(dims) >= 1 else 10.0
            height = dims[1]        if len(dims) >= 2 else 20.0
            code = make_cylinder(radius, height, safe_path)

    elif obj == "cylinder":
        radius = (dims[0] / 2) if len(dims) >= 1 else 10.0
        height = dims[1]        if len(dims) >= 2 else 20.0
        if "hole" in lowered or "bore" in lowered:
            hole_r = (dims[2] / 2) if len(dims) >= 3 else max(radius * 0.35, 1.0)
            code = make_cylinder_with_hole(radius, height, hole_r, safe_path)
        else:
            code = make_cylinder(radius, height, safe_path)

    elif obj == "table":
        code = generate_table(summary, safe_path)

    elif obj == "gear":
        code = generate_gear(summary, safe_path)

    elif obj == "helmet":
        code = generate_helmet(summary, safe_path)

    elif obj == "frame":
        code = generate_frame(summary, safe_path)

    elif obj == "box":
        l = dims[0] if len(dims) >= 1 else 20.0
        w = dims[1] if len(dims) >= 2 else 20.0
        h = dims[2] if len(dims) >= 3 else 10.0
        code = make_box(l, w, h, safe_path)

    else:
        # Generic fallback: cylinder
        radius = (dims[0] / 2) if len(dims) >= 1 else 10.0
        height = dims[1]        if len(dims) >= 2 else 20.0
        code = make_cylinder(radius, height, safe_path)

    print("[Fallback] Code:\n", code)
    return execute_code(code, stl_path, summary)


def build_direct_summary(user_message: str) -> str | None:
    text = user_message.strip()
    if not text:
        return None

    if "all required parameters collected" in text.lower():
        return text

    obj = detect_object(text)
    dims = parse_dims(text)
    lowered = text.lower()

    if obj == "screw":
        if len(dims) < 3:
            return None
        shaft_dia = _extract_labeled_value(text, [r"Shaft\s+Dia(?:meter)?", r"Diameter", r"Dia(?:meter)?"]) or dims[0]
        length = _extract_labeled_value(text, [r"Length", r"Shaft\s+Length"]) or (dims[1] if len(dims) >= 2 else 20.0)
        head_dia = _extract_labeled_value(text, [r"Head\s+Dia(?:meter)?"]) or (dims[2] if len(dims) >= 3 else shaft_dia * 1.8)
        head_h = _extract_labeled_value(text, [r"Head\s+Height", r"Head\s+Thickness"])
        if head_h is None and len(dims) >= 5:
            head_h = dims[3]
        shape = "Round Head" if "flat" not in lowered and "hex" not in lowered else (
            "Flat Head" if "flat" in lowered else "Hex Head"
        )
        pitch = _extract_labeled_value(text, [r"Thread\s+Pitch", r"Pitch"])
        if pitch is None:
            if len(dims) >= 5:
                pitch = dims[4]
            elif len(dims) >= 4 and head_h is None:
                pitch = dims[3]
            else:
                pitch = max(0.8, round(shaft_dia * 0.15, 2))
        features = "External Threads"
        if "center hole" in lowered:
            features += " / Center Hole"
        return (
            "All required parameters collected.\n"
            "Summary:\n"
            f"- Object: Screw\n"
            f"- Dimensions: Shaft Dia={shaft_dia}mm, Length={length}mm, Head Diameter={head_dia}mm, Head Height={head_h or max((shaft_dia / 2) * 0.7, 3.0)}mm, Thread Pitch={pitch}mm\n"
            f"- Shape: {shape}\n"
            f"- Features: {features}\n"
            "Now the design process will be started."
        )

    if obj == "bolt":
        if len(dims) < 3:
            return None
        shaft_dia = _extract_labeled_value(text, [r"Shaft\s+Dia(?:meter)?", r"Diameter", r"Dia(?:meter)?"]) or dims[0]
        length = _extract_labeled_value(text, [r"Length", r"Shaft\s+Length"]) or (dims[1] if len(dims) >= 2 else 30.0)
        head_dia = _extract_labeled_value(text, [r"Head\s+Dia(?:meter)?"]) or (dims[2] if len(dims) >= 3 else shaft_dia * 1.8)
        head_h = _extract_labeled_value(text, [r"Head\s+Height", r"Head\s+Thickness"])
        if head_h is None and len(dims) >= 5:
            head_h = dims[3]
        pitch = _extract_labeled_value(text, [r"Thread\s+Pitch", r"Pitch"])
        if pitch is None:
            if len(dims) >= 5:
                pitch = dims[4]
            elif len(dims) >= 4 and head_h is None:
                pitch = dims[3]
            else:
                pitch = max(0.8, round(shaft_dia * 0.15, 2))
        return (
            "All required parameters collected.\n"
            "Summary:\n"
            f"- Object: Hex Bolt\n"
            f"- Dimensions: Shaft Dia={shaft_dia}mm, Length={length}mm, Head Diameter={head_dia}mm, Head Height={head_h or max((shaft_dia / 2) * 0.7, 4.0)}mm, Thread Pitch={pitch}mm\n"
            f"- Shape: Hex Head\n"
            f"- Features: External Threads\n"
            "Now the design process will be started."
        )

    if obj == "nut":
        if len(dims) < 3:
            return None
        pitch = _extract_labeled_value(text, [r"Thread\s+Pitch", r"Pitch"])
        pitch = pitch or (dims[3] if len(dims) >= 4 else max(0.8, round(dims[0] * 0.15, 2)))
        shape = "Hexagonal" if "hex" in lowered else "Circular"
        object_name = "Hex Nut" if "hex" in lowered else "Nut"
        return (
            "All required parameters collected.\n"
            "Summary:\n"
            f"- Object: {object_name}\n"
            f"- Dimensions: Inner Dia={dims[0]}mm, Outer Dia={dims[1]}mm, Thickness={dims[2]}mm, Thread Pitch={pitch}mm\n"
            f"- Shape: {shape}\n"
            f"- Features: Internal Threads\n"
            "Now the design process will be started."
        )

    if obj == "washer":
        if len(dims) < 3:
            return None
        return (
            "All required parameters collected.\n"
            "Summary:\n"
            f"- Object: Washer\n"
            f"- Dimensions: Inner Dia={dims[0]}mm, Outer Dia={dims[1]}mm, Thickness={dims[2]}mm\n"
            f"- Shape: Circular\n"
            f"- Features: Center Hole\n"
            "Now the design process will be started."
        )

    return None


def should_use_deterministic_pipeline(summary: str) -> bool:
    obj = detect_object(summary)
    if obj == "unknown":
        return False

    s = summary.lower()
    cls = classify_object(summary)

    if obj == "table":
        return False
    if cls == "fastener":
        if obj not in {"washer", "bushing"} or any(k in s for k in ("thread", "threaded")):
            return False
    if cls in {"gear", "assembly", "hole_part", "organic"}:
        return False

    complex_keywords = (
        "thread", "threaded", "screw", "bolt", "nut",
        "table", "gear", "helmet", "frame", "assembly",
        "hole", "bore", "center hole",
        "stepped", "pattern", "grid", "polar", "multi-hole", "multiple holes",
        "fillet", "chamfer", "shell",
    )
    if any(keyword in s for keyword in complex_keywords):
        return False

    return obj in {"box", "cylinder", "plate", "bracket", "washer", "bushing", "shaft"}


def _cache_key(*parts: str) -> str:
    payload = "\n||\n".join(part or "" for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  AGENTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

INTRO_SYSTEM = """You are the MECHAI Parameter Collector. Gather ALL geometric dimensions for 3D modeling.

BEHAVIOR:
1. If the user provides dimensions directly, do NOT ask about material or usage.
2. Ask at most ONE clarification question per turn.
3. Never repeat a question once answered.
4. Stop asking as soon as all required values are known.
5. ALWAYS collect thread pitch when object is a screw, bolt, nut, or stud.

PARAMETERS NEEDED:
- Screws/Bolts: Shaft Diameter, Length, Head Diameter, Head Type (hex/round/flat), Thread Pitch.
- Nuts: Inner Dia, Outer Dia, Thickness, Shape (hex/circle), Thread Pitch.
- Washers: Inner Dia, Outer Dia, Thickness.
- Cylinders: Diameter, Height.
- Boxes/Plates: Length, Width, Height/Thickness; ask if holes are needed.
- Corner-feature plates: Base Length, Base Width, Base Thickness, Corner Feature Length, Corner Feature Width, Corner Feature Height, and whether all 4 corners are required.
- Brackets: Base Length, Base Width, Wall Height, Thickness.
- Bushings/Shafts: Outer Dia, Inner Dia (if hollow), Length.

COMPLETION FORMAT â€” respond EXACTLY with:
"All required parameters collected.
Summary:
- Object: [exact object name e.g. Screw / Hex Bolt / Hex Nut / Washer / Cylinder / Box / Bracket / Plate / Bushing]
- Dimensions: [key=value pairs with units, e.g. Shaft Dia=6mm, Length=30mm, Thread Pitch=1mm]
- Shape: [e.g. Round Head / Hex Head / Hexagonal / Circular]
- Features: [list ALL features: e.g. External Threads / Internal Threads / Center Hole / Chamfer / Fillet / Shell / Holes]
Now the design process will be started."

PROHIBITIONS:
- NEVER ask about Material, Safety Factors, or stress.
- NEVER explain CAD.
- NEVER generate code.
- NEVER add extra notes outside the summary format."""


def prompt_agent(summary: str) -> str:
    """
    Agent 2: Converts dimension summary into a structured CAD blueprint.
    Enforces UNDERSTAND â†’ DECOMPOSE â†’ BUILD â†’ APPLY pipeline.
    """
    system = """You are the MECHAI Prompt Architect. Your job is to convert a dimension summary
into a precise, structured CAD blueprint for a CAD Coder who uses the build123d library.

MANDATORY PIPELINE â€” follow ALL four steps in order:

STEP 1 â€” UNDERSTAND:
  Read the summary. Identify the object type, all dimensions, and ALL features explicitly listed.

STEP 2 â€” DECOMPOSE:
  Break the object into individual geometric parts. List each sub-part on its own line:
  e.g.  PART 1: Shaft (cylinder, radius=3mm, length=30mm)
        PART 2: Hex Head (hexagonal prism, outer_r=5.5mm, height=5mm)
        PART 3: External M6 Thread (Helix pitch=1mm swept over shaft)

STEP 3 â€” BUILD SEQUENCE:
  State the exact build order (which part is built first, second, etc.)
  and which build123d operations to use for each.

STEP 4 â€” APPLY FEATURES:
  List EVERY feature from the summary and the exact build123d call to use:
  - Thread required?      â†’ Helix(radius, pitch, height) + sweep()
  - Hole required?        â†’ Hole(radius, depth)
  - Multiple holes?       â†’ PolarLocations or GridLocations + Hole()
  - Hollow/shell?         â†’ shell(thickness)
  - Chamfer/fillet?       â†’ chamfer(edge, length) / fillet(edge, radius)

CRITICAL RULES:
1. Library: build123d ONLY.
2. Export: export_stl(b.part, 'path') â€” standalone function, NOT b.part.export_stl().
3. BuildSketch MUST be nested INSIDE BuildPart context.
4. extrude() and Hole() MUST be indented inside BuildPart block.
5. For screws/bolts: shaft first, then head on top (offset plane + extrude).
6. Define ALL dimensions as named float variables at the top.
7. NEVER omit a feature that appears in the summary.
8. NEVER replace a threaded shaft with a plain cylinder.
9. NEVER replace an internal thread with a plain Hole().
10. Output ENGLISH ONLY. Do not use Chinese or any other language.

CORNER FEATURE RULES (STRICT):
1. If the summary includes "4 corners", "each corner", or corner rectangles/posts, this is mandatory multi-part geometry.
2. Build order must be: base first, then corner features.
3. Corner features must be positioned at the four corners using offsets derived from length/width.
4. Never collapse corner-feature requests into one primitive box.

Output ONLY the structured blueprint (Steps 1â€“4). No code, no preamble."""

    cache_id = _cache_key("prompt", summary)
    cached = _PROMPT_CACHE.get(cache_id)
    if cached:
        return cached

    res = client.chat.completions.create(
        model="qwen2:7b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Summary:\n{summary}"},
        ],
        temperature=0,
        max_tokens=PROMPT_MAX_TOKENS,
    )
    output = res.choices[0].message.content
    _PROMPT_CACHE[cache_id] = output
    return output


def coder_agent(blueprint: str, stl_path: str, strict_instruction: str = "") -> str:
    """
    Agent 3: Generates complete, executable build123d Python code from blueprint.
    Enforces all feature rules: threads, holes, Shell, chamfer, fillet, patterns.
    """
    safe_path = stl_path.replace("\\", "/")

    system = f"""You are the MECHAI CAD Coder. Write a complete, executable Python script using build123d.

You will receive a structured blueprint (UNDERSTAND / DECOMPOSE / BUILD / APPLY).
You MUST implement EVERY part and EVERY feature described in the blueprint.
You MUST generate complete geometry with full stacking and no omissions.
For tables, model tabletop + exactly 4 legs.
For cylinders/plates with holes, always use Hole().
For stepped shafts, build multiple stacked sections using multiple extrudes.
If user intent includes corner features (e.g. vertical rectangles/posts at 4 corners), build a base first and then exactly 4 corner bodies using Locations().
For 4-corner requests, use corner placement tied to length/width offsets (corners), never center placement.

â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
THREAD GENERATION â€” MANDATORY (NO EXCEPTIONS)
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
External threads (screws, bolts, studs):
  from build123d import *
  shaft_r = 3.0; pitch = 1.0; shaft_len = 30.0
  thread_depth = pitch * 0.6
  with BuildPart() as b:
      # 1. Shaft core
      with BuildSketch(Plane.XY):
          Circle(radius=shaft_r)
      extrude(amount=shaft_len)
      # 2. Thread helix path
      helix = Helix(pitch=pitch, height=shaft_len, radius=shaft_r)
      # 3. Thread profile (triangle cross-section at helix start)
      with BuildSketch(Plane.XY):
          with BuildLine():
              pts = [(shaft_r, 0), (shaft_r + thread_depth, pitch/2), (shaft_r, pitch)]
              Polyline(*pts, close=True)
          make_face()
      thread_profile = b.sketch
      # 4. Sweep profile along helix
      sweep(sections=thread_profile, path=helix, multisection=False)

Internal threads (nuts):
  # 1. Build hex/circle outer body + extrude
  # 2. Cut center bore with Hole(radius=inner_r)
  # 3. Build internal thread helix at inner_r, sweep inward profile
  helix = Helix(pitch=pitch, height=thickness, radius=inner_r)
  # sweep a small inward-pointing triangular profile along helix

â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
HOLE PATTERNS
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  # Bolt circle (polar):
  with PolarLocations(bolt_circle_r, hole_count):
      Hole(radius=hole_r, depth=plate_thickness)

  # Grid holes:
  with GridLocations(x_spacing, y_spacing, cols, rows):
      Hole(radius=hole_r, depth=plate_thickness)

â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
SHELL (hollow objects)
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  with BuildPart() as b:
      Box(length, width, height)
      Shell(amount=-wall_thickness, openings=b.faces().sort_by(Axis.Z)[-1:])

â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
CHAMFER / FILLET
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  # Chamfer top edge of shaft:
  chamfer(b.edges().sort_by(Axis.Z)[-1], length=1.0)
  # Fillet a specific edge:
  fillet(b.edges().filter_by(GeomType.LINE)[0], radius=0.5)

â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
GENERAL RULES
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
1. export_stl(b.part, '{safe_path}')  â€” standalone function, last line.
2. All extrude / Hole / chamfer / fillet calls INSIDE BuildPart block.
3. BuildSketch MUST be nested inside BuildPart.
4. Define ALL dimensions as named float variables at the top of the script.
5. Raw Python ONLY â€” NO markdown, NO backticks, NO comments.
6. If threads are required â†’ implement Helix + sweep. A plain cylinder is FORBIDDEN.
7. If a hole is required â†’ use Hole(). A missing hole is FORBIDDEN.
8. If shell/hollow is required â†’ use shell(). A solid body is FORBIDDEN.
9. If multiple holes are required â†’ use PolarLocations or GridLocations.
10. NEVER simplify a complex object into a single primitive.
11. For external threads use Helix + sweep with mode=Mode.ADD, while keeping shaft/body solids.
12. If 4 corner features are required, include exactly 4 corner placements and do not omit any.
13. For corner-feature parts, use BuildPart + BuildSketch + Locations with corner offsets based on length/width.

Output ONLY the complete Python script â€” nothing else."""

    if strict_instruction.strip():
        system += f"\n\n{strict_instruction.strip()}"
    cache_id = _cache_key("coder", safe_path, blueprint, strict_instruction)
    cached = _CODER_CACHE.get(cache_id)
    if cached:
        return cached

    res = client.chat.completions.create(
        model="deepseek-coder:6.7b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": blueprint},
        ],
        temperature=0,
        max_tokens=CODER_MAX_TOKENS,
    )
    raw = res.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    cleaned = raw.strip()
    _CODER_CACHE[cache_id] = cleaned
    return cleaned


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  EXECUTION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def patch_export(code: str, stl_path: str) -> str:
    """Rewrite any export_stl call to the correct standalone form."""
    safe_path = stl_path.replace("\\", "/")
    lines = code.splitlines()
    out, injected = [], False
    for line in lines:
        if "export_stl" in line:
            out.append(f"export_stl(b.part, '{safe_path}')")
            injected = True
        else:
            out.append(line)
    if not injected:
        out.append(f"export_stl(b.part, '{safe_path}')")
    return "\n".join(out)


def _count_location_tuples(code_text: str) -> int:
    total = 0
    calls = re.findall(
        r"with\s+Locations\s*\((.*?)\)\s*:",
        code_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for call in calls:
        total += len(re.findall(r"\(\s*[^()]+?\s*,\s*[^()]+?\s*\)", call))
    return total


def validate(summary: str, code: str) -> str:
    s = (summary or "").lower()
    code_text = code or ""
    code_lower = code_text.lower()

    if len(code_text.strip()) < CODE_MIN_LENGTH:
        return "invalid_output_too_short"
    if "export_stl(" not in code_lower:
        return "missing_export"

    extrude_count = len(re.findall(r"\bextrude\s*\(", code_text, flags=re.IGNORECASE))
    if "table" in s and extrude_count < 5:
        return "missing_parts_table"
    if "shaft" in s and "stepped" in s and extrude_count < 3:
        return "missing_parts_stepped_shaft"

    needs_hole = ("hole" in s or "bore" in s or classify_object(summary) == "hole_part")
    if needs_hole and "hole(" not in code_lower:
        return "missing_hole"

    needs_thread = any(keyword in s for keyword in ("thread", "threaded", "screw", "bolt", "nut", "stud"))
    if needs_thread:
        if "helix(" not in code_lower and "sweep(" not in code_lower:
            return "missing_thread_feature"
        if "helix(" not in code_lower:
            return "missing_thread_helix"
        if "sweep(" not in code_lower:
            return "missing_thread_sweep"
        if any(k in s for k in ("screw", "bolt", "stud")) and "mode.add" not in code_lower:
            return "missing_threads_mode_add"
        if extrude_count < 1:
            return "thread_replaces_shaft"

    if "gear" in s and "polarlocations" not in code_lower:
        return "missing_gear_pattern"

    if any(k in s for k in ("holes", "pattern", "grid", "bolt circle", "multi-hole", "multiple holes")):
        if "gridlocations" not in code_lower and "polarlocations" not in code_lower:
            return "missing_hole_pattern"

    needs_corner_features = _has_corner_feature_request(summary)
    if needs_corner_features:
        if "locations(" not in code_lower:
            return "missing_corner_locations"
        if extrude_count < 2:
            return "missing_corner_extrudes"
        if _requires_four_corners(summary) and _count_location_tuples(code_text) < 4:
            return "missing_four_corner_features"
        if "x_off" not in code_lower and "y_off" not in code_lower and "length / 2" not in code_lower:
            return "missing_corner_offset_logic"

    return "ok"


def validate_code(code: str, summary: str) -> str:
    return validate(summary, code)


THREAD_TRIGGER_KEYWORDS = (
    "thread",
    "threaded",
    "screw",
    "bolt",
    "stud",
    "nut",
)


def threads_required(summary: str) -> bool:
    s = summary.lower()
    return any(keyword in s for keyword in THREAD_TRIGGER_KEYWORDS)


def detect_thread_mode(summary: str) -> str | None:
    s = summary.lower()
    if "nut" in s:
        return "internal"
    if any(keyword in s for keyword in ("screw", "bolt", "stud", "threaded rod", "threaded")):
        return "external"
    return None


def _extract_labeled_value(summary: str, labels) -> float | None:
    for label in labels:
        pattern = rf"{label}(?:\s+is)?\s*(?:=|:)?\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, summary, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def infer_thread_parameters(summary: str) -> dict | None:
    if not threads_required(summary):
        return None

    s = summary.lower()
    dims = parse_dims(summary)
    mode = detect_thread_mode(summary)
    if mode is None:
        return None

    pitch = _extract_labeled_value(summary, [r"Thread\s+Pitch", r"Pitch"])

    if mode == "internal":
        inner_dia = _extract_labeled_value(summary, [r"Inner\s+Dia(?:meter)?", r"Bore\s+Dia(?:meter)?"])
        thickness = _extract_labeled_value(summary, [r"Thickness", r"Height", r"Length"])

        if inner_dia is None and dims:
            inner_dia = dims[0]
        if thickness is None:
            thickness = dims[2] if len(dims) >= 3 else (dims[1] if len(dims) >= 2 else 6.0)

        inner_dia = inner_dia or 10.0
        thickness = thickness or 6.0
        pitch = pitch or max(0.8, round(inner_dia * 0.15, 2))

        return {
            "mode": "internal",
            "inner_radius": max(inner_dia / 2.0, 0.5),
            "length": max(thickness, pitch * 2.0),
            "pitch": max(pitch, 0.4),
        }

    shaft_dia = _extract_labeled_value(
        summary,
        [r"Shaft\s+Dia(?:meter)?", r"Major\s+Dia(?:meter)?", r"Diameter", r"Dia(?:meter)?"],
    )
    shaft_len = _extract_labeled_value(summary, [r"Length", r"Shaft\s+Length", r"Thread\s+Length"])

    if shaft_dia is None and dims:
        shaft_dia = dims[0]
    if shaft_len is None:
        if "threaded rod" in s and len(dims) >= 2:
            shaft_len = dims[1]
        else:
            shaft_len = dims[1] if len(dims) >= 2 else 20.0

    shaft_dia = shaft_dia or 6.0
    shaft_len = shaft_len or 20.0
    pitch = pitch or max(0.8, round(shaft_dia * 0.15, 2))

    return {
        "mode": "external",
        "shaft_radius": max(shaft_dia / 2.0, 0.5),
        "length": max(shaft_len, pitch * 2.0),
        "pitch": max(pitch, 0.4),
    }


def inject_thread_postprocessing(code: str, summary: str, stl_path: str) -> str:
    params = infer_thread_parameters(summary)
    if not params:
        return patch_export(code, stl_path)
    if re.search(r"\bHelix\s*\(", code) and re.search(r"\bsweep\s*\(", code):
        return patch_export(code, stl_path)

    safe_path = stl_path.replace("\\", "/")
    clean = patch_export(code, stl_path)

    if params["mode"] == "internal":
        apply_call = (
            f"result_part = _mechai_apply_internal_threads(b.part, {params['inner_radius']}, "
            f"{params['length']}, {params['pitch']})"
        )
    else:
        apply_call = (
            f"result_part = _mechai_apply_external_threads(b.part, {params['shaft_radius']}, "
            f"{params['length']}, {params['pitch']})"
        )

    helper_block = f"""
def _mechai_make_thread_section(path, pitch, radial_depth, root_overlap, inward=False):
    section_plane = Plane(origin=path @ 0, z_dir=path % 0)
    half_pitch = max(pitch * 0.22, radial_depth * 1.5)
    root = -root_overlap
    crest = radial_depth if not inward else -radial_depth
    with BuildSketch(section_plane) as section:
        Polygon(
            (root, -half_pitch),
            (crest, 0),
            (root, half_pitch),
        )
    return section.sketch


def _mechai_make_external_thread(radius, length, pitch):
    radial_depth = max(min(pitch * 0.18, radius * 0.10), 0.08)
    root_overlap = max(radial_depth * 0.55, 0.05)
    helix_radius = max(radius - root_overlap * 0.25, 0.2)
    path = Helix(pitch=pitch, height=length, radius=helix_radius)
    section = _mechai_make_thread_section(path, pitch, radial_depth, root_overlap, inward=False)
    return sweep(section, path=path)


def _mechai_make_internal_thread(radius, length, pitch):
    radial_depth = max(min(pitch * 0.14, radius * 0.08), 0.05)
    root_overlap = max(radial_depth * 0.45, 0.04)
    helix_radius = max(radius + root_overlap * 0.20, 0.2)
    path = Helix(pitch=pitch, height=length, radius=helix_radius)
    section = _mechai_make_thread_section(path, pitch, radial_depth, root_overlap, inward=True)
    return sweep(section, path=path)


def _mechai_apply_external_threads(base_part, shaft_radius, shaft_length, pitch):
    pitch = max(float(pitch), 0.4)
    shaft_radius = max(float(shaft_radius), 0.5)
    shaft_length = max(float(shaft_length), pitch * 2.0)

    for pitch_scale in (1.0, 1.15):
        try:
            thread = _mechai_make_external_thread(shaft_radius, shaft_length, pitch * pitch_scale)
            with BuildPart() as result:
                add(base_part)
                add(thread, mode=Mode.ADD)
            return result.part
        except Exception:
            pass
    return base_part


def _mechai_apply_internal_threads(base_part, inner_radius, length, pitch):
    pitch = max(float(pitch), 0.4)
    inner_radius = max(float(inner_radius), 0.5)
    length = max(float(length), pitch * 2.0)

    for pitch_scale in (1.0, 1.15):
        try:
            thread = _mechai_make_internal_thread(inner_radius, length, pitch * pitch_scale)
            with BuildPart() as result:
                add(base_part)
                add(thread, mode=Mode.SUBTRACT)
            return result.part
        except Exception:
            pass
    return base_part
"""

    export_pattern = re.compile(r"^\s*export_stl\s*\([^)]*\)\s*$", flags=re.MULTILINE)
    replacement = f"{apply_call}\nexport_stl(result_part, '{safe_path}')"
    clean = export_pattern.sub(replacement, clean, count=1)

    if (
        "def _mechai_apply_external_threads" not in clean
        and "def _mechai_apply_internal_threads" not in clean
    ):
        clean = helper_block.strip() + "\n\n" + clean.lstrip()
    return clean


def execute_code(code: str, stl_path: str, summary: str = "") -> str:
    """
    Execute generated build123d code in a subprocess.
    Returns the STL path on success, raises RuntimeError on failure.
    """
    safe_path = stl_path.replace("\\", "/")

    # Ensure the export path inside the code is correct
    clean = re.sub(
        r"export_stl\s*\([^,]+,\s*['\"][^'\"]*['\"]\s*\)",
        f"export_stl(b.part, '{safe_path}')",
        code,
    )
    clean = inject_thread_postprocessing(clean, summary, stl_path)
    clean = re.sub(
        r"^\s*export_stl\s*\([^)]*\)\s*$",
        "# MECHAI deferred export",
        clean,
        flags=re.MULTILINE,
    )
    clean = clean.rstrip() + f"""

import os

_mechai_export_target = None
if "result_part" in globals() and result_part is not None:
    _mechai_export_target = result_part
elif "b" in globals() and hasattr(b, "part"):
    _mechai_export_target = b.part

if _mechai_export_target is None:
    raise RuntimeError("No exportable part found in generated script.")

os.makedirs(os.path.dirname(r"{safe_path}"), exist_ok=True)
export_stl(_mechai_export_target, r"{safe_path}")

if not os.path.exists(r"{safe_path}") or os.path.getsize(r"{safe_path}") == 0:
    raise RuntimeError("STL export failed or produced an empty file.")
"""

    if VERBOSE_LOGS:
        print("\n[Engine] Executing:\n" + "=" * 40)
        print(clean[:2000] + ("\n... [truncated]" if len(clean) > 2000 else ""))
        print("=" * 40)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(clean)
        tmp = f.name

    try:
        if os.path.exists(stl_path):
            os.remove(stl_path)

        proc = subprocess.run(
            [sys.executable, tmp],
            stdout=subprocess.PIPE if VERBOSE_LOGS else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if VERBOSE_LOGS and proc.stdout:
            print("[Engine] STDOUT:", proc.stdout)
        if proc.stderr and (proc.returncode != 0 or VERBOSE_LOGS):
            print("[Engine] STDERR:", proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or "Generated script failed.")

        if os.path.exists(stl_path):
            return stl_path
        raise RuntimeError("Script ran but no STL was created.")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def run_pipeline(summary: str):
    normalized_summary = " ".join(summary.strip().split())
    cache_id = _cache_key("pipeline", normalized_summary.lower())
    cached_path = _PIPELINE_CACHE.get(cache_id)
    if cached_path and os.path.exists(cached_path):
        if VERBOSE_LOGS:
            print(f"[MECHAI] Cache hit - {cached_path}")
        return cached_path

    print("\n" + "=" * 50 + "\n[MECHAI] Pipeline start\n" + "=" * 50)

    detected_obj = detect_object(summary)
    classified_obj = classify_object(summary)

    if should_use_deterministic_pipeline(summary):
        print("[MECHAI] Using deterministic simple-shape pipeline.")
        try:
            path = generate_fallback(summary, STL_PATH)
            _PIPELINE_CACHE[cache_id] = path
            print(f"[Deterministic] SUCCESS - {path}")
            return path
        except Exception as e:
            print(f"[Deterministic] Failed: {e}\n-> Falling back to LLM pipeline...")

    # STAGE 1: GENERATE
    blueprint = None
    initial_code = generate_specialized_code(summary, STL_PATH)
    if initial_code is not None and (
        detected_obj == "table"
        or classified_obj == "gear"
        or classified_obj == "hole_part"
        or classified_obj == "corner_feature_part"
        or detected_obj in {"gear", "helmet", "frame"}
        or classified_obj == "organic"
    ):
        code = patch_export(initial_code, STL_PATH)
    else:
        blueprint = prompt_agent(summary)
        if VERBOSE_LOGS:
            print(f"[Agent 2] Blueprint:\n{blueprint[:400]}\n")
        raw_code = coder_agent(blueprint, STL_PATH)
        code = patch_export(raw_code, STL_PATH)

    # STAGE 2: VALIDATE
    validation_error = validate(summary, code)

    # STAGE 3: AUTO-FIX (RETRY)
    if validation_error != "ok":
        print(f"[Validate] FAILED (1/{MAX_AUTOFIX_RETRIES + 1}) - {validation_error}")
        if blueprint is None:
            blueprint = prompt_agent(summary)
            if VERBOSE_LOGS:
                print(f"[Agent 2] Blueprint:\n{blueprint[:400]}\n")

        for retry_idx in range(MAX_AUTOFIX_RETRIES):
            strict_instruction = (
                f"STRICT FIX: previous output invalid because {validation_error}. Regenerate correctly. "
                "MUST include it. DO NOT skip. "
                "Threads MUST use Helix + sweep and Mode.ADD while keeping the shaft/body. "
                "Holes MUST be created using Hole(). "
                "Tables MUST include 1 tabletop and 4 legs. "
                "Stepped shafts MUST include at least 3 stacked extrudes. "
                "Gears MUST include PolarLocations. "
                "If corner features are requested, build base first and include exactly 4 corner features at corner offsets using Locations()."
            )
            raw_code = coder_agent(blueprint, STL_PATH, strict_instruction)
            candidate_code = patch_export(raw_code, STL_PATH)
            validation_error = validate(summary, candidate_code)
            if validation_error == "ok":
                code = candidate_code
                break
            print(
                f"[Validate] FAILED ({retry_idx + 2}/{MAX_AUTOFIX_RETRIES + 1}) - {validation_error}"
            )

    if validation_error == "ok":
        if VERBOSE_LOGS:
            print(f"[Agent 3] Code:\n{code[:400]}\n")

        exec_error = None
        for exec_attempt in range(2):
            try:
                path = execute_code(code, STL_PATH, summary)
                if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                    raise RuntimeError("Execution produced missing or empty STL.")
                _PIPELINE_CACHE[cache_id] = path
                print(f"[Engine] SUCCESS - {path}")
                return path
            except Exception as e:
                exec_error = e
                print(f"[Engine] Execution failed ({exec_attempt + 1}/2): {e}")
        print(f"[Engine] Execution retry exhausted: {exec_error}\n-> Switching to fallback...")
    else:
        print(f"[Validate] Final failure: {validation_error}\n-> Switching to fallback...")

    try:
        fallback_specialized = generate_specialized_code(summary, STL_PATH)
        if fallback_specialized is not None:
            fallback_exec_error = None
            for exec_attempt in range(2):
                try:
                    path = execute_code(fallback_specialized, STL_PATH, summary)
                    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                        raise RuntimeError("Specialized fallback produced missing or empty STL.")
                    _PIPELINE_CACHE[cache_id] = path
                    print(f"[Fallback-Specialized] SUCCESS - {path}")
                    return path
                except Exception as e:
                    fallback_exec_error = e
                    print(f"[Fallback-Specialized] Execution failed ({exec_attempt + 1}/2): {e}")
            print(f"[Fallback-Specialized] Failed after retry: {fallback_exec_error}")

        path = generate_fallback(summary, STL_PATH)
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError("Fallback produced missing or empty STL.")
        _PIPELINE_CACHE[cache_id] = path
        print(f"[Fallback] SUCCESS - {path}")
        return path
    except Exception as e:
        print(f"[Fallback] Failed: {e}\n{traceback.format_exc()}")

    return None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CHAT HANDLER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def chat_handler(user_message, history):
    direct_summary = build_direct_summary(user_message)
    detected_obj = detect_object(user_message)
    if direct_summary or detected_obj != "unknown":
        pipeline_input = direct_summary or user_message.strip()
        bot_reply = direct_summary or (
            "Proceeding with generation from your request.\n"
            f"Detected object: {detected_obj.title()}"
        )
        stl_file = None
        viewer_file = None
        try:
            stl_file = run_pipeline(pipeline_input)
            viewer_file = prepare_viewer_model(stl_file) if stl_file else None
            bot_reply += "\n\nâœ… **3D model generated!** Check the viewer â†’" if stl_file \
                         else "\n\nâš ï¸ Generation failed. Please try again."
        except Exception as e:
            print(traceback.format_exc())
            bot_reply += f"\n\nâš ï¸ Error: {str(e)}"

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_reply})
        return "", history, viewer_file

    messages = [{"role": "system", "content": INTRO_SYSTEM}]
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    res = client.chat.completions.create(
        model="qwen2:7b",
        messages=messages,
        temperature=0,
        max_tokens=COLLECTOR_MAX_TOKENS,
    )
    bot_reply = res.choices[0].message.content

    stl_file = None
    viewer_file = None
    if "All required parameters collected" in bot_reply:
        try:
            stl_file = run_pipeline(bot_reply)
            viewer_file = prepare_viewer_model(stl_file) if stl_file else None
            bot_reply += "\n\nâœ… **3D model generated!** Check the viewer â†’" if stl_file \
                         else "\n\nâš ï¸ Generation failed. Please try again."
        except Exception as e:
            print(traceback.format_exc())
            bot_reply += f"\n\nâš ï¸ Error: {str(e)}"

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": bot_reply})
    return "", history, viewer_file


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  GRADIO UI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

CUSTOM_CSS = """
.gradio-container{background:#0f1319 !important;font-family:'Segoe UI',system-ui,sans-serif;}
#header{text-align:center;padding:20px 0;border-bottom:1px solid #1e2a3a;margin-bottom:10px;}
#header h1{color:#22d3a0;font-size:2.2em;font-weight:700;letter-spacing:2px;margin:0;text-shadow:0 0 30px rgba(34,211,160,0.3);}
#header p{color:#6b7b8d;font-size:0.9em;margin:5px 0 0 0;font-family:monospace;letter-spacing:3px;text-transform:uppercase;}
.chatbot{background:#111820 !important;border:1px solid #1e2a3a !important;border-radius:12px !important;min-height:450px !important;}
.message{border-radius:16px !important;}
.input-row{gap:8px;}
#user-input textarea{background:#1a2332 !important;border:1px solid #2a3a4a !important;border-radius:12px !important;color:#e0e8f0 !important;font-size:15px !important;padding:12px 16px !important;}
#user-input textarea:focus{border-color:#22d3a0 !important;box-shadow:0 0 15px rgba(34,211,160,0.15) !important;}
#send-btn{background:linear-gradient(135deg,#22d3a0,#1aaa80) !important;border:none !important;border-radius:12px !important;color:#0f1319 !important;font-weight:600 !important;min-width:100px !important;transition:all 0.2s !important;}
#send-btn:hover{box-shadow:0 0 20px rgba(34,211,160,0.4) !important;transform:translateY(-1px) !important;}
#clear-btn{background:transparent !important;border:1px solid #2a3a4a !important;border-radius:12px !important;color:#6b7b8d !important;}
#model-viewer{border:1px solid #1e2a3a !important;border-radius:12px !important;background:#111820 !important;min-height:400px !important;}
.label-text,label{color:#8899aa !important;font-family:monospace !important;text-transform:uppercase !important;letter-spacing:1px !important;font-size:0.8em !important;}
#footer{text-align:center;padding:15px 0;border-top:1px solid #1e2a3a;margin-top:10px;}
#footer p{color:#3a4a5a;font-family:monospace;font-size:0.75em;letter-spacing:2px;}
"""


def create_ui():
    with gr.Blocks(
        css=CUSTOM_CSS, title="MECHAI â€” Mechanical Design AI",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Space Grotesk"),
        ).set(
            body_background_fill="#0f1319", body_background_fill_dark="#0f1319",
            block_background_fill="#111820", block_background_fill_dark="#111820",
            input_background_fill="#1a2332", input_background_fill_dark="#1a2332",
            button_primary_background_fill="#22d3a0", button_primary_text_color="#0f1319",
        ),
    ) as app:
        gr.HTML('<div id="header"><h1>MEC<span style="color:#22d3a0">HAI</span></h1><p>Mechanical Design AI</p></div>')

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant",
                            "content": "Hello! I'm **MECHAI**, your AI-powered mechanical design assistant.\n\nWhat are we designing today?"}],
                    elem_classes=["chatbot"], height=500, show_label=False,
                )
                with gr.Row(elem_classes=["input-row"]):
                    user_input = gr.Textbox(
                        placeholder="e.g. 'M6 screw, 30mm long' or 'hex nut inner 10mm outer 20mm thick 8mm'",
                        show_label=False, elem_id="user-input", scale=5, lines=1, max_lines=3,
                    )
                    send_btn = gr.Button("Send âž¤", elem_id="send-btn", scale=1, variant="primary")
                clear_btn = gr.Button("ðŸ”„ New Design", elem_id="clear-btn", size="sm")

            with gr.Column(scale=2):
                gr.Markdown("#### ðŸ”§ 3D Model Viewer")
                model_viewer = gr.Model3D(
                    label="Generated Model", elem_id="model-viewer",
                    height=500, clear_color=[0.059, 0.075, 0.098, 1.0],
                )
                gr.Markdown("*Your 3D model will appear here after generation.*", elem_classes=["label-text"])

        gr.HTML('<div id="footer"><p>MECHAI ENGINE â€¢ Ollama + build123d</p></div>')

        send_btn.click(fn=chat_handler, inputs=[user_input, chatbot], outputs=[user_input, chatbot, model_viewer])
        user_input.submit(fn=chat_handler, inputs=[user_input, chatbot], outputs=[user_input, chatbot, model_viewer])
        clear_btn.click(
            fn=lambda: ([{"role": "assistant", "content": "Ready for a new design! What would you like to create?"}], None),
            outputs=[chatbot, model_viewer],
        )
    return app


if __name__ == "__main__":
    print("=" * 50)
    print("  MECHAI â€” Mechanical Design AI")
    print(f"  STL output: {STL_PATH}")
    print("=" * 50)
    create_ui().launch(inbrowser=True)

