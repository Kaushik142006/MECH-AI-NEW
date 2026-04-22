"""
Image‑to‑Model parser: OCR + dimension extraction → engineering summary.
Strict per‑object parameter filtering – no extra/zero/invalid values.
Now fail‑safe: uses default dimensions when OCR fails.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
import cv2

# Try EasyOCR first
try:
    import easyocr
    OCR_READER = easyocr.Reader(['en'], gpu=False)
    OCR_BACKEND = "easyocr"
except ImportError:
    try:
        import pytesseract
        OCR_BACKEND = "tesseract"
    except ImportError:
        OCR_BACKEND = None
        print("⚠️ No OCR backend. Install 'easyocr' or 'pytesseract'.")

# ----------------------------------------------------------------------
# Allowed dimensions per object (keys that can be passed to modelling)
# ----------------------------------------------------------------------
OBJECT_ALLOWED_KEYS = {
    "bolt": ["shaft_dia", "shaft_length", "head_dia", "head_height", "pitch"],
    "screw": ["shaft_dia", "shaft_length", "head_dia", "head_height", "pitch"],
    "nut": ["inner_dia", "outer_dia", "thickness"],
    "washer": ["inner_dia", "outer_dia", "thickness"],
    "gear": ["outer_dia", "thickness", "tooth_count", "hole_dia"],
    "bevel_gear": ["outer_dia", "face_width", "tooth_count", "hole_dia", "tooth_depth"],
    "helical_gear": ["outer_dia", "face_width", "tooth_count", "helix_angle", "hole_dia", "tooth_depth"],
    "spring": ["mean_dia", "wire_dia", "free_length", "pitch"],
    "cylinder": ["diameter", "height", "hole_dia"],
    "sphere": ["diameter"],
    "cone": ["diameter", "height"],
    "box": ["length", "width", "height"],
    "plate": ["length", "width", "thickness", "hole_count", "hole_dia"],
    "shaft": ["diameter", "length"],
    "bushing": ["inner_dia", "outer_dia", "height"],
    "pulley": ["outer_dia", "face_width", "bore_dia", "groove_dia", "hub_dia", "flange_thickness"],
    "hinge": ["leaf_length", "leaf_width", "leaf_thickness", "pin_dia", "knuckle_dia", "knuckle_count", "hole_dia", "holes_per_leaf"],
    "bracket": ["base_length", "base_width", "wall_height", "thickness"],
    "rivet": ["shank_dia", "shank_length", "head_dia", "head_height"],
    "table": ["top_length", "top_width", "top_thickness", "leg_height", "leg_width"],
    "frame": ["length", "width", "height", "thickness"],
    "helmet": ["outer_dia", "wall_thickness"],
}

# Required keys per object (must be present, >0)
OBJECT_REQUIRED_KEYS = {
    "bolt": ["shaft_dia", "shaft_length"],
    "screw": ["shaft_dia", "shaft_length"],
    "nut": ["inner_dia", "outer_dia", "thickness"],
    "washer": ["inner_dia", "outer_dia", "thickness"],
    "gear": ["outer_dia", "thickness", "tooth_count"],
    "bevel_gear": ["outer_dia", "face_width", "tooth_count"],
    "helical_gear": ["outer_dia", "face_width", "tooth_count", "helix_angle"],
    "spring": ["mean_dia", "wire_dia", "free_length"],
    "cylinder": ["diameter", "height"],
    "sphere": ["diameter"],
    "cone": ["diameter", "height"],
    "box": ["length", "width", "height"],
    "plate": ["length", "width", "thickness"],
    "shaft": ["diameter", "length"],
    "bushing": ["inner_dia", "outer_dia", "height"],
    "pulley": ["outer_dia", "face_width", "bore_dia"],
    "hinge": ["leaf_length", "leaf_width", "leaf_thickness", "pin_dia"],
    "bracket": ["base_length", "base_width", "wall_height", "thickness"],
    "rivet": ["shank_dia", "shank_length", "head_dia", "head_height"],
    "table": ["top_length", "top_width", "top_thickness", "leg_height"],
    "frame": ["length", "width", "height", "thickness"],
    "helmet": ["outer_dia"],
}

# ----------------------------------------------------------------------
# Keyword patterns for dimension extraction
# ----------------------------------------------------------------------
KEYWORD_PATTERNS = {
    "diameter": r"(?:outer|overall)?\s*diameter\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "inner_dia": r"(?:inner|bore|hole)\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "outer_dia": r"(?:outer|overall)\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "shaft_dia": r"shaft\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "head_dia": r"head\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "shaft_length": r"shaft\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "length": r"(?:total|free)?\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "height": r"height\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "width": r"width\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "thickness": r"(?:plate|base|leaf)?\s*thickness\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "face_width": r"face\s*width\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "tooth_count": r"(?:tooth\s*count|teeth)\s*[:=]?\s*(\d+)",
    "helix_angle": r"helix\s*angle\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "pitch": r"pitch\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "wire_dia": r"wire\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "mean_dia": r"mean\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "free_length": r"free\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "hole_dia": r"hole\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "bore_dia": r"bore\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "leaf_length": r"leaf\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "leaf_width": r"leaf\s*width\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "leaf_thickness": r"leaf\s*thickness\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "pin_dia": r"pin\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "base_length": r"base\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "base_width": r"base\s*width\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "wall_height": r"wall\s*height\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "top_length": r"top\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "top_width": r"top\s*width\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "top_thickness": r"top\s*thickness\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "leg_height": r"leg\s*height\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "leg_width": r"leg\s*width\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "shank_dia": r"shank\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "shank_length": r"shank\s*length\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "head_height": r"head\s*height\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "knuckle_dia": r"knuckle\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "knuckle_count": r"knuckle\s*count\s*[:=]?\s*(\d+)",
    "holes_per_leaf": r"holes\s*per\s*leaf\s*[:=]?\s*(\d+)",
    "groove_dia": r"groove\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "hub_dia": r"hub\s*(?:diameter|dia)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "flange_thickness": r"flange\s*thickness\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "wall_thickness": r"wall\s*thickness\s*[:=]?\s*(\d+(?:\.\d+)?)",
    "tooth_depth": r"tooth\s*depth\s*[:=]?\s*(\d+(?:\.\d+)?)",
}

SYMBOL_PATTERNS = [
    (r"Ø\s*(\d+(?:\.\d+)?)", "diameter"),
    (r"⌀\s*(\d+(?:\.\d+)?)", "diameter"),
]

# ----------------------------------------------------------------------
# Helper: extract a value using regex
# ----------------------------------------------------------------------
def _extract_value(patterns: List[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

# ----------------------------------------------------------------------
# Generate default dimensions for any object (fail‑safe)
# ----------------------------------------------------------------------
def generate_default_dimensions(obj_type: str) -> Dict[str, float]:
    """Return sensible default dimensions for the given object type."""
    defaults = {
        "pulley": {"outer_dia": 100.0, "face_width": 30.0, "bore_dia": 20.0},
        "gear": {"outer_dia": 60.0, "thickness": 10.0, "tooth_count": 20.0, "hole_dia": 12.0},
        "bevel_gear": {"outer_dia": 60.0, "face_width": 12.0, "tooth_count": 20, "hole_dia": 12.0},
        "helical_gear": {"outer_dia": 60.0, "face_width": 15.0, "tooth_count": 20, "helix_angle": 25.0, "hole_dia": 12.0},
        "nut": {"inner_dia": 10.0, "outer_dia": 20.0, "thickness": 8.0},
        "washer": {"inner_dia": 10.0, "outer_dia": 20.0, "thickness": 2.0},
        "bolt": {"shaft_dia": 10.0, "shaft_length": 50.0, "head_dia": 16.0, "head_height": 6.0},
        "screw": {"shaft_dia": 6.0, "shaft_length": 30.0, "head_dia": 10.0, "head_height": 4.0},
        "spring": {"mean_dia": 20.0, "wire_dia": 2.0, "free_length": 50.0, "pitch": 3.0},
        "cylinder": {"diameter": 30.0, "height": 50.0},
        "sphere": {"diameter": 40.0},
        "cone": {"diameter": 40.0, "height": 50.0},
        "box": {"length": 50.0, "width": 40.0, "height": 30.0},
        "plate": {"length": 80.0, "width": 60.0, "thickness": 8.0},
        "shaft": {"diameter": 20.0, "length": 100.0},
        "bushing": {"inner_dia": 12.0, "outer_dia": 20.0, "height": 25.0},
        "hinge": {"leaf_length": 50.0, "leaf_width": 30.0, "leaf_thickness": 4.0, "pin_dia": 5.0},
        "bracket": {"base_length": 50.0, "base_width": 40.0, "wall_height": 40.0, "thickness": 6.0},
        "rivet": {"shank_dia": 5.0, "shank_length": 15.0, "head_dia": 8.0, "head_height": 3.0},
        "table": {"top_length": 120.0, "top_width": 70.0, "top_thickness": 6.0, "leg_height": 72.0},
        "frame": {"length": 100.0, "width": 80.0, "height": 80.0, "thickness": 8.0},
        "helmet": {"outer_dia": 100.0, "wall_thickness": 3.0},
    }
    return defaults.get(obj_type, {"diameter": 50.0, "height": 50.0})

# ----------------------------------------------------------------------
# Extract dimensions from OCR text (with fallback to defaults)
# ----------------------------------------------------------------------
def extract_dimensions(text: str, obj_type: str) -> Dict[str, float]:
    """Extract dimensions using keywords + fallback order, but keep only allowed keys.
       If extraction fails, returns default dimensions for that object."""
    text_lower = text.lower()
    raw_dims = {}

    # Keyword extraction
    for key, pat in KEYWORD_PATTERNS.items():
        val = _extract_value([pat], text_lower)
        if val is not None:
            raw_dims[key] = val

    # Symbol extraction
    for pat, key in SYMBOL_PATTERNS:
        m = re.search(pat, text)
        if m and key not in raw_dims:
            raw_dims[key] = float(m.group(1))

    # Fallback numeric order for missing required keys
    allowed = OBJECT_ALLOWED_KEYS.get(obj_type, [])
    required = OBJECT_REQUIRED_KEYS.get(obj_type, [])
    missing_req = [r for r in required if r not in raw_dims]
    if missing_req:
        # Extract all numbers
        numbers = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", text)]
        # Use a sensible order for this object
        order_map = {
            "bolt": ["shaft_dia", "shaft_length", "head_dia", "head_height", "pitch"],
            "screw": ["shaft_dia", "shaft_length", "head_dia", "head_height", "pitch"],
            "nut": ["inner_dia", "outer_dia", "thickness", "pitch"],
            "washer": ["inner_dia", "outer_dia", "thickness"],
            "gear": ["outer_dia", "thickness", "tooth_count", "hole_dia"],
            "bevel_gear": ["outer_dia", "face_width", "tooth_count", "hole_dia"],
            "helical_gear": ["outer_dia", "face_width", "tooth_count", "helix_angle", "hole_dia"],
            "spring": ["mean_dia", "wire_dia", "free_length", "pitch"],
            "cylinder": ["diameter", "height", "hole_dia"],
            "sphere": ["diameter"],
            "cone": ["diameter", "height"],
            "box": ["length", "width", "height"],
            "plate": ["length", "width", "thickness", "hole_count", "hole_dia"],
            "shaft": ["diameter", "length"],
            "bushing": ["inner_dia", "outer_dia", "height"],
            "pulley": ["outer_dia", "face_width", "bore_dia", "groove_dia", "hub_dia"],
            "hinge": ["leaf_length", "leaf_width", "leaf_thickness", "pin_dia"],
            "bracket": ["base_length", "base_width", "wall_height", "thickness"],
            "rivet": ["shank_dia", "shank_length", "head_dia", "head_height"],
            "table": ["top_length", "top_width", "top_thickness", "leg_height", "leg_width"],
            "frame": ["length", "width", "height", "thickness"],
            "helmet": ["outer_dia", "wall_thickness"],
        }
        order = order_map.get(obj_type, [])
        idx = 0
        for req in missing_req:
            if req in order:
                pos = order.index(req)
                if pos < len(numbers):
                    raw_dims[req] = numbers[pos]
                    idx += 1

    # --- Conflict resolution ---
    if "inner_dia" in raw_dims and "diameter" in raw_dims:
        del raw_dims["diameter"]
    if "outer_dia" in raw_dims and "diameter" in raw_dims:
        del raw_dims["diameter"]

    # --- Filter only allowed keys ---
    filtered = {k: v for k, v in raw_dims.items() if k in allowed and v > 0}

    # If still missing required keys, fill with defaults (fail‑safe)
    if any(r not in filtered for r in required):
        print(f"[WARNING] Missing dimensions for {obj_type}, using fallback defaults.")
        defaults = generate_default_dimensions(obj_type)
        for req in required:
            if req not in filtered:
                filtered[req] = defaults.get(req, 0)
        # Also add optional allowed defaults if missing (e.g., hole_dia for gear)
        for key, val in defaults.items():
            if key in allowed and key not in filtered:
                filtered[key] = val

    # Ensure all values > 0 (clip negatives)
    filtered = {k: max(v, 0.1) for k, v in filtered.items()}

    return filtered

# ----------------------------------------------------------------------
# Validation: no longer raises error, just logs warning
# ----------------------------------------------------------------------
def validate_dimensions(obj_type: str, dims: Dict[str, float]) -> Tuple[bool, List[str]]:
    required = OBJECT_REQUIRED_KEYS.get(obj_type, [])
    missing = [r for r in required if r not in dims or dims[r] <= 0]
    if missing:
        print(f"[WARNING] Validation: missing required dimensions {missing} for {obj_type} – will use defaults.")
    return len(missing) == 0, missing

# ----------------------------------------------------------------------
# Build clean prompt summary (only valid keys)
# ----------------------------------------------------------------------
def build_prompt_summary(obj_type: str, dims: Dict[str, float]) -> str:
    """Generate summary with only the filtered, valid dimensions."""
    lines = ["All required parameters collected.", "Summary:"]
    lines.append(f"- Object: {obj_type.title()}")

    label_map = {
        "shaft_dia": "Shaft Dia", "shaft_length": "Length",
        "head_dia": "Head Dia", "head_height": "Head Height",
        "inner_dia": "Inner Dia", "outer_dia": "Outer Dia",
        "thickness": "Thickness", "pitch": "Thread Pitch",
        "tooth_count": "Tooth Count", "face_width": "Face Width",
        "helix_angle": "Helix Angle", "hole_dia": "Hole Dia",
        "mean_dia": "Mean Dia", "wire_dia": "Wire Dia",
        "free_length": "Free Length", "diameter": "Diameter",
        "height": "Height", "length": "Length", "width": "Width",
        "base_length": "Base Length", "base_width": "Base Width",
        "wall_height": "Wall Height", "leaf_length": "Leaf Length",
        "leaf_width": "Leaf Width", "leaf_thickness": "Leaf Thickness",
        "pin_dia": "Pin Dia", "bore_dia": "Bore Dia",
        "top_length": "Top Length", "top_width": "Top Width",
        "top_thickness": "Top Thickness", "leg_height": "Leg Height",
        "leg_width": "Leg Width", "shank_dia": "Shank Dia",
        "shank_length": "Shank Length", "head_dia": "Head Dia",
        "head_height": "Head Height", "knuckle_dia": "Knuckle Dia",
        "knuckle_count": "Knuckle Count", "holes_per_leaf": "Holes per Leaf",
        "groove_dia": "Groove Dia", "hub_dia": "Hub Dia",
        "flange_thickness": "Flange Thickness", "wall_thickness": "Wall Thickness",
        "tooth_depth": "Tooth Depth"
    }

    dim_parts = []
    for key, val in dims.items():
        label = label_map.get(key, key.replace("_", " ").title())
        if "count" in key or key in ["tooth_count", "knuckle_count", "holes_per_leaf"]:
            dim_parts.append(f"{label}={int(val)}")
        else:
            dim_parts.append(f"{label}={val:.3g}mm" if isinstance(val, float) else f"{label}={val}mm")
    lines.append(f"- Dimensions: {', '.join(dim_parts)}")
    lines.append("- Shape: (auto-detected)")
    lines.append("- Features: (auto-detected)")
    lines.append("Now the design process will be started.")
    return "\n".join(lines)

# ----------------------------------------------------------------------
# OCR entry point (unchanged)
# ----------------------------------------------------------------------
def extract_text_from_image(image_path: str) -> str:
    if OCR_BACKEND is None:
        raise RuntimeError("No OCR backend (install easyocr or pytesseract).")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if OCR_BACKEND == "easyocr":
        result = OCR_READER.readtext(thresh, detail=0, paragraph=True)
        text = " ".join(result)
    else:
        text = pytesseract.image_to_string(thresh)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------------------------------------------------
# Main public function (fail‑safe: never raises exception for missing dims)
# ----------------------------------------------------------------------
def build_summary_from_image(image_path: str) -> Tuple[str, str, Dict[str, Any]]:
    raw_text = extract_text_from_image(image_path)

    from Modelling import detect_object
    obj_type = detect_object(raw_text)
    if obj_type == "unknown":
        # Instead of raising, fallback to a generic object (pulley) with defaults
        print("[WARNING] Object detection returned 'unknown'. Falling back to pulley with defaults.")
        obj_type = "pulley"

    # Extract dimensions (already includes fallback to defaults)
    dims = extract_dimensions(raw_text, obj_type)

    # Validation is just for logging; we already have valid dims
    validate_dimensions(obj_type, dims)

    summary = build_prompt_summary(obj_type, dims)
    return summary, obj_type, dims

# ----------------------------------------------------------------------
# Optional: user refinement (replace values in summary)
# ----------------------------------------------------------------------
def refine_summary_with_user_input(summary: str, edits: Dict[str, str]) -> str:
    for key, new_val in edits.items():
        pattern = rf"{key}=(\d+(?:\.\d+)?)mm"
        summary = re.sub(pattern, f"{key}={new_val}mm", summary, flags=re.IGNORECASE)
    return summary