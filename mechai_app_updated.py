import gradio as gr
import os
import re
import traceback
import tempfile
import subprocess
import sys
from openai import OpenAI

# ── Ollama client ────────────────────────────────────────────────────────
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
STL_PATH = os.path.join(MODEL_DIR, "model.stl").replace("\\", "/")


# ── build123d Code Generation Helpers ────────────────────────────────────

def make_threaded_hex_nut_code(inner_dia: float, outer_dia: float, thickness: float, thread_pitch: float, stl_path: str) -> str:
    return (
        "from build123d import *\n"
        f"inner_dia = {inner_dia}\n"
        f"outer_dia = {outer_dia}\n"
        f"thickness = {thickness}\n"
        f"thread_pitch = {thread_pitch}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        RegularPolygon(radius=outer_dia/2, side_count=6)\n"
        "    with b.extrude(amount=thickness):\n"
        "        # Create a tapped hole for internal threading\n"
        "        TappedHole(radius=inner_dia/2, depth=thickness, pitch=thread_pitch, simple=False)\n"
        f"export_stl(b.part, \'{stl_path}\')\n"
    )

def make_threaded_circle_nut_code(inner_dia: float, outer_dia: float, thickness: float, thread_pitch: float, stl_path: str) -> str:
    return (
        "from build123d import *\n"
        f"inner_dia = {inner_dia}\n"
        f"outer_dia = {outer_dia}\n"
        f"thickness = {thickness}\n"
        f"thread_pitch = {thread_pitch}\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Circle(radius=outer_dia/2)\n"
        "    with b.extrude(amount=thickness):\n"
        "        # Create a tapped hole for internal threading\n"
        "        TappedHole(radius=inner_dia/2, depth=thickness, pitch=thread_pitch, simple=False)\n"
        f"export_stl(b.part, \'{stl_path}\')\n"
    )

def make_threaded_bolt_code(major_dia: float, length: float, head_dia: float, head_thickness: float, thread_pitch: float, stl_path: str) -> str:
    return (
        "from build123d import *\n"
        f"major_dia = {major_dia}\n"
        f"length = {length}\n"
        f"head_dia = {head_dia}\n"
        f"head_thickness = {head_thickness}\n"
        f"thread_pitch = {thread_pitch}\n"
        "with BuildPart() as b:\n"
        "    # Bolt head\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Circle(radius=head_dia/2)\n"
        "    b.extrude(amount=head_thickness)\n"
        "    # Bolt shaft with external thread\n"
        "    with BuildSketch(Plane.XY).workplane(offset=head_thickness):\n"
        "        Circle(radius=major_dia/2)\n"
        "    b.extrude(amount=length, taper=0)\n"
        "    # Apply external thread to the shaft\n"
        "    with b.add(b.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]): # Select top face of shaft\n"
        "        ScrewThread(major_radius=major_dia/2, pitch=thread_pitch, length=length, external=True)\n"
        f"export_stl(b.part, \'{stl_path}\')\n"
    )

def make_box_code(length: float, width: float, height: float, stl_path: str) -> str:
    return (
        "from build123d import *\n"
        f"with BuildPart() as b:\n"
        f"    Box({length}, {width}, {height})\n"
        f"export_stl(b.part, \'{stl_path}\')\n"
    )

def make_cylinder_code(radius: float, height: float, stl_path: str) -> str:
    return (
        "from build123d import *\n"
        f"with BuildPart() as b:\n"
        f"    Cylinder(radius={radius}, height={height})\n"
        f"export_stl(b.part, \'{stl_path}\')\n"
    )

def make_helmet_code(outer_radius: float, thickness: float, height: float, stl_path: str) -> str:
    return (
        "from build123d import *\n"
        f"outer_radius = {outer_radius}\n"
        f"thickness = {thickness}\n"
        f"height = {height}\n"
        "with BuildPart() as b:\n"
        "    # Outer shell of the helmet (half-sphere/ellipsoid like shape)\n"
        "    with BuildSketch(Plane.XZ) as sk_outer:\n"
        "        Bezier(\n"
        "            (0, 0), (outer_radius * 1.2, height * 0.5), (0, height),\n"
        "            tangents=((1, 0), (0, 1), (-1, 0)),\n"
        "            tangent_scalars=(1, 1, 1)\n"
        "        )\n"
        "        mirror(sk_outer.vertices()[-1], sk_outer.vertices()[0])\n"
        "    b.revolve(axis=Axis.Z)\n"
        "    # Inner shell to create hollow interior\n"
        "    inner_radius = outer_radius - thickness\n"
        "    inner_height = height - thickness # Adjust inner height slightly\n"
        "    with BuildSketch(Plane.XZ) as sk_inner:\n"
        "        Bezier(\n"
        "            (0, 0), (inner_radius * 1.2, inner_height * 0.5), (0, inner_height),\n"
        "            tangents=((1, 0), (0, 1), (-1, 0)),\n"
        "            tangent_scalars=(1, 1, 1)\n"
        "        )\n"
        "        mirror(sk_inner.vertices()[-1], sk_inner.vertices()[0])\n"
        "    b.revolve(axis=Axis.Z, mode=Mode.SUBTRACT)\n"
        "    # Add a small opening at the bottom (optional, for realism)\n"
        "    with BuildSketch(Plane.XY).workplane(offset=-1) as sk_cut:\n"
        "        Circle(outer_radius * 0.8)\n"
        "    b.extrude(amount=2, mode=Mode.SUBTRACT)\n"
        f"export_stl(b.part, \'{stl_path}\')\n"
    )


# ── Agent 1: Parameter Collector ────────────────────────────────────────
INTRO_SYSTEM = """You are the MECHAI Parameter Collector. Your sole purpose is to gather geometric dimensions for 3D modeling.

CRITICAL BEHAVIOR:
1. MINIMALISM: If the user provides dimensions, DO NOT ask about material, safety factors, or usage.
2. THE "SHAPE" RULE: You may ask ONLY ONE clarifying question about profile shape.
3. NO REPETITION: Once a value is given, never ask for it again.
4. TRIGGER: As soon as you have all dimensions and shape, stop asking.

REQUIRED PARAMETERS:
- Fasteners (Nuts): Inner Dia, Outer Dia, Thickness, Shape, Thread Pitch.
- Fasteners (Bolts): Major Dia, Length, Head Dia, Head Thickness, Thread Pitch.
- Basic Shapes (Boxes): Length, Width, Height.
- Cylinders: Diameter, Height.
- Complex (Helmets): Outer Radius, Thickness, Height.

RESPONSE PROTOCOL:
- Greeting: "Hello! What are we designing today?"
- Acknowledgment: "Ok, noted."
- Completion — respond EXACTLY in this format:
  "All required parameters collected.
  Summary:
  - Object: [Name]
  - Dimensions: [List values, e.g., Inner Dia=Xmm, Outer Dia=Ymm, Thickness=Zmm, Pitch=Pmm]
  - Shape: [Value]
  Now the design process will be started."

STRICT PROHIBITIONS:
- NEVER ask about Material Type.
- NEVER ask about Safety Factors.
- NEVER explain how CAD works.
- NEVER generate code.
- Do NOT add extra notes or bullet points beyond the summary format."""


# ── Agent 2: Prompt Architect ────────────────────────────────────────────
def prompt_agent(summary: str) -> str:
    system = (
        "You are the MECHAI Prompt Architect.\n"
        "Convert the dimension summary into a precise technical prompt for a CAD Coder.\n\n"
        "Rules:\n"
        "1. Library: build123d\n"
        "2. Export via standalone function: export_stl(b.part, path)\n"
        "3. Use context managers for BuildPart and BuildSketch.\n"
        "4. For nuts, use TappedHole for internal threading.\n"
        "5. For bolts, use ScrewThread for external threading.\n"
        "6. For complex shapes like helmets, use advanced features like Bezier curves, revolve, and Mode.SUBTRACT for hollowing.\n"
        "7. Define all dimensions as float variables.\n\n"
        "Output ONLY the technical prompt. No conversation."
    )
    res = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Summary: {summary}"},
        ],
    )
    return res.choices[0].message.content


# ── Agent 3: Coder ───────────────────────────────────────────────────────
def coder_agent(blueprint: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")

    system = (
        "You are the MECHAI CAD Coder. Write a complete Python script using build123d.\n\n"
        "USE THIS EXACT STRUCTURE — do not deviate. Adapt the structure based on the object type (nut, bolt, helmet, etc.).\n\n"
        "Example for a threaded hex nut:\n"
        "from build123d import *\n"
        "inner_dia = <value>\n"
        "outer_dia = <value>\n"
        "thickness = <value>\n"
        "thread_pitch = <value>\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        RegularPolygon(radius=outer_dia/2, side_count=6)\n"
        "    with b.extrude(amount=thickness):\n"
        "        TappedHole(radius=inner_dia/2, depth=thickness, pitch=thread_pitch, simple=False)\n"
        f"export_stl(b.part, \'{safe_path}\')\n\n"
        "Example for a threaded bolt:\n"
        "from build123d import *\n"
        "major_dia = <value>\n"
        "length = <value>\n"
        "head_dia = <value>\n"
        "head_thickness = <value>\n"
        "thread_pitch = <value>\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XY):\n"
        "        Circle(radius=head_dia/2)\n"
        "    b.extrude(amount=head_thickness)\n"
        "    with BuildSketch(Plane.XY).workplane(offset=head_thickness):\n"
        "        Circle(radius=major_dia/2)\n"
        "    b.extrude(amount=length, taper=0)\n"
        "    with b.add(b.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]):\n"
        "        ScrewThread(major_radius=major_dia/2, pitch=thread_pitch, length=length, external=True)\n"
        f"export_stl(b.part, \'{safe_path}\')\n\n"
        "Example for a helmet:\n"
        "from build123d import *\n"
        "outer_radius = <value>\n"
        "thickness = <value>\n"
        "height = <value>\n"
        "with BuildPart() as b:\n"
        "    with BuildSketch(Plane.XZ) as sk_outer:\n"
        "        Bezier(\n"
        "            (0, 0), (outer_radius * 1.2, height * 0.5), (0, height),\n"
        "            tangents=((1, 0), (0, 1), (-1, 0)),\n"
        "            tangent_scalars=(1, 1, 1)\n"
        "        )\n"
        "        mirror(sk_outer.vertices()[-1], sk_outer.vertices()[0])\n"
        "    b.revolve(axis=Axis.Z)\n"
        "    inner_radius = outer_radius - thickness\n"
        "    inner_height = height - thickness\n"
        "    with BuildSketch(Plane.XZ) as sk_inner:\n"
        "        Bezier(\n"
        "            (0, 0), (inner_radius * 1.2, inner_height * 0.5), (0, inner_height),\n"
        "            tangents=((1, 0), (0, 1), (-1, 0)),\n"
        "            tangent_scalars=(1, 1, 1)\n"
        "        )\n"
        "        mirror(sk_inner.vertices()[-1], sk_inner.vertices()[0])\n"
        "    b.revolve(axis=Axis.Z, mode=Mode.SUBTRACT)\n"
        "    with BuildSketch(Plane.XY).workplane(offset=-1) as sk_cut:\n"
        "        Circle(outer_radius * 0.8)\n"
        "    b.extrude(amount=2, mode=Mode.SUBTRACT)\n"
        f"export_stl(b.part, \'{safe_path}\')\n\n"
        "CRITICAL RULES:\n"
        "1. Export MUST be: export_stl(b.part, 'path')  — NOT b.part.export_stl()\n"
        "2. Ensure all build123d operations are correctly nested within context managers (BuildPart, BuildSketch, extrude, etc.).\n"
        "3. Output raw Python only — no markdown, no backticks, no comments."
    )

    res = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": blueprint},
        ],
        temperature=0,
    )
    raw = res.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


# ── Fix LLM code: patch export line to guaranteed working form ────────────
def patch_export(code: str, stl_path: str) -> str:
    """Replace ANY export_stl call with the correct standalone form."""
    safe_path = stl_path.replace("\\", "/")
    lines = code.splitlines()
    out = []
    injected = False
    for line in lines:
        if "export_stl" in line:
            out.append(f"export_stl(b.part, \'{safe_path}\')")
            injected = True
        else:
            out.append(line)
    if not injected:
        out.append(f"export_stl(b.part, \'{safe_path}\')")
    return "\n".join(out)


# ── Run code in subprocess ────────────────────────────────────────────────
def execute_code(code: str, stl_path: str) -> str:
    print("\n[Engine] Executing:\n" + "=" * 40)
    print(code)
    print("=" * 40)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        if os.path.exists(stl_path):
            os.remove(stl_path)

        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=60
        )
        print("[Engine] STDOUT:", proc.stdout)
        if proc.stderr:
            print("[Engine] STDERR:", proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)

        if os.path.exists(stl_path):
            return stl_path
        raise RuntimeError("Script ran but no STL was created.")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ── Fallback hardcoded generator ──────────────────────────────────────────
def parse_dims(summary: str):
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", summary)
    return [float(m) for m in matches if float(m) > 0]


def generate_fallback(summary: str, stl_path: str) -> str:
    safe_path = stl_path.replace("\\", "/")
    dims = parse_dims(summary)
    print(f"[Fallback] Parsed dims: {dims}")

    # Default values for threading if not provided
    default_pitch = 1.0
    
    is_hex_nut = "hex nut" in summary.lower()
    is_circle_nut = "circle nut" in summary.lower() or "round nut" in summary.lower()
    is_bolt = "bolt" in summary.lower() or "screw" in summary.lower()
    is_box = "box" in summary.lower() or "rect" in summary.lower()
    is_cylinder = "cylinder" in summary.lower()
    is_helmet = "helmet" in summary.lower()

    code = ""
    if is_hex_nut and len(dims) >= 4: # inner_dia, outer_dia, thickness, pitch
        inner_dia, outer_dia, thickness, thread_pitch = dims[0], dims[1], dims[2], dims[3]
        code = make_threaded_hex_nut_code(inner_dia, outer_dia, thickness, thread_pitch, safe_path)
    elif is_circle_nut and len(dims) >= 4: # inner_dia, outer_dia, thickness, pitch
        inner_dia, outer_dia, thickness, thread_pitch = dims[0], dims[1], dims[2], dims[3]
        code = make_threaded_circle_nut_code(inner_dia, outer_dia, thickness, thread_pitch, safe_path)
    elif is_bolt and len(dims) >= 5: # major_dia, length, head_dia, head_thickness, pitch
        major_dia, length, head_dia, head_thickness, thread_pitch = dims[0], dims[1], dims[2], dims[3], dims[4]
        code = make_threaded_bolt_code(major_dia, length, head_dia, head_thickness, thread_pitch, safe_path)
    elif is_box and len(dims) >= 3:
        code = make_box_code(dims[0], dims[1], dims[2], safe_path)
    elif is_cylinder and len(dims) >= 2:
        code = make_cylinder_code(dims[0]/2, dims[1], safe_path)
    elif is_helmet and len(dims) >= 3: # outer_radius, thickness, height
        outer_radius, thickness, height = dims[0], dims[1], dims[2]
        code = make_helmet_code(outer_radius, thickness, height, safe_path)
    else:
        # Fallback to a default threaded hex nut if specific object not recognized or insufficient dims
        print("[Fallback] Defaulting to a threaded hex nut due to insufficient or unrecognized parameters.")
        code = make_threaded_hex_nut_code(6.0, 15.0, 10.0, default_pitch, safe_path)

    print("[Fallback] Code:\n", code)
    return execute_code(code, stl_path)


# ── Full pipeline ─────────────────────────────────────────────────────────
def run_pipeline(summary: str):
    print("\n" + "=" * 50 + "\n[MECHAI] Pipeline start\n" + "=" * 50)

    blueprint = prompt_agent(summary)
    print(f"[Agent 2] Blueprint:\n{blueprint[:200]}\n")

    raw_code = coder_agent(blueprint, STL_PATH)
    code = patch_export(raw_code, STL_PATH)
    print(f"[Agent 3] Code:\n{code[:300]}\n")

    try:
        path = execute_code(code, STL_PATH)
        print(f"[Engine] SUCCESS — {path}")
        return path
    except Exception as e:
        print(f"[Engine] LLM code failed: {e}\n→ Trying fallback...")

    try:
        path = generate_fallback(summary, STL_PATH)
        print(f"[Fallback] SUCCESS — {path}")
        return path
    except Exception as e:
        print(f"[Fallback] Failed: {e}\n{traceback.format_exc()}")

    return None


# ── Chat handler ──────────────────────────────────────────────────────────
def chat_handler(user_message, history):
    messages = [{"role": "system", "content": INTRO_SYSTEM}]
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            # Gradio legacy format [user, bot]
            messages.append({"role": "user", "content": msg[0]})
            messages.append({"role": "assistant", "content": msg[1]})
            continue
        else:
            continue
            
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    res = client.chat.completions.create(model="llama3.2", messages=messages)
    bot_reply = res.choices[0].message.content

    stl_file = None
    if "All required parameters collected" in bot_reply:
        try:
            stl_file = run_pipeline(bot_reply)
            bot_reply += "\n\n✅ **3D model generated!** Check the viewer →" if stl_file \
                         else "\n\n⚠️ Generation failed. Please try again."
        except Exception as e:
            print(traceback.format_exc())
            bot_reply += f"\n\n⚠️ Error: {str(e)}"

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": bot_reply})
    return "", history, stl_file


# ── Gradio UI ─────────────────────────────────────────────────────────────
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
#send-btn{background:linear-gradient(135deg, #22d3a0, #1aaa80) !important;border:none !important;border-radius:12px !important;color:#0f1319 !important;font-weight:600 !important;min-width:100px !important;transition:all 0.2s !important;}
#send-btn:hover{box-shadow:0 0 20px rgba(34,211,160,0.4) !important;transform:translateY(-1px) !important;}
#clear-btn{background:transparent !important;border:1px solid #2a3a4a !important;border-radius:12px !important;color:#6b7b8d !important;}
#model-viewer{border:1px solid #1e2a3a !important;border-radius:12px !important;background:#111820 !important;min-height:400px !important;}
.label-text,label{color:#8899aa !important;font-family:monospace !important;text-transform:uppercase !important;letter-spacing:1px !important;font-size:0.8em !important;}
#footer{text-align:center;padding:15px 0;border-top:1px solid #1e2a3a;margin-top:10px;}
#footer p{color:#3a4a5a;font-family:monospace;font-size:0.75em;letter-spacing:2px;}
"""

def create_ui():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="MECHAI — Mechanical Design AI",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Space Grotesk"),
        ).set(
            body_background_fill="#0f1319",
            body_background_fill_dark="#0f1319",
            block_background_fill="#111820",
            block_background_fill_dark="#111820",
            input_background_fill="#1a2332",
            input_background_fill_dark="#1a2332",
            button_primary_background_fill="#22d3a0",
            button_primary_text_color="#0f1319",
        ),
    ) as app:

        gr.HTML("""
            <div id="header">
                <h1>MEC<span style="color:#22d3a0">HAI</span></h1>
                <p>Mechanical Design AI</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[
                        {"role": "assistant", "content": "Hello! I'm **MECHAI**, your AI-powered mechanical design assistant.\n\nWhat are we designing today?"}
                    ],
                    elem_classes=["chatbot"],
                    height=500,
                    show_label=False,
                )

                with gr.Row(elem_classes=["input-row"]):
                    user_input = gr.Textbox(
                        placeholder="Describe your part or provide dimensions...",
                        show_label=False,
                        elem_id="user-input",
                        scale=5,
                        lines=1,
                        max_lines=3,
                    )
                    send_btn = gr.Button(
                        "Send ➤",
                        elem_id="send-btn",
                        scale=1,
                        variant="primary",
                    )

                clear_btn = gr.Button(
                    "🔄 New Design",
                    elem_id="clear-btn",
                    size="sm",
                )

            with gr.Column(scale=2):
                gr.Markdown("#### 🔧 3D Model Viewer")
                model_viewer = gr.Model3D(
                    label="Generated Model",
                    elem_id="model-viewer",
                    height=500,
                    clear_color=[0.059, 0.075, 0.098, 1.0],
                )
                gr.Markdown(
                    "*Your 3D model will appear here after the design is generated.*",
                    elem_classes=["label-text"],
                )

        gr.HTML("""
            <div id="footer">
                <p>MECHAI ENGINE • Ollama + build123d Pipeline</p>
            </div>
        """)

        send_btn.click(
            fn=chat_handler,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot, model_viewer],
        )

        user_input.submit(
            fn=chat_handler,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot, model_viewer],
        )

        clear_btn.click(
            fn=lambda: (
                [], # Clear chatbot history
                None,
            ),
            outputs=[chatbot, model_viewer],
        )

    return app


# ── Launch ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  MECHAI — Mechanical Design AI")
    print("  Powered by Ollama + build123d + Gradio")
    print("=" * 50)
    print(f"  STL output: {STL_PATH}")
    print()

    app = create_ui()
    app.launch(

        inbrowser=True,
    )
