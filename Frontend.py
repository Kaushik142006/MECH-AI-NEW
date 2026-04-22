"""Gradio UI and chat handler."""
import traceback

import gradio as gr

from Modelling import (
    COLLECTOR_MAX_TOKENS,
    INTRO_SYSTEM,
    MAX_HISTORY_MESSAGES,
    build_direct_summary,
    client,
    detect_object,
    parse_dims,
    prepare_viewer_model,
    run_pipeline,
)
from Sim import find_failure, run_simulation, sim_state

# NEW: Image‑to‑Model parser
from image_parser import build_summary_from_image, refine_summary_with_user_input


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
            # Update sim state with the detected object for simulation tab
            sim_state.last_detected_object = detected_obj if detected_obj != "unknown" else sim_state.last_detected_object
            sim_state.last_dims = parse_dims(pipeline_input)
            bot_reply += "\n\n✅ **3D model generated!** Check the viewer →" if stl_file \
                         else "\n\n⚠️ Generation failed. Please try again."
        except Exception as e:
            print(traceback.format_exc())
            bot_reply += f"\n\n⚠️ Error: {str(e)}"

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
            detected = detect_object(bot_reply)
            sim_state.last_detected_object = detected if detected != "unknown" else sim_state.last_detected_object
            sim_state.last_dims = parse_dims(bot_reply)
            bot_reply += "\n\n✅ **3D model generated!** Check the viewer →" if stl_file \
                         else "\n\n⚠️ Generation failed. Please try again."
        except Exception as e:
            print(traceback.format_exc())
            bot_reply += f"\n\n⚠️ Error: {str(e)}"

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": bot_reply})
    return "", history, viewer_file


CUSTOM_CSS = """
.gradio-container{background:#0f1319 !important;font-family:'Segoe UI',system-ui,sans-serif;color:#e0e8f0;}
#header{text-align:center;padding:20px 0;border-bottom:2px solid #22d3a0;margin-bottom:10px;background:linear-gradient(90deg,transparent,rgba(34,211,160,0.08),transparent);}
#header h1{color:#22d3a0;font-size:2.4em;font-weight:700;letter-spacing:3px;margin:0;text-shadow:0 0 30px rgba(34,211,160,0.3);}
#header p{color:#6b7b8d;font-size:0.85em;margin:5px 0 0 0;font-family:monospace;letter-spacing:3px;text-transform:uppercase;}
.chatbot{background:#111820 !important;border:1px solid #1e2a3a !important;border-radius:12px !important;min-height:420px !important;}
.panel{background:#111820;border-radius:12px;border:1px solid #1e2a3a;padding:16px;}
#user-input textarea{background:#1a2332 !important;border:1px solid #2a3a4a !important;border-radius:12px !important;color:#e0e8f0 !important;font-size:15px !important;padding:12px 16px !important;}
#user-input textarea:focus{border-color:#22d3a0 !important;box-shadow:0 0 15px rgba(34,211,160,0.15) !important;}
#send-btn{background:linear-gradient(135deg,#22d3a0,#1aaa80) !important;border:none !important;border-radius:12px !important;color:#0f1319 !important;font-weight:600 !important;}
#send-btn:hover{box-shadow:0 0 20px rgba(34,211,160,0.4) !important;transform:translateY(-1px) !important;}
#model-viewer{border:1px solid #1e2a3a !important;border-radius:12px !important;background:#111820 !important;min-height:400px !important;}
.failure-alert{background:linear-gradient(135deg,rgba(239,68,68,0.2),rgba(239,68,68,0.05));border:2px solid #ef4444;border-radius:12px;padding:16px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{box-shadow:0 0 20px rgba(239,68,68,0.3);}50%{box-shadow:0 0 40px rgba(239,68,68,0.6);}}
.gradient-bar{width:180px;height:16px;background:linear-gradient(90deg,#0088ff,#00ddcc,#00ff66,#ffcc00,#ff2222);border-radius:8px;border:1px solid #2a3a4a;}
label,#label-text{color:#8899aa !important;font-family:monospace !important;text-transform:uppercase !important;letter-spacing:1px !important;font-size:0.8em !important;}
#footer{text-align:center;padding:12px 0;border-top:1px solid #1e2a3a;margin-top:10px;}
#footer p{color:#3a4a5a;font-family:monospace;font-size:0.72em;letter-spacing:2px;}
#landing-screen{min-height:86vh;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;background:radial-gradient(circle at 20% 20%,rgba(34,211,160,0.12),transparent 35%),radial-gradient(circle at 80% 0%,rgba(59,130,246,0.16),transparent 32%),#0f1319;border:1px solid #1e2a3a;border-radius:18px;margin-top:12px;}
.hero-card{position:relative;z-index:2;max-width:860px;text-align:center;padding:46px 34px;border:1px solid #2a3a4a;border-radius:20px;background:linear-gradient(180deg,rgba(17,24,32,0.9),rgba(15,19,25,0.88));box-shadow:0 20px 80px rgba(0,0,0,0.45);}
.hero-title{font-size:3.2rem;font-weight:800;letter-spacing:2px;line-height:1.05;margin:0;color:#e0f7ef;text-shadow:0 0 25px rgba(34,211,160,0.25);}
.hero-sub{margin:16px auto 0 auto;max-width:760px;color:#8ea0b3;font-size:1.06rem;line-height:1.6;min-height:56px;}
.hero-chip{display:inline-block;margin-top:18px;padding:7px 14px;border:1px solid #2c425f;border-radius:999px;color:#7dc9ff;background:rgba(17,34,50,0.55);font-family:monospace;letter-spacing:1px;}
#start-building-btn{margin-top:28px !important;padding:14px 28px !important;border-radius:14px !important;border:1px solid rgba(34,211,160,0.7) !important;background:linear-gradient(135deg,#22d3a0,#3b82f6) !important;color:#091018 !important;font-weight:800 !important;letter-spacing:1px !important;box-shadow:0 8px 30px rgba(34,211,160,0.28) !important;transition:all .25s ease !important;}
#start-building-btn:hover{transform:translateY(-2px) scale(1.02);box-shadow:0 14px 38px rgba(59,130,246,0.35) !important;}
#start-building-btn:active{transform:scale(0.98);}
.hero-orb{position:absolute;border-radius:999px;filter:blur(2px);opacity:.4;animation:floaty 9s ease-in-out infinite;}
.orb-a{width:260px;height:260px;left:-60px;top:-50px;background:radial-gradient(circle,#22d3a0,transparent 65%);}
.orb-b{width:300px;height:300px;right:-90px;bottom:-80px;background:radial-gradient(circle,#3b82f6,transparent 65%);animation-delay:1.5s;}
.orb-c{width:180px;height:180px;right:16%;top:8%;background:radial-gradient(circle,#06b6d4,transparent 65%);animation-delay:3.2s;}
.component-stream{position:absolute;z-index:1;inset:0;pointer-events:none;overflow:hidden;}
.component-pill{position:absolute;padding:6px 10px;border:1px solid #2a3a4a;border-radius:999px;color:#89a3b8;background:rgba(12,18,26,0.65);font-size:.76rem;font-family:monospace;animation:drift 14s linear infinite;}
.p1{left:8%;top:18%;animation-delay:0s}.p2{left:18%;top:72%;animation-delay:2.5s}.p3{left:34%;top:28%;animation-delay:1.2s}.p4{left:52%;top:76%;animation-delay:4.2s}.p5{left:67%;top:24%;animation-delay:3.5s}.p6{left:81%;top:66%;animation-delay:5.5s}.p7{left:73%;top:12%;animation-delay:6.2s}.p8{left:12%;top:48%;animation-delay:7.1s}
@keyframes drift{0%{transform:translateY(0px) translateX(0px);}50%{transform:translateY(-14px) translateX(8px);}100%{transform:translateY(0px) translateX(0px);}}
@keyframes floaty{0%,100%{transform:translateY(0px) translateX(0px);}50%{transform:translateY(-14px) translateX(8px);}}
"""


def create_ui():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="MECH-AI — Modelling + Simulation",
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

        with gr.Group(visible=False, elem_id="landing-screen") as landing_view:
            gr.HTML("""
            <div class="hero-orb orb-a"></div>
            <div class="hero-orb orb-b"></div>
            <div class="hero-orb orb-c"></div>
            <div class="component-stream">
                <span class="component-pill p1">BOLT</span>
                <span class="component-pill p2">GEAR</span>
                <span class="component-pill p3">BEARING</span>
                <span class="component-pill p4">SCREW</span>
                <span class="component-pill p5">SPRING</span>
                <span class="component-pill p6">PULLEY</span>
                <span class="component-pill p7">BRACKET</span>
                <span class="component-pill p8">NUT</span>
            </div>
            <div class="hero-card">
                <h1 class="hero-title">MECH-AI</h1>
                <p class="hero-sub"><span id="typed-hero-line"></span></p>
                <div class="hero-chip">Ollama + build123d + ML Simulation</div>
            </div>
            <script>
            (function () {
                const lines = [
                    "Design mechanical components with AI precision.",
                    "Simulate stress and safety in interactive 3D.",
                    "Build faster with modelling + simulation in one flow."
                ];
                const el = document.getElementById("typed-hero-line");
                if (!el) return;
                let li = 0, ci = 0, deleting = false;
                function tick() {
                    const text = lines[li];
                    if (!deleting) {
                        ci = Math.min(ci + 1, text.length);
                    } else {
                        ci = Math.max(ci - 1, 0);
                    }
                    el.textContent = text.slice(0, ci);
                    let delay = deleting ? 35 : 55;
                    if (!deleting && ci === text.length) { deleting = true; delay = 1300; }
                    else if (deleting && ci === 0) { deleting = false; li = (li + 1) % lines.length; delay = 300; }
                    setTimeout(tick, delay);
                }
                tick();
            })();
            </script>
            """)
            start_btn = gr.Button("Start Building", elem_id="start-building-btn", variant="primary")

        with gr.Group(visible=True) as main_view:
            # ── Header ──────────────────────────────────────────────────────
            gr.HTML("""
            <div id="header">
                <h1>MEC<span style="color:#3b82f6">H</span>-AI</h1>
                <p>AI-Powered Mechanical Design &amp; Simulation Platform</p>
            </div>
            """)

            # ── Tab navigation ───────────────────────────────────────────────
            with gr.Tabs():

                # ════════════════════════════════════════════════════════════
                # TAB 1 — MODELLING (text chat)
                # ════════════════════════════════════════════════════════════
                with gr.TabItem("🔧 3D Modelling"):
                    gr.Markdown("Describe your component in natural language.")

                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                value=[{"role": "assistant",
                                        "content": "Hello! I'm **MECHAI**, your AI-powered mechanical design assistant.\n\nWhat are we designing today?"}],
                                elem_classes=["chatbot"], height=480, show_label=False,
                            )
                            with gr.Row():
                                user_input = gr.Textbox(
                                    placeholder="e.g. 'M6 screw, 30mm long' or 'hex nut inner 10mm outer 20mm thick 8mm'",
                                    show_label=False, elem_id="user-input", scale=5, lines=1, max_lines=3,
                                )
                                send_btn = gr.Button("Send ➤", elem_id="send-btn", scale=1, variant="primary")
                            clear_btn = gr.Button("🔄 New Design", size="sm")

                        with gr.Column(scale=2):
                            gr.Markdown("#### 🎯 3D Model Viewer")
                            model_viewer = gr.Model3D(
                                label="Generated Model", elem_id="model-viewer",
                                height=480, clear_color=[0.059, 0.075, 0.098, 1.0],
                            )
                            gr.Markdown("*Your 3D model will appear here after generation.*")

                    # Events for modelling tab
                    send_btn.click(
                        fn=chat_handler,
                        inputs=[user_input, chatbot],
                        outputs=[user_input, chatbot, model_viewer]
                    )
                    user_input.submit(
                        fn=chat_handler,
                        inputs=[user_input, chatbot],
                        outputs=[user_input, chatbot, model_viewer]
                    )
                    clear_btn.click(
                        fn=lambda: (
                            [{"role": "assistant", "content": "Ready for a new design! What would you like to create?"}],
                            None
                        ),
                        outputs=[chatbot, model_viewer],
                    )

                # ════════════════════════════════════════════════════════════
                # TAB 2 — IMAGE TO MODEL (NEW)
                # ════════════════════════════════════════════════════════════
                with gr.TabItem("📸 Image to Model"):
                    gr.Markdown("Upload an engineering drawing or sketch – the AI will extract dimensions and create a 3D model.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(type="filepath", label="Upload Image", elem_id="image-upload")
                            extract_btn = gr.Button("🔍 Extract Data", variant="primary")
                        with gr.Column(scale=2):
                            extracted_text = gr.Textbox(label="Extracted Text (OCR)", lines=6, interactive=False)
                            detected_obj = gr.Textbox(label="Detected Object", interactive=False)
                            with gr.Group():
                                gr.Markdown("### ✏️ Edit Dimensions (optional)")
                                dims_editor = gr.Textbox(label="Dimensions (key=value, comma separated)", lines=2,
                                                         placeholder="e.g. Diameter=20, Length=30, Tooth Count=18")
                            generate_btn = gr.Button("🚀 Generate 3D Model", variant="primary", visible=False)
                            model_viewer_img = gr.Model3D(label="Generated Model", elem_id="model-viewer", height=400)

                    # State to hold the raw summary
                    raw_summary_state = gr.State("")

                    def on_extract(image_path):
                        if not image_path:
                            return "", "", "", "", None
                        try:
                            summary, obj, dims = build_summary_from_image(image_path)
                            # Prepare display text
                            dims_display = ", ".join([f"{k}={v}" for k, v in dims.items() if k not in ["raw_numbers"]])
                            return summary, obj, dims_display, summary, gr.update(visible=True)
                        except Exception as e:
                            return f"Error: {str(e)}", "unknown", "", "", gr.update(visible=False)

                    extract_btn.click(
                        on_extract,
                        inputs=[image_input],
                        outputs=[extracted_text, detected_obj, dims_editor, raw_summary_state, generate_btn]
                    )

                    def on_generate_with_edits(summary, edits_str):
                        # Parse edits from comma‑separated string
                        edits = {}
                        if edits_str:
                            for part in edits_str.split(','):
                                if '=' in part:
                                    k, v = part.split('=', 1)
                                    edits[k.strip()] = v.strip()
                        final_summary = refine_summary_with_user_input(summary, edits)
                        # Use existing run_pipeline
                        from Modelling import run_pipeline, prepare_viewer_model
                        stl_path = run_pipeline(final_summary)
                        viewer_path = prepare_viewer_model(stl_path) if stl_path else None
                        return viewer_path

                    generate_btn.click(
                        on_generate_with_edits,
                        inputs=[raw_summary_state, dims_editor],
                        outputs=[model_viewer_img]
                    )

                # ════════════════════════════════════════════════════════════
                # TAB 3 — SIMULATION LAB
                # ════════════════════════════════════════════════════════════
                with gr.TabItem("⚡ Simulation Lab"):
                    gr.Markdown("## ⚡ Simulation Lab")
                    gr.Markdown("Real-time FEA with AI-driven optimization. Uses the last modelled object automatically, or adjust parameters manually.")

                    with gr.Row():
                        # ── Controls ─────────────────────────────────────────
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["panel"]):
                                gr.Markdown("### 🎛️ Parameters")

                                sim_component = gr.Dropdown(
                                    choices=[
                                        "bolt", "screw", "nut", "washer", "gear", "bearing", "pulley",
                                        "sprocket", "spring", "shaft", "hinge", "bracket", "rivet",
                                        "bushing", "coupling", "cube", "cuboid", "box", "cylinder",
                                        "sphere", "cone"
                                    ],
                                    value="bolt",
                                    label="Component Type"
                                )
                                sim_temperature = gr.Slider(-50, 400, value=25, step=1, label="Temperature (°C)")

                                gr.Markdown("### ⚡ Load")
                                sim_load = gr.Slider(100, 10000, value=500, step=100, label="Applied Load (N)")

                                with gr.Row():
                                    run_sim_btn  = gr.Button("▶️ Run Sim", variant="primary")
                                    find_fail_btn = gr.Button("💥 Find Failure", variant="stop")

                            with gr.Group(elem_classes=["panel"]):
                                gr.Markdown("### 📊 Metrics")
                                with gr.Row():
                                    sf_metric     = gr.Number(label="Safety Factor", value=2.0)
                                    stress_metric = gr.Number(label="Stress (MPa)", value=0)
                                with gr.Row():
                                    deform_metric = gr.Number(label="Strain", value=0)
                                    mass_metric   = gr.Number(label="Yield Point (MPa)", value=0)

                        # ── Visualizations ────────────────────────────────────
                        with gr.Column(scale=2):
                            with gr.Group(elem_classes=["panel"]):
                                gr.Markdown("### 🌡️ 3D Stress Heatmap")
                                heatmap_plot = gr.Plot()
                                gr.HTML("""
                                <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
                                    <span style="color:#0088ff;font-size:0.8em;">Low Stress</span>
                                    <div class="gradient-bar"></div>
                                    <span style="color:#ff2222;font-size:0.8em;">Yield</span>
                                </div>
                                """)

                            with gr.Group(elem_classes=["panel"]):
                                gr.Markdown("### 📈 Response Curves")
                                with gr.Tabs():
                                    with gr.TabItem("Safety Factor"):
                                        sf_curve = gr.Plot()
                                    with gr.TabItem("Strain"):
                                        deform_curve = gr.Plot()

                        # ── AI Analysis ───────────────────────────────────────
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["panel"]):
                                gr.Markdown("### 🤖 AI Analysis")
                                compliance_gauge = gr.Plot()

                                with gr.Group(visible=False) as failure_panel:
                                    gr.HTML("""
                                    <div class="failure-alert">
                                        <h3 style="color:#ef4444;margin-top:0;">⚠️ FAILURE PREDICTED</h3>
                                        <p>Reduce load or upgrade material.</p>
                                    </div>
                                    """)

                                ai_report = gr.Markdown()

                    # ── Simulation Events ────────────────────────────────────
                    sim_outputs = [
                        sf_metric, stress_metric, deform_metric, mass_metric,
                        sf_curve, deform_curve, heatmap_plot, compliance_gauge,
                        ai_report, failure_panel
                    ]

                    run_sim_btn.click(
                        fn=run_simulation,
                        inputs=[sim_component, sim_temperature, sim_load],
                        outputs=sim_outputs
                    )

                    find_fail_btn.click(
                        fn=find_failure,
                        inputs=[sim_component, sim_temperature, sim_load],
                        outputs=[sim_load, sf_metric, stress_metric, ai_report, failure_panel, sf_curve]
                    )

                    # Live slider updates
                    for component in [sim_component, sim_temperature, sim_load]:
                        component.change(
                            fn=run_simulation,
                            inputs=[sim_component, sim_temperature, sim_load],
                            outputs=sim_outputs
                        )

            gr.HTML('<div id="footer"><p>MECH-AI ENGINE &bull; Ollama + build123d + FEA Simulation</p></div>')

        # Landing flow disabled in app; use external index.html as entry page.

    return app