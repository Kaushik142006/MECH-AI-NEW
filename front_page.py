"""MECH-AI Landing Page â€” clean, hand-crafted Gradio entry point."""

import gradio as gr

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.gradio-container {
    background: #0a0b0d !important;
    color: #c9d1d9;
    font-family: 'DM Mono', monospace;
    overflow-x: hidden;
}

footer, .svelte-1rjryqp { display: none !important; }
.contain { padding: 0 !important; }

#lp {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 22px 48px;
    border-bottom: 1px solid #1a1f26;
}
.topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: 4px;
    text-transform: uppercase;
}
.topbar-meta {
    font-size: 11px;
    color: #3d4f5c;
    letter-spacing: 2px;
}
.topbar-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00c87a;
    margin-right: 8px;
    vertical-align: middle;
    animation: blink 2.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

.hero {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 80px 48px 60px;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    left: 48px; right: 48px; top: 0;
    height: 1px;
    background: linear-gradient(90deg, #00c87a22, #00c87a55 40%, transparent);
}

.hero-eyebrow {
    font-size: 11px;
    letter-spacing: 3px;
    color: #00c87a;
    text-transform: uppercase;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-eyebrow::before {
    content: '';
    display: block;
    width: 32px; height: 1px;
    background: #00c87a;
    flex-shrink: 0;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(72px, 13vw, 140px);
    font-weight: 800;
    line-height: 0.88;
    letter-spacing: -2px;
    color: #e6edf3;
    margin-bottom: 24px;
}

.hero-tagline {
    font-size: 12px;
    letter-spacing: 6px;
    color: #3d4f5c;
    text-transform: uppercase;
    margin-bottom: 52px;
}

.dvs {
    display: flex;
    align-items: center;
    margin-bottom: 56px;
    flex-wrap: wrap;
    gap: 0;
}
.dvs-word {
    font-family: 'Syne', sans-serif;
    font-size: clamp(13px, 2vw, 17px);
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e6edf3;
    padding: 10px 24px;
    border: 1px solid #1e2730;
}
.dvs-word.active {
    color: #00c87a;
    border-color: #00c87a33;
    background: #00c87a08;
}
.dvs-sep {
    font-size: 11px;
    color: #2a3540;
    padding: 0 6px;
}

.launch-row {
    padding: 0 48px 0 !important;
    margin: 0 !important;
    gap: 0 !important;
}

#launch-btn {
    display: inline-flex !important;
    align-items: center !important;
    height: 52px !important;
    padding: 0 40px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #0a0b0d !important;
    background: #00c87a !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    transition: background .18s, transform .12s !important;
    cursor: pointer !important;
    max-width: 280px !important;
}
#launch-btn:hover {
    background: #00f093 !important;
    transform: translateX(4px) !important;
}
#launch-btn:active { transform: scale(0.97) !important; }

.ticker-outer {
    border-top: 1px solid #1a1f26;
    border-bottom: 1px solid #1a1f26;
    overflow: hidden;
    padding: 11px 0;
    background: #080a0c;
    white-space: nowrap;
}
.ticker-track {
    display: inline-flex;
    animation: scroll 50s linear infinite;
}
.ticker-item {
    display: inline-flex;
    align-items: center;
    padding: 0 32px;
    font-size: 10.5px;
    letter-spacing: 1.5px;
    color: #2e3f4c;
    text-transform: uppercase;
    white-space: nowrap;
}
.ticker-item span.hi { color: #00c87a; }
.ticker-item::before { content: '//'; margin-right: 14px; color: #1a2530; }
@keyframes scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }

.stat-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid #1a1f26;
}
.stat-cell {
    padding: 28px 32px;
    border-right: 1px solid #1a1f26;
}
.stat-cell:last-child { border-right: none; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #e6edf3;
    line-height: 1;
    margin-bottom: 6px;
}
.stat-label {
    font-size: 10px;
    letter-spacing: 2px;
    color: #3d4f5c;
    text-transform: uppercase;
}

.lp-footer {
    padding: 18px 48px;
    border-top: 1px solid #1a1f26;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #080a0c;
}
.footer-l {
    font-size: 10px;
    color: #2a3540;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.footer-r { font-size: 10px; color: #1e2730; letter-spacing: 1px; }

#main-col { display: block; }
"""


LANDING_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Mono"), "monospace"],
)
TICKER_MSGS = [
    ("Structural load analysis", "complete"),
    ("RandomForest models", "7 loaded"),
    ("Safety factor", "21.66"),
    ("FEA solver", "active"),
    ("Von Mises stress", "128 MPa"),
    ("STL export", "ready"),
    ("Ollama LLM", "online"),
    ("ASME compliance", "passed"),
    ("Yield point", "340 MPa"),
    ("build123d", "initialised"),
    ("ISO 9001", "ready"),
    ("Strain tensor", "computed"),
    ("Material matrix", "loaded"),
    ("GNN inference", "complete"),
    ("Heatmap renderer", "active"),
]


def _ticker_html():
    items = "".join(
        f'<span class="ticker-item">{label} &nbsp;<span class="hi">{val}</span></span>'
        for label, val in TICKER_MSGS * 2
    )
    return f'<div class="ticker-outer"><div class="ticker-track">{items}</div></div>'


LANDING_HTML = f"""
<div id="lp">
  <div class="topbar">
    <span class="topbar-logo">Mech&nbsp;AI</span>
    <span class="topbar-meta"><span class="topbar-dot"></span>System online</span>
  </div>

  <div class="hero">
    <div class="hero-eyebrow">AI mechanical engineering platform</div>
    <h1 class="hero-title">MECH<br>AI</h1>
    <p class="hero-tagline">Integrated modelling &amp; simulation â€” v1.0</p>
    <div class="dvs">
      <span class="dvs-word active">Design</span>
      <span class="dvs-sep">â€”</span>
      <span class="dvs-word">Validate</span>
      <span class="dvs-sep">â€”</span>
      <span class="dvs-word">Simulate</span>
    </div>
  </div>

  {_ticker_html()}

  <div class="stat-strip">
    <div class="stat-cell"><div class="stat-num">7</div><div class="stat-label">ML models</div></div>
    <div class="stat-cell"><div class="stat-num">21+</div><div class="stat-label">Components</div></div>
    <div class="stat-cell"><div class="stat-num">100%</div><div class="stat-label">Local LLM</div></div>
    <div class="stat-cell"><div class="stat-num">âˆž</div><div class="stat-label">Iterations</div></div>
  </div>

  <div class="lp-footer">
    <span class="footer-l">Ollama &nbsp;Â·&nbsp; build123d &nbsp;Â·&nbsp; scikit-learn &nbsp;Â·&nbsp; Gradio</span>
    <span class="footer-r">Â© 2025 Mech-AI</span>
  </div>
</div>
"""


def create_landing(platform_url: str = "http://localhost:7861"):
    launch_html = f"""
    <a id="launch-btn" href="{platform_url}" target="_self" rel="noopener">Launch platform  -&gt;</a>
    """
    with gr.Blocks(title="MECH-AI", fill_width=True) as app:
        with gr.Column(elem_id="landing-col", visible=True):
            gr.HTML(LANDING_HTML)
            with gr.Row(elem_classes=["launch-row"]):
                gr.HTML(launch_html)
    return app


if __name__ == "__main__":
    print("MECH-AI  â†’  http://localhost:7860")
    create_landing().launch(inbrowser=True, debug=True, css=CSS, theme=LANDING_THEME)



