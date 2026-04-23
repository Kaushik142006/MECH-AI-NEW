"""MECH-AI entry: landing page that opens the original platform UI."""
from Modelling import STL_PATH
from Frontend import create_ui
from front_page import CSS as LANDING_CSS, LANDING_THEME, create_landing

PLATFORM_PORT = 7861
LANDING_PORT = 7860

if __name__ == "__main__":
    platform_url = f"http://localhost:{PLATFORM_PORT}"
    print("=" * 60)
    print("  MECH-AI Landing + Original Platform UI")
    print(f"  STL output: {STL_PATH}")
    print(f"  Landing:  http://localhost:{LANDING_PORT}")
    print(f"  Platform: {platform_url}")
    print("=" * 60)

    create_ui().launch(
        server_port=PLATFORM_PORT,
        inbrowser=False,
        prevent_thread_lock=True,
        debug=False,
    )
    create_landing(platform_url=platform_url).launch(
        server_port=LANDING_PORT,
        inbrowser=True,
        debug=True,
        css=LANDING_CSS,
        theme=LANDING_THEME,
    )
