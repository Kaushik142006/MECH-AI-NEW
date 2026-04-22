"""MECH-AI entry: integrated modelling + simulation."""
from Modelling import STL_PATH
from Frontend import create_ui

if __name__ == "__main__":
    print("=" * 60)
    print("  MECH-AI Integrated Modelling & Simulation Platform")
    print(f"  STL output: {STL_PATH}")
    print("  Server: http://localhost:7860")
    print("=" * 60)
    create_ui().launch(
        inbrowser=True,
        debug=True,
    )
