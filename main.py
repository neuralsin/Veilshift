#!/usr/bin/env python3
"""
QT-2.23 — Quantum Sensor Fusion & Stealth Detection Console

Entry point. Launches the CustomTkinter application shell.

Usage:
    python main.py
    python main.py --presentation    (launch in fullscreen presentation mode)
    python main.py --seed 42         (set default random seed)
    python main.py --samples 3000    (reduce sample count for faster testing)
"""

import sys
import os
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="QT-2.23 — Quantum Sensor Fusion Console")
    parser.add_argument("--presentation", action="store_true", help="Launch in presentation mode")
    parser.add_argument("--seed", type=int, default=42, help="Default random seed")
    parser.add_argument("--samples", type=int, default=6000, help="Default sample count per sensor")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Set matplotlib backend before any imports
    import matplotlib
    matplotlib.use("Agg")

    # Import after path setup
    from ui.app_shell import AppShell

    # Create and configure application
    app = AppShell()

    # Apply CLI args
    if args.seed != 42:
        app.app_state.current_experiment.seed = args.seed
    if args.samples != 6000:
        app.app_state.current_experiment.radar_config.num_samples = args.samples
        app.app_state.current_experiment.thermal_config.num_samples = args.samples
        app.app_state.current_experiment.acoustic_config.num_samples = args.samples
    if args.debug:
        app.app_state.debug_mode = True
    if args.presentation:
        app.after(500, app._toggle_presentation_mode)

    # Set window close handler
    app.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Run
    app.mainloop()


if __name__ == "__main__":
    main()
