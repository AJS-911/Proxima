#!/usr/bin/env python3
"""Launcher script for Proxima TUI."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    try:
        from proxima.tui.app import ProximaTUI
        print("Starting Proxima TUI...")
        app = ProximaTUI()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
