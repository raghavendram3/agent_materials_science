"""
Main entry point for running the agent as a module.

Usage:
    python -m agent_materials_science --material Si --miller 1,1,1 --adsorbate H
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
