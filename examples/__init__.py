"""
Example scripts for the Materials Science Agent.

This package contains example workflows demonstrating various features:
- basic_usage.py: Simple end-to-end workflow
- energy_calculations.py: Energy calculations with FairChem
- interactive_workflow.py: Step-by-step interactive analysis
"""

from .basic_usage import run_basic_workflow
from .energy_calculations import run_energy_workflow

__all__ = ["run_basic_workflow", "run_energy_workflow"]
