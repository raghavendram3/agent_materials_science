"""
Materials Science Adsorption Agent

A Python agent for automated materials science workflows, focusing on
surface adsorption analysis using Materials Project data, ASE/pymatgen 
tools, and FairChem UMA machine learning potentials.
"""

from .config import AgentConfig
from .agent import (
    MaterialsScienceAgent,
    AgentResult,
    InteractiveAgent,
    run_analysis,
)

__version__ = "0.2.1"
__author__ = "Materials Science Agent"

__all__ = [
    "MaterialsScienceAgent",
    "AgentConfig",
    "AgentResult",
    "InteractiveAgent",
    "run_analysis",
]
