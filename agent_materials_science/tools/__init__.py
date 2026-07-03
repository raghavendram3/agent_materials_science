"""
Tools for materials science operations.
"""

from .materials_project import MaterialsProjectTool
from .ase_tools import (
    structure_to_atoms,
    build_slab,
    build_slab_with_termination,
    apply_supercell,
    save_outputs,
    get_slab_info,
)
from .fairchem_calc import FairchemCalculator, FAIRCHEM_AVAILABLE
from .converters import create_adsorbate, place_adsorbate_at_site

__all__ = [
    "MaterialsProjectTool",
    "structure_to_atoms",
    "build_slab",
    "build_slab_with_termination",
    "apply_supercell",
    "save_outputs",
    "get_slab_info",
    "FairchemCalculator",
    "FAIRCHEM_AVAILABLE",
    "create_adsorbate",
    "place_adsorbate_at_site",
]
