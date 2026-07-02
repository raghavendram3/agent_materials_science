"""
Surface/slab generation utilities.

This module provides a high-level interface for creating surface slabs
from bulk structures with control over Miller indices, termination,
and slab parameters.
"""

import warnings
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

from ..tools.ase_tools import (
    structure_to_atoms,
    build_slab,
    apply_supercell,
    get_slab_info,
    fix_bottom_layers,
)


@dataclass
class TerminationInfo:
    """Information about a surface termination."""
    index: int
    formula: str
    n_atoms: int
    is_symmetric: bool
    is_polar: bool
    surface_area: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "formula": self.formula,
            "n_atoms": self.n_atoms,
            "is_symmetric": self.is_symmetric,
            "is_polar": self.is_polar,
            "surface_area": self.surface_area,
        }


class SurfaceBuilder:
    """
    Build surface slabs from bulk structures.
    
    This class provides methods to:
    - Generate surface slabs with specific Miller indices
    - Select among different surface terminations
    - Apply supercell expansions
    - Fix bottom layers for calculations
    """
    
    def __init__(
        self,
        structure: Structure,
        miller_indices: Tuple[int, int, int] = (1, 1, 1),
    ):
        """
        Initialize the surface builder.
        
        Args:
            structure: Pymatgen Structure (bulk)
            miller_indices: Miller indices (h, k, l) for the surface
        """
        self.bulk_structure = structure
        self.miller_indices = miller_indices

        # Caches keyed by the generation parameters. (A single unkeyed cache
        # silently returned stale slabs when the same builder was reused with
        # different slab/vacuum sizes.)
        self._slabs_cache: Dict[Tuple, list] = {}
        self._terminations_cache: Dict[Tuple, List[TerminationInfo]] = {}
    
    @property
    def formula(self) -> str:
        """Chemical formula of the bulk structure."""
        return self.bulk_structure.composition.reduced_formula

    def _get_slabs(
        self,
        min_slab_size: float,
        min_vacuum_size: float,
        in_unit_planes: bool,
        center_slab: bool,
    ) -> list:
        """Generate (or fetch cached) pymatgen slabs for the given parameters."""
        key = (float(min_slab_size), float(min_vacuum_size),
               bool(in_unit_planes), bool(center_slab))
        if key in self._slabs_cache:
            return self._slabs_cache[key]

        h, k, l = self.miller_indices
        slabgen = SlabGenerator(
            initial_structure=self.bulk_structure,
            miller_index=(h, k, l),
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=center_slab,
            in_unit_planes=in_unit_planes,
            primitive=True,
            max_normal_search=max(abs(h), abs(k), abs(l)) + 1,
        )
        slabs = slabgen.get_slabs()
        if not slabs:
            raise ValueError(
                f"Could not generate slabs for ({h},{k},{l}). "
                "The Miller indices may be incompatible with the crystal structure."
            )
        self._slabs_cache[key] = slabs
        return slabs

    def get_available_terminations(
        self,
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 15.0,
        in_unit_planes: bool = False,
        center_slab: bool = True,
    ) -> List[TerminationInfo]:
        """
        Get all available surface terminations.

        Args:
            min_slab_size: Minimum slab thickness. In Angstrom when
                ``in_unit_planes=False``; in number of (hkl) planes when
                ``in_unit_planes=True``.
            min_vacuum_size: Minimum vacuum thickness (Å)
            in_unit_planes: Interpret ``min_slab_size`` in crystal planes.
            center_slab: Center the slab along the surface normal.

        Returns:
            List of TerminationInfo objects
        """
        key = (float(min_slab_size), float(min_vacuum_size),
               bool(in_unit_planes), bool(center_slab))
        if key in self._terminations_cache:
            return self._terminations_cache[key]

        slabs = self._get_slabs(min_slab_size, min_vacuum_size,
                                in_unit_planes, center_slab)

        terminations = []
        for i, slab in enumerate(slabs):
            term = TerminationInfo(
                index=i,
                formula=slab.composition.reduced_formula,
                n_atoms=len(slab),
                is_symmetric=slab.is_symmetric(),
                is_polar=slab.is_polar(),
                surface_area=slab.surface_area,
            )
            terminations.append(term)

        self._terminations_cache[key] = terminations
        return terminations
    
    def build_slab(
        self,
        termination: int = 0,
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 15.0,
        supercell: Tuple[int, int] = (1, 1),
        fix_layers: int = 0,
        in_unit_planes: bool = False,
        center_slab: bool = True,
    ) -> Atoms:
        """
        Build a surface slab with specified parameters.

        Args:
            termination: Index of surface termination to use
            min_slab_size: Minimum slab thickness. In Angstrom when
                ``in_unit_planes=False``; in number of (hkl) planes when
                ``in_unit_planes=True``.
            min_vacuum_size: Vacuum thickness (Å)
            supercell: In-plane supercell (nx, ny)
            fix_layers: Number of bottom layers to fix (0 = no fixing)
            in_unit_planes: Interpret ``min_slab_size`` in crystal planes.
            center_slab: Center the slab along the surface normal.

        Returns:
            ASE Atoms slab
        """
        slabs = self._get_slabs(
            min_slab_size, min_vacuum_size, in_unit_planes, center_slab
        )

        # Validate termination index
        if termination >= len(slabs):
            warnings.warn(
                f"Termination {termination} not available "
                f"(only {len(slabs)} termination(s)). Using 0.",
                stacklevel=2,
            )
            termination = 0

        # Get pymatgen slab
        pmg_slab = slabs[termination]
        
        # Convert to ASE
        slab = structure_to_atoms(pmg_slab)
        
        # Apply supercell
        nx, ny = supercell
        if nx > 1 or ny > 1:
            slab = apply_supercell(slab, nx, ny)
        
        # Fix bottom layers if requested
        if fix_layers > 0:
            slab = fix_bottom_layers(slab, n_layers=fix_layers)
        
        return slab
    
    def build_slab_simple(
        self,
        n_layers: int = 6,
        vacuum: float = 15.0,
        supercell: Tuple[int, int] = (1, 1),
    ) -> Atoms:
        """
        Build a simple slab using ASE's surface() function.
        
        This method is faster but provides less control over termination.
        
        Args:
            n_layers: Number of atomic layers
            vacuum: Vacuum thickness (Å)
            supercell: In-plane supercell (nx, ny)
            
        Returns:
            ASE Atoms slab
        """
        # Convert bulk to ASE
        bulk_atoms = structure_to_atoms(self.bulk_structure)
        
        # Build slab
        slab = build_slab(
            bulk_atoms,
            self.miller_indices,
            layers=n_layers,
            vacuum=vacuum,
        )
        
        # Apply supercell
        nx, ny = supercell
        if nx > 1 or ny > 1:
            slab = apply_supercell(slab, nx, ny)
        
        return slab
    
    def get_slab_info(self, slab: Atoms) -> Dict[str, Any]:
        """
        Get information about a generated slab.
        
        Args:
            slab: ASE Atoms slab
            
        Returns:
            Dictionary with slab properties
        """
        info = get_slab_info(slab)
        info["miller_indices"] = list(self.miller_indices)
        info["bulk_formula"] = self.formula
        return info
    
    def print_terminations(self) -> None:
        """Print available terminations in a formatted table."""
        terms = self.get_available_terminations()
        
        print(f"\nAvailable terminations for {self.formula} ({self.miller_indices}):")
        print("-" * 70)
        print(f"{'Index':^6} {'Formula':^12} {'Atoms':^6} {'Symmetric':^10} {'Polar':^6} {'Area (Å²)':^10}")
        print("-" * 70)
        
        for t in terms:
            print(f"{t.index:^6} {t.formula:^12} {t.n_atoms:^6} "
                  f"{'Yes' if t.is_symmetric else 'No':^10} "
                  f"{'Yes' if t.is_polar else 'No':^6} "
                  f"{t.surface_area:^10.2f}")
        
        print("-" * 70)


def create_surface_from_bulk(
    bulk_structure: Structure,
    miller_indices: Tuple[int, int, int],
    termination: int = 0,
    min_slab_size: float = 10.0,
    vacuum: float = 15.0,
    supercell: Tuple[int, int] = (1, 1),
) -> Tuple[Atoms, List[TerminationInfo]]:
    """
    Convenience function to create a surface slab.
    
    Args:
        bulk_structure: Pymatgen Structure (bulk)
        miller_indices: Miller indices (h, k, l)
        termination: Termination index to use
        min_slab_size: Minimum slab thickness (Å)
        vacuum: Vacuum thickness (Å)
        supercell: In-plane supercell (nx, ny)
        
    Returns:
        Tuple of (ASE Atoms slab, list of available terminations)
    """
    builder = SurfaceBuilder(bulk_structure, miller_indices)
    
    # Get terminations
    terminations = builder.get_available_terminations(min_slab_size, vacuum)
    
    # Build slab
    slab = builder.build_slab(
        termination=termination,
        min_slab_size=min_slab_size,
        min_vacuum_size=vacuum,
        supercell=supercell,
    )
    
    return slab, terminations


def miller_to_string(miller: Tuple[int, int, int]) -> str:
    """
    Convert Miller indices to a string representation.
    
    Args:
        miller: (h, k, l) tuple
        
    Returns:
        String like '111' or '1m10' for negative indices
    """
    def fmt(x: int) -> str:
        if x < 0:
            return f"m{abs(x)}"
        return str(x)
    
    h, k, l = miller
    return f"{fmt(h)}{fmt(k)}{fmt(l)}"


def string_to_miller(s: str) -> Tuple[int, int, int]:
    """
    Parse a string to Miller indices.
    
    Args:
        s: String like '111', '110', '1,1,1', '1-10'
        
    Returns:
        (h, k, l) tuple
    """
    s = s.strip().replace(" ", "")
    
    # Handle comma-separated
    if "," in s:
        parts = s.split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid Miller indices: {s}")
        return tuple(int(p) for p in parts)
    
    # Handle 'm' for negative (e.g., '1m10' = (1, -1, 0))
    indices = []
    i = 0
    while i < len(s):
        if s[i] == "m" or s[i] == "-":
            if i + 1 < len(s):
                indices.append(-int(s[i + 1]))
                i += 2
            else:
                raise ValueError(f"Invalid Miller indices: {s}")
        else:
            indices.append(int(s[i]))
            i += 1
    
    if len(indices) != 3:
        raise ValueError(f"Invalid Miller indices: {s}")
    
    return tuple(indices)
