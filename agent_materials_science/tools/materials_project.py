"""
Materials Project API wrapper for fetching crystal structures.
"""

import os
from typing import Optional, Tuple, List, Dict, Any

from dotenv import load_dotenv

try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    MPRester = None

from pymatgen.core import Structure


def _load_api_key(provided: Optional[str] = None) -> Optional[str]:
    """
    Load MP API key from provided argument or environment.
    Looks for MP_API_KEY (preferred) or MAPI_KEY (legacy).
    """
    load_dotenv()
    return provided or os.getenv("MP_API_KEY") or os.getenv("MAPI_KEY")


class MaterialsProjectTool:
    """
    Wrapper around Materials Project API for fetching crystal structures.
    
    This tool provides methods to:
    - Fetch structures by MP material ID (e.g., 'mp-149')
    - Search structures by chemical formula (e.g., 'Si', 'SrTiO3')
    - Get material properties and metadata
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Materials Project tool.
        
        Args:
            api_key: Materials Project API key. If not provided,
                    will look for MP_API_KEY or MAPI_KEY in environment.
        """
        self.api_key = _load_api_key(api_key)
        
        if not MP_API_AVAILABLE:
            raise ImportError(
                "mp-api is not installed. Please install it with: pip install mp-api"
            )

    def _get_client(self):
        """Get MP API client context manager."""
        if not self.api_key:
            raise RuntimeError(
                "Materials Project API key not found. Set MP_API_KEY environment "
                "variable or pass api_key to MaterialsProjectTool constructor. "
                "Get your API key at: https://materialsproject.org/dashboard"
            )
        return MPRester(self.api_key)

    def get_structure_by_mp_id(self, mp_id: str) -> Structure:
        """
        Fetch a pymatgen Structure by Materials Project ID.
        
        Args:
            mp_id: Materials Project ID (e.g., 'mp-149' for Silicon)
            
        Returns:
            pymatgen Structure object
            
        Raises:
            RuntimeError: If structure cannot be fetched
        """
        mp_id = mp_id.strip()
        
        try:
            with self._get_client() as mpr:
                # Canonical high-level accessor: resolves deprecated /
                # re-mapped material IDs automatically.
                struct = mpr.get_structure_by_material_id(mp_id)

                if struct is None:
                    raise RuntimeError(f"No structure found for {mp_id}")

                return struct
                
        except Exception as e:
            error_msg = str(e)
            if "Invalid" in error_msg or "401" in error_msg or "API key" in error_msg:
                raise RuntimeError(
                    "Invalid or expired Materials Project API key. "
                    "Get a new key at: https://materialsproject.org/dashboard"
                ) from e
            raise RuntimeError(f"Failed to fetch structure for {mp_id}: {e}") from e

    def get_structure_by_formula(
        self, 
        formula: str,
        stable_only: bool = True
    ) -> Tuple[Structure, str]:
        """
        Fetch the most stable structure for a given chemical formula.
        
        Args:
            formula: Chemical formula (e.g., 'Si', 'SrTiO3', 'Fe2O3')
            stable_only: If True, prefer thermodynamically stable structures
            
        Returns:
            Tuple of (Structure, material_id)
            
        Raises:
            RuntimeError: If no structure found
        """
        formula = formula.strip()
        
        try:
            with self._get_client() as mpr:
                results = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "structure", "energy_above_hull", 
                           "formula_pretty", "is_stable"],
                )
                
                if not results:
                    raise RuntimeError(f"No materials found for formula '{formula}'")
                
                # Sort by energy above hull (stability)
                def get_energy(doc):
                    e = getattr(doc, "energy_above_hull", None)
                    return float("inf") if e is None else e
                
                results_sorted = sorted(results, key=get_energy)
                
                # Get most stable structure
                top = results_sorted[0]
                mp_id = getattr(top, "material_id", None)
                struct = getattr(top, "structure", None)
                
                if not mp_id or struct is None:
                    raise RuntimeError(f"Invalid data returned for formula '{formula}'")
                
                return struct, str(mp_id)
                
        except Exception as e:
            error_msg = str(e)
            if "Invalid" in error_msg or "401" in error_msg or "API key" in error_msg:
                raise RuntimeError(
                    "Invalid or expired Materials Project API key. "
                    "Get a new key at: https://materialsproject.org/dashboard"
                ) from e
            raise RuntimeError(f"Failed to fetch structure for '{formula}': {e}") from e

    def search_materials(
        self,
        formula: Optional[str] = None,
        elements: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for materials matching criteria.
        
        Args:
            formula: Chemical formula to search
            elements: List of elements that must be present
            max_results: Maximum number of results to return
            
        Returns:
            List of material summaries

        Raises:
            ValueError: If neither ``formula`` nor ``elements`` is given
                (an unconstrained query would download the entire Materials
                Project summary collection - roughly 150k documents).
        """
        if not formula and not elements:
            raise ValueError(
                "search_materials requires at least one criterion "
                "(formula or elements); an unconstrained query would "
                "download the entire Materials Project database."
            )

        try:
            with self._get_client() as mpr:
                kwargs = {
                    "fields": ["material_id", "formula_pretty", "energy_above_hull",
                              "band_gap", "is_stable", "symmetry"],
                    # Bound the download: one chunk, sized to the request.
                    "num_chunks": 1,
                    "chunk_size": max(10, min(int(max_results), 100)),
                }
                
                if formula:
                    kwargs["formula"] = formula
                if elements:
                    kwargs["elements"] = elements
                
                results = mpr.materials.summary.search(**kwargs)
                
                # Convert to dictionaries
                materials = []
                for doc in results[:max_results]:
                    mat = {
                        "material_id": str(getattr(doc, "material_id", "")),
                        "formula": getattr(doc, "formula_pretty", ""),
                        "energy_above_hull": getattr(doc, "energy_above_hull", None),
                        "band_gap": getattr(doc, "band_gap", None),
                        "is_stable": getattr(doc, "is_stable", None),
                    }
                    materials.append(mat)
                
                return materials
                
        except Exception as e:
            raise RuntimeError(f"Material search failed: {e}") from e

    def get_material_properties(self, mp_id: str) -> Dict[str, Any]:
        """
        Get detailed properties for a material.
        
        Args:
            mp_id: Materials Project ID
            
        Returns:
            Dictionary of material properties
        """
        mp_id = mp_id.strip()
        
        try:
            with self._get_client() as mpr:
                results = mpr.materials.summary.search(
                    material_ids=[mp_id],
                    fields=[
                        "material_id", "formula_pretty", "structure",
                        "energy_above_hull", "energy_per_atom", "formation_energy_per_atom",
                        "band_gap", "is_stable", "is_metal",
                        "symmetry", "volume", "density",
                        "nsites", "elements"
                    ],
                )
                
                if not results:
                    raise RuntimeError(f"No material found for {mp_id}")
                
                doc = results[0]
                
                # Extract symmetry info
                symmetry = getattr(doc, "symmetry", None)
                space_group = None
                crystal_system = None
                if symmetry:
                    space_group = getattr(symmetry, "symbol", None)
                    crystal_system = getattr(symmetry, "crystal_system", None)
                
                return {
                    "material_id": str(getattr(doc, "material_id", "")),
                    "formula": getattr(doc, "formula_pretty", ""),
                    "energy_above_hull": getattr(doc, "energy_above_hull", None),
                    "energy_per_atom": getattr(doc, "energy_per_atom", None),
                    "formation_energy": getattr(doc, "formation_energy_per_atom", None),
                    "band_gap": getattr(doc, "band_gap", None),
                    "is_stable": getattr(doc, "is_stable", None),
                    "is_metal": getattr(doc, "is_metal", None),
                    "space_group": space_group,
                    "crystal_system": str(crystal_system) if crystal_system else None,
                    "volume": getattr(doc, "volume", None),
                    "density": getattr(doc, "density", None),
                    "n_sites": getattr(doc, "nsites", None),
                    "elements": [str(e) for e in getattr(doc, "elements", [])],
                }
                
        except Exception as e:
            raise RuntimeError(f"Failed to get properties for {mp_id}: {e}") from e
