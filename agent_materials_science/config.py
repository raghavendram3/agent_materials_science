"""
Configuration management for the Materials Science Agent.

Precedence for FairChem settings (model, task, device):
    explicit constructor argument > environment variable > built-in default
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from dotenv import load_dotenv

# Single source of truth for adsorbate geometries lives in tools.converters.
from .tools.converters import ADSORBATE_GEOMETRIES

# Load environment variables
load_dotenv()

# Built-in defaults (used only when neither an explicit argument nor an
# environment variable is provided).
DEFAULT_FAIRCHEM_MODEL = "uma-s-1p2"
DEFAULT_FAIRCHEM_TASK = "oc20"
DEFAULT_DEVICE = "auto"

VALID_DEVICES = {"auto", "cpu", "cuda"}
VALID_SITE_TYPES = {"top", "bridge", "hollow", "fcc", "hcp"}
VALID_SITE_FINDERS = {"auto", "builtin", "pymatgen"}


@dataclass
class AgentConfig:
    """
    Configuration for the Materials Science Adsorption Agent.

    Attributes:
        material: Chemical formula or Materials Project ID (e.g., 'Si', 'mp-149')
        miller_indices: Miller indices for surface cleaving (h, k, l)
        adsorbate: Adsorbate species (e.g., 'H', 'O', 'CO', 'OH')
        termination: Optional surface termination index (0 = most stable)
        n_layers: Slab thickness. With ``layers_in_unit_planes=True`` (default)
            this is the number of (hkl) atomic planes; otherwise it is
            converted to Angstrom with a rough 2.5 A/layer heuristic
            (legacy behaviour).
        layers_in_unit_planes: Interpret ``n_layers`` as crystal planes
            (pymatgen ``in_unit_planes=True``) instead of a 2.5 A heuristic.
        vacuum: Vacuum thickness in Angstrom
        supercell: In-plane supercell dimensions (nx, ny)
        center_slab: Center the slab in the cell along the surface normal
        fix_layers: Number of bottom atomic layers to constrain (FixAtoms).
            Recommended >= 2 when relaxing structures so the slab bottom
            mimics bulk. 0 disables constraints (legacy behaviour).
        height_offset: Height above surface for adsorbate placement (Angstrom)
        site_types: Types of adsorption sites to consider
        site_finder: "auto" prefers pymatgen's AdsorbateSiteFinder when
            installed (symmetry-aware, fewer redundant sites) and falls back
            to the built-in geometric finder otherwise. Use "builtin" or
            "pymatgen" to force a backend.
        symm_reduce: Collapse symmetry-equivalent sites (pymatgen backend)
        calculate_energies: Whether to calculate adsorption energies
        relax_structures: Whether to relax structures before energy
            calculations. The clean slab is relaxed once (before site
            finding) and each slab+adsorbate system is relaxed, so
            E_ads = E(slab+ads) - E(slab) - E(ads) is internally consistent.
        fairchem_model: FairChem model name (None -> env FAIRCHEM_MODEL ->
            default). Also accepts a path to a local checkpoint file.
        fairchem_task: UMA task head (None -> env FAIRCHEM_TASK -> 'oc20').
            'oc20' is the catalysis/adsorption head; 'omat' is for bulk
            inorganic materials and 'omol' for isolated molecules. Mixing
            tasks between the slab and the adsorbate reference makes the
            resulting adsorption energies physically meaningless.
        device: 'auto' (CUDA if available), 'cpu' or 'cuda'
            (None -> env FAIRCHEM_DEVICE -> 'auto').
        use_gpu: Deprecated boolean alias for ``device``; kept for backward
            compatibility. use_gpu=True maps to device='cuda',
            use_gpu=False maps to device='cpu'.
        output_dir: Directory for output files
        save_all_sites: Save structures for all sites (not just best)
        mp_api_key: Materials Project API key (from env if not provided)
    """

    # Material specification
    material: str = ""
    mp_id: Optional[str] = None

    # Surface parameters
    miller_indices: Tuple[int, int, int] = (1, 1, 1)
    termination: int = 0  # 0 = most stable termination
    n_layers: int = 6
    layers_in_unit_planes: bool = True
    vacuum: float = 15.0
    supercell: Tuple[int, int] = (1, 1)
    center_slab: bool = True
    fix_layers: int = 0

    # Adsorbate parameters
    adsorbate: str = "H"
    height_offset: float = 2.0
    site_types: List[str] = field(default_factory=lambda: ["top", "bridge", "hollow"])

    # Site-finding backend
    site_finder: str = "auto"
    symm_reduce: bool = True  # collapse symmetry-equivalent sites (pymatgen backend)

    # Calculation parameters
    calculate_energies: bool = False
    relax_structures: bool = False
    relax_fmax: float = 0.05  # eV/A
    relax_steps: int = 200

    # FairChem settings (None = resolve from environment, then defaults)
    fairchem_model: Optional[str] = None
    fairchem_task: Optional[str] = None
    device: Optional[str] = None
    use_gpu: Optional[bool] = None  # deprecated; use `device`

    # Output settings
    output_dir: str = "outputs"
    save_all_sites: bool = False
    verbose: bool = True

    # API settings
    mp_api_key: Optional[str] = None

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Load API key from environment if not provided
        if self.mp_api_key is None:
            self.mp_api_key = os.getenv("MP_API_KEY") or os.getenv("MAPI_KEY")

        # Resolve FairChem model/task: explicit argument wins over environment.
        if self.fairchem_model is None:
            self.fairchem_model = os.getenv("FAIRCHEM_MODEL") or DEFAULT_FAIRCHEM_MODEL
        if self.fairchem_task is None:
            self.fairchem_task = os.getenv("FAIRCHEM_TASK") or DEFAULT_FAIRCHEM_TASK

        # Resolve device: explicit `device` > deprecated `use_gpu` > env > auto.
        if self.device is None:
            if self.use_gpu is True:
                self.device = "cuda"
            elif self.use_gpu is False:
                self.device = "cpu"
            else:
                env_device = (os.getenv("FAIRCHEM_DEVICE") or "").lower().strip()
                self.device = env_device if env_device in VALID_DEVICES else DEFAULT_DEVICE
        self.device = self.device.lower()
        if self.device not in VALID_DEVICES:
            raise ValueError(f"Invalid device: {self.device}. Valid: {sorted(VALID_DEVICES)}")
        # Keep the deprecated flag coherent for any downstream reader.
        self.use_gpu = self.device == "cuda"

        # Parse material specification
        if self.material and not self.mp_id:
            if self.material.startswith("mp-"):
                self.mp_id = self.material
                self.material = ""

        # Validate miller indices
        if len(self.miller_indices) != 3 or not all(
            isinstance(i, int) for i in self.miller_indices
        ):
            raise ValueError("Miller indices must be three integers")

        # Validate surface parameters
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.vacuum < 0:
            raise ValueError("vacuum must be >= 0")
        if self.fix_layers < 0:
            raise ValueError("fix_layers must be >= 0")

        # Validate site types
        for st in self.site_types:
            if st not in VALID_SITE_TYPES:
                raise ValueError(
                    f"Invalid site type: {st}. Valid types: {sorted(VALID_SITE_TYPES)}"
                )

        # Validate site-finder backend
        if self.site_finder not in VALID_SITE_FINDERS:
            raise ValueError(
                f"Invalid site_finder: {self.site_finder}. "
                f"Valid: {sorted(VALID_SITE_FINDERS)}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from dictionary."""
        # Work on a copy so we never mutate the caller's dictionary.
        config_dict = dict(config_dict)

        # Handle miller_indices if provided as list
        if "miller_indices" in config_dict and isinstance(config_dict["miller_indices"], list):
            config_dict["miller_indices"] = tuple(config_dict["miller_indices"])

        # Handle supercell if provided as list
        if "supercell" in config_dict and isinstance(config_dict["supercell"], list):
            config_dict["supercell"] = tuple(config_dict["supercell"])

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "material": self.material,
            "mp_id": self.mp_id,
            "miller_indices": list(self.miller_indices),
            "termination": self.termination,
            "n_layers": self.n_layers,
            "layers_in_unit_planes": self.layers_in_unit_planes,
            "vacuum": self.vacuum,
            "supercell": list(self.supercell),
            "center_slab": self.center_slab,
            "fix_layers": self.fix_layers,
            "adsorbate": self.adsorbate,
            "height_offset": self.height_offset,
            "site_types": self.site_types,
            "site_finder": self.site_finder,
            "symm_reduce": self.symm_reduce,
            "calculate_energies": self.calculate_energies,
            "relax_structures": self.relax_structures,
            "relax_fmax": self.relax_fmax,
            "relax_steps": self.relax_steps,
            "fairchem_model": self.fairchem_model,
            "fairchem_task": self.fairchem_task,
            "device": self.device,
            "output_dir": self.output_dir,
            "save_all_sites": self.save_all_sites,
            "verbose": self.verbose,
        }


# ---------------------------------------------------------------------------
# Adsorbates
# ---------------------------------------------------------------------------
# The geometry table in tools.converters is the single source of truth for
# which species are supported. Descriptions below are optional metadata; every
# geometry-table species is available even without a description here.
ADSORBATE_DESCRIPTIONS = {
    "H": "Hydrogen atom",
    "O": "Oxygen atom",
    "N": "Nitrogen atom",
    "C": "Carbon atom",
    "S": "Sulfur atom",
    "F": "Fluorine atom",
    "Cl": "Chlorine atom",
    "H2": "Hydrogen molecule",
    "O2": "Oxygen molecule",
    "N2": "Nitrogen molecule",
    "CO": "Carbon monoxide (C-down)",
    "OC": "Carbon monoxide (O-down)",
    "OH": "Hydroxyl (O-down)",
    "HO": "Hydroxyl (H-down, dissociation studies)",
    "NO": "Nitric oxide (N-down)",
    "ON": "Nitric oxide (O-down)",
    "HF": "Hydrogen fluoride",
    "HCl": "Hydrogen chloride",
    "H2O": "Water molecule",
    "CO2": "Carbon dioxide",
    "N2O": "Nitrous oxide",
    "NH3": "Ammonia",
    "CH4": "Methane",
    "C2H2": "Acetylene",
    "C2H4": "Ethylene",
    "HCOO": "Formate",
    "CH3OH": "Methanol",
}


def get_available_adsorbates() -> List[str]:
    """Get list of available adsorbate species (from the geometry table)."""
    return list(ADSORBATE_GEOMETRIES.keys())


def get_adsorbate_info(species: str) -> Dict[str, Any]:
    """Get information about an adsorbate species."""
    if species not in ADSORBATE_GEOMETRIES:
        raise ValueError(
            f"Unknown adsorbate: {species}. Available: {get_available_adsorbates()}"
        )
    geometry = ADSORBATE_GEOMETRIES[species]
    return {
        "atoms": geometry,
        "description": ADSORBATE_DESCRIPTIONS.get(species, species),
    }


# ---------------------------------------------------------------------------
# FairChem models
# ---------------------------------------------------------------------------
# Static fallback list, current as of fairchem-core 2.21. NOTE: 'uma-s-1' has
# been removed from the FairChem registry and is no longer downloadable.
# When fairchem-core is installed, get_available_models() queries the live
# registry instead so this list can never go stale.
FAIRCHEM_MODELS = {
    "uma-s-1p2": "Universal Model for Atoms - Small v1.2 (default)",
    "uma-s-1p1": "Universal Model for Atoms - Small v1.1",
    "uma-m-1p1": "Universal Model for Atoms - Medium v1.1 (higher accuracy)",
    "esen-md-direct-all-omol": "eSEN MD direct (OMol)",
    "esen-sm-conserving-all-omol": "eSEN small conserving (OMol)",
    "esen-sm-direct-all-omol": "eSEN small direct (OMol)",
    "esen-sm-conserving-all-oc25": "eSEN small conserving (OC25)",
    "esen-md-direct-all-oc25": "eSEN MD direct (OC25)",
}


def get_available_models() -> List[str]:
    """
    Get the list of available FairChem model names.

    Queries the installed fairchem-core registry when available; falls back
    to a static list otherwise.
    """
    try:
        from fairchem.core.calculate.pretrained_mlip import available_models
        return list(available_models)
    except Exception:
        return list(FAIRCHEM_MODELS.keys())
