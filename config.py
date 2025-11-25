"""
Configuration management for the Materials Science Agent.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


@dataclass
class AgentConfig:
    """
    Configuration for the Materials Science Adsorption Agent.
    
    Attributes:
        material: Chemical formula or Materials Project ID (e.g., 'Si', 'mp-149')
        miller_indices: Miller indices for surface cleaving (h, k, l)
        adsorbate: Adsorbate species (e.g., 'H', 'O', 'CO', 'OH')
        termination: Optional surface termination index (0 = most stable)
        n_layers: Number of atomic layers in slab
        vacuum: Vacuum thickness in Angstrom
        supercell: In-plane supercell dimensions (nx, ny)
        height_offset: Height above surface for adsorbate placement (Angstrom)
        site_types: Types of adsorption sites to consider
        calculate_energies: Whether to calculate adsorption energies
        relax_structures: Whether to relax structures before energy calculations
        fairchem_model: FairChem model name for ML calculations
        use_gpu: Whether to use GPU for calculations
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
    vacuum: float = 15.0
    supercell: Tuple[int, int] = (1, 1)
    center_slab: bool = True
    
    # Adsorbate parameters
    adsorbate: str = "H"
    height_offset: float = 2.0
    site_types: List[str] = field(default_factory=lambda: ["top", "bridge", "hollow"])
    
    # Calculation parameters
    calculate_energies: bool = False
    relax_structures: bool = False
    relax_fmax: float = 0.05  # eV/Å
    relax_steps: int = 200
    
    # FairChem settings
    fairchem_model: str = "uma-s-1"
    use_gpu: bool = False
    
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
        
        # Load FairChem model from environment if available
        env_model = os.getenv("FAIRCHEM_MODEL")
        if env_model:
            self.fairchem_model = env_model
        
        # Check GPU setting from environment
        env_device = os.getenv("FAIRCHEM_DEVICE", "").lower()
        if env_device == "cuda":
            self.use_gpu = True
        elif env_device == "cpu":
            self.use_gpu = False
        
        # Parse material specification
        if self.material and not self.mp_id:
            if self.material.startswith("mp-"):
                self.mp_id = self.material
                self.material = ""
        
        # Validate miller indices
        if not all(isinstance(i, int) for i in self.miller_indices):
            raise ValueError("Miller indices must be integers")
        
        # Validate site types
        valid_site_types = {"top", "bridge", "hollow", "fcc", "hcp"}
        for st in self.site_types:
            if st not in valid_site_types:
                raise ValueError(f"Invalid site type: {st}. Valid types: {valid_site_types}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from dictionary."""
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
            "vacuum": self.vacuum,
            "supercell": list(self.supercell),
            "center_slab": self.center_slab,
            "adsorbate": self.adsorbate,
            "height_offset": self.height_offset,
            "site_types": self.site_types,
            "calculate_energies": self.calculate_energies,
            "relax_structures": self.relax_structures,
            "relax_fmax": self.relax_fmax,
            "relax_steps": self.relax_steps,
            "fairchem_model": self.fairchem_model,
            "use_gpu": self.use_gpu,
            "output_dir": self.output_dir,
            "save_all_sites": self.save_all_sites,
            "verbose": self.verbose,
        }


# Predefined adsorbate configurations
ADSORBATES = {
    # Single atoms
    "H": {"atoms": [("H", (0, 0, 0))], "description": "Hydrogen atom"},
    "O": {"atoms": [("O", (0, 0, 0))], "description": "Oxygen atom"},
    "N": {"atoms": [("N", (0, 0, 0))], "description": "Nitrogen atom"},
    "C": {"atoms": [("C", (0, 0, 0))], "description": "Carbon atom"},
    "S": {"atoms": [("S", (0, 0, 0))], "description": "Sulfur atom"},
    
    # Diatomic molecules
    "CO": {
        "atoms": [("C", (0, 0, 0)), ("O", (0, 0, 1.128))],
        "description": "Carbon monoxide (C-down)"
    },
    "OH": {
        "atoms": [("O", (0, 0, 0)), ("H", (0, 0, 0.97))],
        "description": "Hydroxyl radical"
    },
    "NO": {
        "atoms": [("N", (0, 0, 0)), ("O", (0, 0, 1.15))],
        "description": "Nitric oxide"
    },
    
    # Triatomic molecules
    "H2O": {
        "atoms": [
            ("O", (0, 0, 0)),
            ("H", (0.757, 0.587, 0)),
            ("H", (-0.757, 0.587, 0))
        ],
        "description": "Water molecule"
    },
    "CO2": {
        "atoms": [
            ("C", (0, 0, 0)),
            ("O", (0, 0, 1.16)),
            ("O", (0, 0, -1.16))
        ],
        "description": "Carbon dioxide"
    },
    
    # Larger molecules
    "NH3": {
        "atoms": [
            ("N", (0, 0, 0)),
            ("H", (0, 0.942, 0.38)),
            ("H", (0.816, -0.471, 0.38)),
            ("H", (-0.816, -0.471, 0.38))
        ],
        "description": "Ammonia"
    },
    "CH4": {
        "atoms": [
            ("C", (0, 0, 0)),
            ("H", (0.629, 0.629, 0.629)),
            ("H", (-0.629, -0.629, 0.629)),
            ("H", (-0.629, 0.629, -0.629)),
            ("H", (0.629, -0.629, -0.629))
        ],
        "description": "Methane"
    },
}


# Available FairChem models
FAIRCHEM_MODELS = {
    "uma-s-1": "Universal Materials Accelerator - Small v1",
    "uma-s-1p1": "Universal Materials Accelerator - Small v1.1",
    "uma-m-1p1": "Universal Materials Accelerator - Medium v1.1",
    "esen-md-direct-all-omol": "ESEN model for molecular dynamics",
    "esen-sm-conserving-all-omol": "ESEN small conserving model",
    "esen-sm-direct-all-omol": "ESEN small direct model",
}


def get_available_adsorbates() -> List[str]:
    """Get list of available adsorbate species."""
    return list(ADSORBATES.keys())


def get_adsorbate_info(species: str) -> Dict[str, Any]:
    """Get information about an adsorbate species."""
    if species not in ADSORBATES:
        raise ValueError(f"Unknown adsorbate: {species}. Available: {get_available_adsorbates()}")
    return ADSORBATES[species]


def get_available_models() -> List[str]:
    """Get list of available FairChem models."""
    return list(FAIRCHEM_MODELS.keys())
