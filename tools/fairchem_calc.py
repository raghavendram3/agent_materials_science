"""
FairChem (Open Catalyst Project) integration for ML-based energy calculations.

This module provides a wrapper around FairChem's Universal Materials Accelerator (UMA)
models for fast, ML-accelerated DFT-quality calculations of adsorption energies.
"""

import os
from typing import Optional, Dict, List
import numpy as np

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator

# Try to import fairchem
try:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    FAIRCHEM_AVAILABLE = True
    FAIRCHEM_NEW_API = True
except ImportError:
    try:
        # Try older API
        from fairchem.core.models.model_registry import available_pretrained_models
        FAIRCHEM_AVAILABLE = True
        FAIRCHEM_NEW_API = False
    except ImportError:
        FAIRCHEM_AVAILABLE = False
        FAIRCHEM_NEW_API = False

FAIRCHEM_IMPORT_ERROR = None if FAIRCHEM_AVAILABLE else "fairchem-core not installed"

# Reference energies for isolated atoms/molecules (in eV)
# These are typical DFT reference values - adjust based on your reference state
ADSORBATE_REFERENCE_ENERGIES = {
    "H": -3.39,      # 1/2 H2 gas phase reference
    "H2": -6.78,     # H2 molecule
    "O": -4.93,      # 1/2 O2 gas phase reference  
    "O2": -9.86,     # O2 molecule
    "C": -7.37,      # Graphite reference
    "CO": -14.79,    # CO molecule
    "CO2": -22.96,   # CO2 molecule
    "N": -8.32,      # 1/2 N2 gas phase reference
    "N2": -16.64,    # N2 molecule
    "OH": -8.32,     # OH radical
    "H2O": -14.22,   # H2O molecule
    "CH4": -24.03,   # Methane
    "NH3": -19.54,   # Ammonia
}


class FairchemCalculator:
    """
    Wrapper for FairChem calculator for ML-accelerated energy and force calculations.
    
    This calculator uses pre-trained models from the Open Catalyst Project (OCP)
    to compute energies, forces, and perform structure relaxations at near-DFT
    accuracy with ML speed.
    """
    
    def __init__(
        self,
        model_name: str = "uma-s-1p1",
        checkpoint_path: Optional[str] = None,
        cpu: bool = True,
        task_name: str = "omat"
    ):
        """
        Initialize FairChem calculator.
        
        Args:
            model_name: Name of the pre-trained model. Available models:
                       - 'uma-s-1': UMA Small v1 (fastest, good accuracy)
                       - 'uma-s-1p1': UMA Small v1.1 (improved)
                       - 'uma-m-1p1': UMA Medium v1.1 (higher accuracy)
                       - 'esen-*': Various ESEN models
            checkpoint_path: Optional path to a custom checkpoint file
            cpu: If True, force CPU usage. If False, use CUDA if available.
            task_name: Task name for the calculator (e.g., 'omat', 'oc20')
        """
        if not FAIRCHEM_AVAILABLE:
            error_msg = (
                "FairChem is not installed. Install it with:\n"
                "  pip install fairchem-core\n\n"
                "Or for full installation:\n"
                "  pip install fairchem-core torch"
            )
            raise ImportError(error_msg)
        
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.cpu = cpu
        self.task_name = task_name
        self._calculator = None
        
    def get_calculator(self) -> Calculator:
        """
        Get or create the ASE calculator.
        
        Returns:
            ASE Calculator object configured with FairChem model
        """
        if self._calculator is None:
            if not FAIRCHEM_NEW_API:
                raise ImportError(
                    "FairChem new API (FAIRChemCalculator) is required. "
                    "Please update fairchem-core: pip install --upgrade fairchem-core"
                )
            
            # Determine device
            device = 'cpu' if self.cpu else 'cuda'
            
            try:
                # Load pretrained model
                if self.checkpoint_path:
                    # Use custom checkpoint
                    predictor = pretrained_mlip.get_predict_unit(
                        checkpoint_path=self.checkpoint_path,
                        device=device
                    )
                else:
                    # Use pretrained model by name
                    predictor = pretrained_mlip.get_predict_unit(
                        self.model_name,
                        device=device
                    )
                
                # Create calculator
                self._calculator = FAIRChemCalculator(
                    predictor,
                    task_name=self.task_name
                )
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize FairChem calculator with model '{self.model_name}': {e}\n"
                    f"Available models: {get_available_models()}"
                ) from e
        
        return self._calculator
    
    def calculate_energy(self, atoms: Atoms) -> float:
        """
        Calculate potential energy of a structure.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Potential energy in eV
        """
        atoms_copy = atoms.copy()
        atoms_copy.set_pbc([True, True, True])
        
        calc = self.get_calculator()
        atoms_copy.calc = calc
        
        energy = atoms_copy.get_potential_energy()
        return float(energy)
    
    def calculate_forces(self, atoms: Atoms) -> np.ndarray:
        """
        Calculate forces on atoms.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Forces array of shape (N, 3) in eV/Å
        """
        atoms_copy = atoms.copy()
        atoms_copy.set_pbc([True, True, True])
        
        calc = self.get_calculator()
        atoms_copy.calc = calc
        
        forces = atoms_copy.get_forces()
        return forces
    
    def relax_structure(
        self,
        atoms: Atoms,
        fmax: float = 0.05,
        steps: int = 200,
        optimizer: str = "LBFGS"
    ) -> Atoms:
        """
        Relax a structure using the FairChem calculator.
        
        Args:
            atoms: ASE Atoms object to relax
            fmax: Maximum force criterion for convergence (eV/Å)
            steps: Maximum number of optimization steps
            optimizer: Optimizer to use ('LBFGS', 'BFGS', 'FIRE')
            
        Returns:
            Relaxed ASE Atoms object
        """
        from ase.optimize import LBFGS, BFGS, FIRE
        
        optimizers = {
            "LBFGS": LBFGS,
            "BFGS": BFGS,
            "FIRE": FIRE,
        }
        
        if optimizer not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer}. Use one of {list(optimizers.keys())}")
        
        atoms_copy = atoms.copy()
        atoms_copy.set_pbc([True, True, True])
        
        calc = self.get_calculator()
        atoms_copy.calc = calc
        
        opt = optimizers[optimizer](atoms_copy, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        
        return atoms_copy
    
    def calculate_adsorption_energy(
        self,
        slab_with_adsorbate: Atoms,
        clean_slab: Atoms,
        adsorbate: Optional[Atoms] = None,
        adsorbate_name: Optional[str] = None,
        relax: bool = False,
        fmax: float = 0.05,
        use_reference_energy: bool = True
    ) -> Dict[str, float]:
        """
        Calculate adsorption energy.
        
        E_ads = E(slab+ads) - E(slab) - E(ads)
        
        A negative adsorption energy indicates favorable binding.
        
        Note: For isolated adsorbates (single atoms or small molecules), FairChem
        cannot calculate energies directly due to the graph cutoff radius. Use
        either `adsorbate_name` with reference energies, or provide your own
        reference energy.
        
        Args:
            slab_with_adsorbate: Slab with adsorbate placed on it
            clean_slab: Clean slab without adsorbate
            adsorbate: Isolated adsorbate molecule (optional if using reference)
            adsorbate_name: Name of adsorbate for reference energy lookup 
                           (e.g., 'H', 'CO', 'O', 'H2O'). Required if adsorbate
                           is a single atom or small molecule.
            relax: Whether to relax structures before calculating energies
            fmax: Maximum force for relaxation
            use_reference_energy: If True, use tabulated reference energies for
                                  the adsorbate instead of calculating directly.
                                  Recommended for single atoms and small molecules.
            
        Returns:
            Dictionary with energies:
                - 'adsorption_energy': E_ads in eV
                - 'e_combined': E(slab+ads) in eV
                - 'e_slab': E(slab) in eV
                - 'e_adsorbate': E(ads) in eV
                - 'reference_used': Whether reference energy was used
        """
        # Optionally relax structures (only slab structures, not isolated adsorbate)
        if relax:
            slab_with_ads_calc = self.relax_structure(slab_with_adsorbate, fmax=fmax)
            clean_slab_calc = self.relax_structure(clean_slab, fmax=fmax)
        else:
            slab_with_ads_calc = slab_with_adsorbate
            clean_slab_calc = clean_slab
        
        # Calculate energies for slab systems
        e_combined = self.calculate_energy(slab_with_ads_calc)
        e_slab = self.calculate_energy(clean_slab_calc)
        
        # Get adsorbate energy - use reference if available/requested
        reference_used = False
        e_ads = None
        
        # Try to determine adsorbate name from Atoms object if not provided
        if adsorbate_name is None and adsorbate is not None:
            # Try to infer from chemical formula
            formula = adsorbate.get_chemical_formula()
            if formula in ADSORBATE_REFERENCE_ENERGIES:
                adsorbate_name = formula
        
        # Use reference energy if requested and available
        if use_reference_energy and adsorbate_name is not None:
            if adsorbate_name in ADSORBATE_REFERENCE_ENERGIES:
                e_ads = ADSORBATE_REFERENCE_ENERGIES[adsorbate_name]
                reference_used = True
            else:
                available = list(ADSORBATE_REFERENCE_ENERGIES.keys())
                print(f"Warning: No reference energy for '{adsorbate_name}'. "
                      f"Available: {available}. Attempting direct calculation.")
        
        # If no reference energy, try to calculate directly
        if e_ads is None:
            if adsorbate is None:
                raise ValueError(
                    "Either provide an adsorbate Atoms object or specify "
                    "adsorbate_name for reference energy lookup."
                )
            
            # Check if adsorbate can be calculated (needs multiple atoms within cutoff)
            if len(adsorbate) == 1:
                raise ValueError(
                    f"Cannot calculate energy for single atom '{adsorbate.get_chemical_formula()}'. "
                    f"Use adsorbate_name parameter with one of: {list(ADSORBATE_REFERENCE_ENERGIES.keys())}"
                )
            
            # Check if atoms are close enough (within 6 Å cutoff)
            if len(adsorbate) > 1:
                from ase.geometry import get_distances
                positions = adsorbate.get_positions()
                _, distances = get_distances(positions, cell=adsorbate.get_cell(), pbc=adsorbate.get_pbc())
                min_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else float('inf')
                
                if min_dist > 6.0:
                    raise ValueError(
                        f"Adsorbate atoms are {min_dist:.2f} Å apart, exceeding the 6 Å cutoff. "
                        f"Use adsorbate_name parameter with one of: {list(ADSORBATE_REFERENCE_ENERGIES.keys())}"
                    )
            
            try:
                e_ads = self.calculate_energy(adsorbate)
            except Exception as e:
                if "No edges found" in str(e) or "single atom" in str(e).lower():
                    raise ValueError(
                        f"Cannot calculate isolated adsorbate energy directly. "
                        f"Use adsorbate_name parameter with one of: {list(ADSORBATE_REFERENCE_ENERGIES.keys())}"
                    ) from e
                raise
        
        # Calculate adsorption energy
        e_adsorption = e_combined - e_slab - e_ads
        
        return {
            "adsorption_energy": e_adsorption,
            "e_combined": e_combined,
            "e_slab": e_slab,
            "e_adsorbate": e_ads,
            "reference_used": reference_used,
        }


def get_available_models() -> List[str]:
    """
    Get list of available FairChem model names.
    
    Returns:
        List of model name strings
    """
    return [
        "uma-s-1",
        "uma-s-1p1",
        "uma-m-1p1",
        "esen-md-direct-all-omol",
        "esen-sm-conserving-all-omol",
        "esen-sm-direct-all-omol",
        "esen-sm-conserving-all-oc25",
        "esen-md-direct-all-oc25",
    ]


def check_fairchem_installation() -> Dict[str, bool]:
    """
    Check FairChem installation status.
    
    Returns:
        Dictionary with installation status
    """
    status = {
        "fairchem_available": FAIRCHEM_AVAILABLE,
        "new_api": FAIRCHEM_NEW_API,
        "error": FAIRCHEM_IMPORT_ERROR,
    }
    
    # Check for CUDA
    try:
        import torch
        status["cuda_available"] = torch.cuda.is_available()
        status["torch_version"] = torch.__version__
    except ImportError:
        status["cuda_available"] = False
        status["torch_version"] = None
    
    return status


def create_calculator(
    model: str = "uma-s-1p1",
    cpu: bool = True
) -> Optional[Calculator]:
    """
    Create a simple FairChem calculator instance.
    
    Args:
        model: Model name
        cpu: Force CPU usage
        
    Returns:
        ASE Calculator or None if FairChem not available
    """
    if not FAIRCHEM_AVAILABLE:
        return None
    
    try:
        calc = FairchemCalculator(model_name=model, cpu=cpu)
        return calc.get_calculator()
    except Exception as e:
        print(f"Warning: Could not create FairChem calculator: {e}")
        return None
