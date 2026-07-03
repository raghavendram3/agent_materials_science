"""
Materials Science Adsorption Agent

This module provides the main agent class that orchestrates the complete
workflow for surface adsorption analysis.
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from .config import AgentConfig, get_available_adsorbates, get_available_models
from .core.workflow import AdsorptionWorkflow, WorkflowResult
from .core.surface import SurfaceBuilder, TerminationInfo
from .core.adsorption import AdsorptionSiteFinder, AdsorptionSite
from .tools.materials_project import MaterialsProjectTool
from .tools.ase_tools import structure_to_atoms, get_slab_info
from .tools.converters import create_adsorbate, list_available_adsorbates
from .tools.fairchem_calc import FAIRCHEM_AVAILABLE, check_fairchem_installation


# Re-export WorkflowResult as AgentResult for cleaner API
AgentResult = WorkflowResult


class MaterialsScienceAgent:
    """
    Main agent for materials science adsorption analysis.
    
    This agent provides a complete workflow for:
    1. Fetching materials from Materials Project
    2. Creating surface slabs with specific Miller indices and terminations
    3. Identifying adsorption sites on surfaces
    4. Calculating adsorption energies using ML potentials (FairChem)
    5. Ranking sites and generating reports
    
    Example:
        ```python
        from agent_materials_science import MaterialsScienceAgent, AgentConfig
        
        config = AgentConfig(
            material="Si",
            miller_indices=(1, 1, 1),
            adsorbate="H",
            calculate_energies=True,
        )
        
        agent = MaterialsScienceAgent(config)
        result = agent.run()
        
        print(result.summary())
        ```
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration specifying material, surface,
                   adsorbate, and calculation parameters.
        """
        self.config = config
        self._workflow = None
        self._result = None
    
    def run(self) -> AgentResult:
        """
        Execute the complete adsorption analysis workflow.
        
        Returns:
            AgentResult containing all analysis data, including:
            - Material and slab information
            - Identified adsorption sites
            - Calculated energies (if requested)
            - Output file paths
        """
        self._workflow = AdsorptionWorkflow(self.config)
        self._result = self._workflow.run()
        return self._result
    
    @property
    def result(self) -> Optional[AgentResult]:
        """Get the last workflow result, if available."""
        return self._result
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of the results.
        
        Returns:
            Formatted summary string
        """
        if self._result is None:
            return "No results available. Run the agent first."
        return self._result.summary()
    
    def get_best_site(self) -> Optional[Dict[str, Any]]:
        """
        Get the best adsorption site from the last run.
        
        Returns:
            Dictionary with site information, or None
        """
        if self._result is None:
            return None
        return self._result.best_site
    
    def get_output_files(self) -> List[str]:
        """
        Get list of output files from the last run.
        
        Returns:
            List of file paths
        """
        if self._result is None:
            return []
        return self._result.output_files
    
    @staticmethod
    def check_installation() -> Dict[str, Any]:
        """
        Check the installation status of required dependencies.
        
        Returns:
            Dictionary with installation status
        """
        status = {
            "fairchem": check_fairchem_installation(),
            "available_adsorbates": get_available_adsorbates(),
            "available_models": get_available_models(),
        }
        
        # Check Materials Project API
        try:
            from mp_api.client import MPRester
            status["mp_api"] = True
        except ImportError:
            status["mp_api"] = False
        
        # Check pymatgen
        try:
            from pymatgen.core import Structure
            status["pymatgen"] = True
        except ImportError:
            status["pymatgen"] = False
        
        # Check ASE
        try:
            from ase.atoms import Atoms
            status["ase"] = True
        except ImportError:
            status["ase"] = False
        
        return status
    
    @staticmethod
    def list_adsorbates() -> List[str]:
        """
        List available adsorbate species.
        
        Returns:
            List of adsorbate names
        """
        return list_available_adsorbates()
    
    @staticmethod
    def list_models() -> List[str]:
        """
        List available FairChem models.
        
        Returns:
            List of model names
        """
        return get_available_models()


class InteractiveAgent:
    """
    Interactive version of the agent with step-by-step control.
    
    This class allows users to execute individual steps of the workflow
    and inspect intermediate results.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the interactive agent.
        
        Args:
            api_key: Materials Project API key (optional, uses env var if not provided)
        """
        self.mp_tool = MaterialsProjectTool(api_key=api_key)
        self._bulk_structure = None
        self._material_id = None
        self._slab = None
        self._terminations = None
        self._sites = None
        self._adsorbate = None
    
    def fetch_material(self, material: str) -> Dict[str, Any]:
        """
        Fetch a material from Materials Project.
        
        Args:
            material: Formula (e.g., 'Si') or MP ID (e.g., 'mp-149')
            
        Returns:
            Material properties dictionary
        """
        if material.startswith("mp-"):
            self._bulk_structure = self.mp_tool.get_structure_by_mp_id(material)
            self._material_id = material
        else:
            self._bulk_structure, self._material_id = self.mp_tool.get_structure_by_formula(material)
        
        props = self.mp_tool.get_material_properties(self._material_id)
        return props
    
    def list_terminations(
        self,
        miller: Tuple[int, int, int] = (1, 1, 1),
        min_slab_size: float = 10.0,
        vacuum: float = 15.0,
    ) -> List[Dict[str, Any]]:
        """
        List available surface terminations.
        
        Args:
            miller: Miller indices
            min_slab_size: Minimum slab thickness (Å)
            vacuum: Vacuum thickness (Å)
            
        Returns:
            List of termination info dictionaries
        """
        if self._bulk_structure is None:
            raise RuntimeError("Fetch a material first with fetch_material()")
        
        builder = SurfaceBuilder(self._bulk_structure, miller)
        self._terminations = builder.get_available_terminations(min_slab_size, vacuum)
        
        return [t.to_dict() for t in self._terminations]
    
    def create_slab(
        self,
        miller: Tuple[int, int, int] = (1, 1, 1),
        termination: int = 0,
        min_slab_size: float = 10.0,
        vacuum: float = 15.0,
        supercell: Tuple[int, int] = (1, 1),
    ) -> Dict[str, Any]:
        """
        Create a surface slab.
        
        Args:
            miller: Miller indices
            termination: Termination index (use list_terminations to see options)
            min_slab_size: Minimum slab thickness (Å)
            vacuum: Vacuum thickness (Å)
            supercell: In-plane supercell (nx, ny)
            
        Returns:
            Slab info dictionary
        """
        if self._bulk_structure is None:
            raise RuntimeError("Fetch a material first with fetch_material()")
        
        builder = SurfaceBuilder(self._bulk_structure, miller)
        self._slab = builder.build_slab(
            termination=termination,
            min_slab_size=min_slab_size,
            min_vacuum_size=vacuum,
            supercell=supercell,
        )
        
        return get_slab_info(self._slab)
    
    def find_sites(
        self,
        height_offset: float = 2.0,
        site_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find adsorption sites on the slab.
        
        Args:
            height_offset: Height above surface for adsorbate (Å)
            site_types: Types to include (default: all)
            
        Returns:
            List of site dictionaries
        """
        if self._slab is None:
            raise RuntimeError("Create a slab first with create_slab()")
        
        finder = AdsorptionSiteFinder(self._slab, height_offset=height_offset)
        sites = finder.find_all_sites()
        
        if site_types:
            sites = finder.filter_by_type(sites, site_types)
        
        sites = finder.remove_duplicates(sites)
        self._sites = sites
        
        return [s.to_dict() for s in sites]
    
    def select_adsorbate(self, species: str) -> Dict[str, Any]:
        """
        Select an adsorbate species.
        
        Args:
            species: Adsorbate name (e.g., 'H', 'CO', 'OH')
            
        Returns:
            Adsorbate info dictionary
        """
        self._adsorbate = create_adsorbate(species, height=0)
        
        return {
            "species": species,
            "n_atoms": len(self._adsorbate),
            "elements": list(set(self._adsorbate.get_chemical_symbols())),
        }
    
    def get_slab(self):
        """Get the current slab (ASE Atoms object)."""
        return self._slab
    
    def get_sites(self) -> List[AdsorptionSite]:
        """Get the current sites (AdsorptionSite objects)."""
        return self._sites or []
    
    def get_adsorbate(self):
        """Get the current adsorbate (ASE Atoms object)."""
        return self._adsorbate


def run_analysis(
    material: str,
    miller: Tuple[int, int, int] = (1, 1, 1),
    adsorbate: str = "H",
    output_dir: str = "outputs",
    calculate_energies: bool = False,
    **kwargs
) -> AgentResult:
    """
    Convenience function to run a complete analysis.
    
    Args:
        material: Material formula or MP ID
        miller: Miller indices
        adsorbate: Adsorbate species
        output_dir: Output directory
        calculate_energies: Whether to calculate adsorption energies
        **kwargs: Additional configuration options
        
    Returns:
        AgentResult with analysis data
        
    Example:
        ```python
        from agent_materials_science import run_analysis
        
        result = run_analysis(
            material="Pt",
            miller=(1, 1, 1),
            adsorbate="CO",
            calculate_energies=True,
        )
        
        print(f"Best site: {result.best_site}")
        ```
    """
    config = AgentConfig(
        material=material,
        miller_indices=miller,
        adsorbate=adsorbate,
        output_dir=output_dir,
        calculate_energies=calculate_energies,
        **kwargs
    )
    
    agent = MaterialsScienceAgent(config)
    return agent.run()
