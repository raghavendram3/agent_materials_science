"""
Complete workflow for adsorption site analysis.

This module orchestrates the full workflow:
1. Fetch material from Materials Project
2. Generate surface slab
3. Find adsorption sites
4. Calculate adsorption energies (optional)
5. Rank sites and identify best site
6. Save results
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

from ase.atoms import Atoms

from ..config import AgentConfig
from ..tools.materials_project import MaterialsProjectTool
from ..tools.ase_tools import save_outputs, get_slab_info
from ..tools.converters import create_adsorbate, place_adsorbate_at_site
from ..tools.fairchem_calc import (
    FairchemCalculator, FAIRCHEM_AVAILABLE,
    get_adsorbate_reference_energy,
)

from .surface import SurfaceBuilder, TerminationInfo, miller_to_string
from .adsorption import (
    AdsorptionSiteFinder, AdsorptionSite,
    find_sites_pymatgen, summarize_sites, PYMATGEN_ASF_AVAILABLE,
)


@dataclass
class WorkflowResult:
    """
    Results from the adsorption analysis workflow.
    
    Contains all information about the analysis including material data,
    slab properties, adsorption sites, and calculated energies.
    """
    # Material info
    material_id: str
    formula: str
    
    # Slab info
    miller_indices: Tuple[int, int, int]
    slab_info: Dict[str, Any]
    terminations: List[Dict[str, Any]]
    selected_termination: int
    
    # Adsorbate info
    adsorbate: str
    
    # Sites
    sites: List[Dict[str, Any]]
    best_site: Optional[Dict[str, Any]]
    site_summary: Dict[str, Any]
    
    # Energies (if calculated)
    energies_calculated: bool = False
    energy_data: Optional[Dict[str, float]] = None
    
    # Output files
    output_files: List[str] = field(default_factory=list)
    
    # Metadata
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Ensure miller indices are list for JSON
        d["miller_indices"] = list(self.miller_indices)
        return d
    
    def to_json(self, filepath: str) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        thickness = self.slab_info.get('thickness')
        thickness_str = f"{thickness:.2f} Å" if isinstance(thickness, (int, float)) else "? Å"
        lines = [
            "=" * 60,
            "ADSORPTION ANALYSIS RESULTS",
            "=" * 60,
            f"Material: {self.formula} ({self.material_id})",
            f"Surface: {self.miller_indices}",
            f"Adsorbate: {self.adsorbate}",
            "",
            f"Slab: {self.slab_info.get('n_atoms', '?')} atoms, "
            f"thickness: {thickness_str}",
            "",
            f"Total adsorption sites found: {len(self.sites)}",
        ]
        
        # Site breakdown
        if self.site_summary.get("sites_by_type"):
            lines.append("Sites by type:")
            for stype, count in self.site_summary["sites_by_type"].items():
                lines.append(f"  - {stype}: {count}")
        
        # Best site
        if self.best_site:
            lines.append("")
            lines.append("Best adsorption site:")
            lines.append(f"  Type: {self.best_site.get('site_type', 'unknown')}")
            pos = self.best_site.get("position", [0, 0, 0])
            lines.append(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            if self.best_site.get("energy") is not None:
                lines.append(f"  Adsorption energy: {self.best_site['energy']:.3f} eV")
        
        # Energy info
        if self.energies_calculated and self.energy_data:
            lines.append("")
            lines.append("Energy calculations:")
            if "e_slab" in self.energy_data:
                lines.append(f"  Clean slab: {self.energy_data['e_slab']:.3f} eV")
            if "e_adsorbate" in self.energy_data:
                lines.append(f"  Isolated adsorbate: {self.energy_data['e_adsorbate']:.3f} eV")
        
        # Output files
        if self.output_files:
            lines.append("")
            lines.append("Output files:")
            for f in self.output_files:
                lines.append(f"  - {f}")
        
        # Warnings/errors
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class AdsorptionWorkflow:
    """
    Complete workflow for automated adsorption site analysis.
    
    This class orchestrates all steps from fetching a material from
    Materials Project to calculating adsorption energies and saving results.
    """
    
    def __init__(
        self,
        config: AgentConfig,
    ):
        """
        Initialize the workflow.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.mp_tool = MaterialsProjectTool(api_key=config.mp_api_key)
        
        # State
        self._bulk_structure = None
        self._slab = None
        self._sites = []
        self._adsorbate = None
        self._calculator = None
        # Relaxed slab+adsorbate structures, keyed by id(site).
        self._relaxed_structures: Dict[int, Atoms] = {}
        
        # Logging
        self.errors = []
        self.warnings = []
    
    def log(self, msg: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.config.verbose:
            print(msg)

    def _get_calculator(self) -> FairchemCalculator:
        """Lazily create (and cache) the FairChem calculator."""
        if self._calculator is None:
            self._calculator = FairchemCalculator(
                model_name=self.config.fairchem_model,
                device=self.config.device,
                task_name=self.config.fairchem_task,
            )
        return self._calculator
    
    def run(self) -> WorkflowResult:
        """
        Execute the complete workflow.
        
        Returns:
            WorkflowResult with all analysis data
        """
        self.log("=" * 60)
        self.log("Starting Adsorption Analysis Workflow")
        self.log("=" * 60)
        
        # Step 1: Fetch bulk structure
        self.log("\n[1/6] Fetching bulk structure from Materials Project...")
        bulk_structure, material_id = self._fetch_structure()
        formula = bulk_structure.composition.reduced_formula
        self.log(f"  ✓ Retrieved {formula} (ID: {material_id})")
        
        # Step 2: Create surface slab
        self.log(f"\n[2/6] Creating surface slab {self.config.miller_indices}...")
        slab, terminations = self._create_slab(bulk_structure)
        slab_info = get_slab_info(slab)
        self.log(f"  ✓ Generated slab with {slab_info['n_atoms']} atoms")
        self.log(f"    Thickness: {slab_info['thickness']:.2f} Å")
        self.log(f"    Available terminations: {len(terminations)}")

        # Step 2b: Pre-relax the clean slab so that (a) adsorption sites are
        # found on the relaxed surface geometry and (b) E_slab in
        # E_ads = E(slab+ads) - E(slab) - E(ads) is computed from the same
        # relaxed reference as the combined system. (Previously only the
        # slab+adsorbate system was relaxed, which systematically
        # over-estimated binding by the slab-relaxation energy.)
        if (
            self.config.calculate_energies
            and self.config.relax_structures
            and FAIRCHEM_AVAILABLE
        ):
            self.log("    Relaxing clean slab (consistent E_ads reference)...")
            try:
                calc = self._get_calculator()
                slab = calc.relax_structure(
                    slab,
                    fmax=self.config.relax_fmax,
                    steps=self.config.relax_steps,
                )
                relax_info = slab.info.get("relaxation", {})
                self.log(
                    f"    ✓ Clean slab relaxed in "
                    f"{relax_info.get('steps', '?')} steps "
                    f"(converged: {relax_info.get('converged', '?')})"
                )
                slab_info = get_slab_info(slab)
            except Exception as e:
                self.warnings.append(f"Clean-slab relaxation failed: {e}")
                self.log(f"    ! Clean-slab relaxation failed: {e}")

        # Step 3: Find adsorption sites
        self.log(f"\n[3/6] Finding adsorption sites...")
        sites, site_summary = self._find_sites(slab)
        self.log(f"  ✓ Found {len(sites)} adsorption sites")
        for stype, count in site_summary.get("sites_by_type", {}).items():
            self.log(f"    - {stype}: {count}")
        
        # Step 4: Create adsorbate
        self.log(f"\n[4/6] Creating adsorbate: {self.config.adsorbate}...")
        adsorbate = create_adsorbate(self.config.adsorbate, height=0)
        self.log(f"  ✓ Created {self.config.adsorbate} ({len(adsorbate)} atoms)")
        
        # Step 5: Calculate energies (if requested)
        energy_data = None
        if self.config.calculate_energies:
            self.log(f"\n[5/6] Calculating adsorption energies with FairChem...")
            if FAIRCHEM_AVAILABLE:
                sites, energy_data = self._calculate_energies(slab, sites, adsorbate)
                self.log(f"  ✓ Completed energy calculations")
            else:
                self.warnings.append("FairChem not available - skipping energy calculations")
                self.log(f"  ! FairChem not available - skipping")
        else:
            self.log(f"\n[5/6] Skipping energy calculations (not requested)")
        
        # Step 6: Rank sites and get best
        self.log(f"\n[6/6] Ranking sites and saving results...")
        sites_ranked = self._rank_sites(sites)
        best_site = sites_ranked[0] if sites_ranked else None
        
        if best_site:
            self.log(f"  ✓ Best site: {best_site.site_type}")
            if best_site.energy is not None:
                self.log(f"    Adsorption energy: {best_site.energy:.3f} eV")
        
        # Save outputs
        output_files = self._save_outputs(
            slab, sites_ranked, best_site, adsorbate, material_id
        )
        self.log(f"  ✓ Saved {len(output_files)} output files")
        
        # Create result
        result = WorkflowResult(
            material_id=material_id,
            formula=formula,
            miller_indices=self.config.miller_indices,
            slab_info=slab_info,
            terminations=[t.to_dict() for t in terminations],
            selected_termination=self.config.termination,
            adsorbate=self.config.adsorbate,
            sites=[s.to_dict() for s in sites_ranked],
            best_site=best_site.to_dict() if best_site else None,
            site_summary=site_summary,
            energies_calculated=self.config.calculate_energies and energy_data is not None,
            energy_data=energy_data,
            output_files=output_files,
            errors=self.errors,
            warnings=self.warnings,
        )
        
        self.log("\n" + "=" * 60)
        self.log("Workflow completed!")
        self.log("=" * 60)
        
        return result
    
    def _fetch_structure(self):
        """Fetch bulk structure from Materials Project."""
        try:
            if self.config.mp_id:
                structure = self.mp_tool.get_structure_by_mp_id(self.config.mp_id)
                return structure, self.config.mp_id
            elif self.config.material:
                structure, mp_id = self.mp_tool.get_structure_by_formula(
                    self.config.material
                )
                return structure, mp_id
            else:
                raise ValueError("Must provide either material formula or MP ID")
        except Exception as e:
            self.errors.append(f"Failed to fetch structure: {e}")
            raise
    
    def _create_slab(self, bulk_structure) -> Tuple[Atoms, List[TerminationInfo]]:
        """Create surface slab."""
        try:
            builder = SurfaceBuilder(bulk_structure, self.config.miller_indices)

            # Slab thickness: interpret n_layers as (hkl) planes when
            # layers_in_unit_planes=True (pymatgen in_unit_planes); otherwise
            # fall back to the legacy 2.5 Å/layer heuristic.
            in_unit_planes = self.config.layers_in_unit_planes
            min_slab_size = (
                self.config.n_layers if in_unit_planes
                else self.config.n_layers * 2.5
            )

            # Get terminations
            terminations = builder.get_available_terminations(
                min_slab_size=min_slab_size,
                min_vacuum_size=self.config.vacuum,
                in_unit_planes=in_unit_planes,
                center_slab=self.config.center_slab,
            )

            # Build slab
            slab = builder.build_slab(
                termination=self.config.termination,
                min_slab_size=min_slab_size,
                min_vacuum_size=self.config.vacuum,
                supercell=self.config.supercell,
                fix_layers=self.config.fix_layers,
                in_unit_planes=in_unit_planes,
                center_slab=self.config.center_slab,
            )

            return slab, terminations
            
        except Exception as e:
            self.errors.append(f"Failed to create slab: {e}")
            raise
    
    def _find_sites(self, slab: Atoms) -> Tuple[List[AdsorptionSite], Dict]:
        """Find adsorption sites on slab using the configured backend."""
        backend = self.config.site_finder
        use_pymatgen = (
            backend == "pymatgen"
            or (backend == "auto" and PYMATGEN_ASF_AVAILABLE)
        )

        if backend == "pymatgen" and not PYMATGEN_ASF_AVAILABLE:
            self.warnings.append(
                "site_finder='pymatgen' requested but pymatgen's "
                "AdsorbateSiteFinder is unavailable; using built-in finder."
            )
            use_pymatgen = False

        try:
            if use_pymatgen:
                self.log("    Using pymatgen AdsorbateSiteFinder (symmetry-aware)")
                # symm_reduce/near_reduce are pymatgen tolerances; 0 disables.
                symm_tol = 0.01 if self.config.symm_reduce else 0.0
                unique_sites = find_sites_pymatgen(
                    slab,
                    height_offset=self.config.height_offset,
                    site_types=self.config.site_types,
                    symm_reduce=symm_tol,
                )
                # Count surface atoms for the summary (top-layer heuristic).
                z = slab.get_positions()[:, 2]
                n_surface = int(np.sum((z.max() - z) < 1.5))
                summary = summarize_sites(unique_sites, n_surface)
                return unique_sites, summary

            finder = AdsorptionSiteFinder(
                slab,
                height_offset=self.config.height_offset,
            )

            # Find all sites
            all_sites = finder.find_all_sites()

            # Filter by type
            if self.config.site_types:
                all_sites = finder.filter_by_type(all_sites, self.config.site_types)

            # Remove duplicates
            unique_sites = finder.remove_duplicates(all_sites)

            # Get summary
            summary = finder.get_site_summary(unique_sites)

            return unique_sites, summary

        except Exception as e:
            self.errors.append(f"Failed to find sites: {e}")
            raise
    
    def _calculate_energies(
        self,
        slab: Atoms,
        sites: List[AdsorptionSite],
        adsorbate: Atoms,
    ) -> Tuple[List[AdsorptionSite], Dict[str, float]]:
        """
        Calculate adsorption energies for sites using FairChem ML potential.
        
        E_ads = E(slab+ads) - E(slab) - E_gas(ads)

        The gas-phase reference E_gas follows the OC20 convention (linear
        combination of per-element reference energies), which is the scheme
        recommended by FairChem for the UMA 'oc20' task. When
        relax_structures=True, the incoming slab has already been relaxed
        (see run()), so E_slab and E_combined are computed consistently.
        """
        try:
            calc = self._get_calculator()

            # Clean slab energy (single point; slab is pre-relaxed when
            # relax_structures=True).
            self.log(f"    Calculating clean slab energy...")
            e_slab = calc.calculate_energy(slab)
            self.log(f"    Clean slab energy: {e_slab:.3f} eV")

            # Gas-phase reference energy (OC20 linear atomic scheme).
            adsorbate_name = self.config.adsorbate
            reference_details: Dict[str, Any] = {}
            try:
                e_ads, reference_details = get_adsorbate_reference_energy(
                    adsorbate_name
                )
                reference_used = True
                self.log(
                    f"    Gas-phase reference ({adsorbate_name}, OC20 "
                    f"convention): {e_ads:.3f} eV"
                )
                self.warnings.append(
                    "Adsorption energies follow the OC20 convention: E_gas is "
                    "a linear combination of per-element reference energies "
                    "(H/C/O/N from the OC20 paper), consistent with UMA "
                    f"'{self.config.fairchem_task}' total energies. Remaining "
                    "uncertainty comes from the ML potential itself and from "
                    "heuristic initial placements; for production-grade "
                    "minima consider the AdsorbML workflow "
                    "(pip install fairchem-core[adsorbml])."
                )
            except ValueError as err:
                reference_used = False
                e_ads = 0.0
                self.warnings.append(
                    f"{err} Using e_ads=0: energies are RELATIVE (site "
                    f"ranking is still valid)."
                )
                self.log(f"    ! {err}")
                self.log(f"    Using e_ads = 0 (relative energies only)")

            # Energy for each site.
            n_sites = len(sites)
            for i, site in enumerate(sites):
                self.log(f"    Calculating site {i+1}/{n_sites} ({site.site_type})...")

                # Place adsorbate at site
                slab_with_ads = place_adsorbate_at_site(
                    slab, adsorbate, site.position
                )

                # Optionally relax structure
                if self.config.relax_structures:
                    self.log(f"      Relaxing structure...")
                    slab_with_ads = calc.relax_structure(
                        slab_with_ads,
                        fmax=self.config.relax_fmax,
                        steps=self.config.relax_steps,
                    )
                    self._relaxed_structures[id(site)] = slab_with_ads
                    relax_info = slab_with_ads.info.get("relaxation", {})
                    site.metadata["relaxation"] = relax_info

                # Energy of the slab+adsorbate system
                e_combined = calc.calculate_energy(slab_with_ads)

                # Adsorption energy: E_ads = E(slab+ads) - E(slab) - E(gas)
                site.energy = e_combined - e_slab - e_ads
                self.log(f"      E_combined: {e_combined:.3f} eV, E_ads: {site.energy:.3f} eV")

            energy_data = {
                "e_slab": e_slab,
                "e_adsorbate_reference": e_ads,
                "adsorbate_name": adsorbate_name,
                "reference_used": reference_used,
                "reference_details": reference_details,
                "model": self.config.fairchem_model,
                "task": self.config.fairchem_task,
                "relaxed": self.config.relax_structures,
                "relax_fmax": self.config.relax_fmax if self.config.relax_structures else None,
            }

            return sites, energy_data

        except Exception as e:
            self.errors.append(f"Energy calculation failed: {e}")
            self.warnings.append("Continuing without energies")
            return sites, None
    
    def _rank_sites(self, sites: List[AdsorptionSite]) -> List[AdsorptionSite]:
        """Rank sites by energy (if available) or coordination."""
        if not sites:
            return []
        
        # Check if energies are available
        has_energies = any(s.energy is not None for s in sites)
        
        if has_energies:
            # Sort by energy (lowest first = most favorable)
            sites_with_e = [s for s in sites if s.energy is not None]
            sites_without_e = [s for s in sites if s.energy is None]
            return sorted(sites_with_e, key=lambda s: s.energy) + sites_without_e
        else:
            # Sort by coordination (highest first)
            return sorted(sites, key=lambda s: len(s.coordinating_atoms), reverse=True)
    
    def _save_outputs(
        self,
        slab: Atoms,
        sites: List[AdsorptionSite],
        best_site: Optional[AdsorptionSite],
        adsorbate: Atoms,
        material_id: str,
    ) -> List[str]:
        """Save output files."""
        files = []
        
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Base name for files
            miller_str = miller_to_string(self.config.miller_indices)
            base_name = f"{material_id}_{miller_str}"
            
            # Save clean slab
            slab_files = save_outputs(slab, self.config.output_dir, f"{base_name}_slab")
            files.extend(slab_files)
            
            # Save best site structure (slab + adsorbate). Use the relaxed
            # geometry when the reported energy came from a relaxation, so
            # the saved structure matches the reported number.
            if best_site:
                slab_with_ads = self._relaxed_structures.get(id(best_site))
                if slab_with_ads is None:
                    slab_with_ads = place_adsorbate_at_site(
                        slab, adsorbate, best_site.position
                    )
                best_files = save_outputs(
                    slab_with_ads, 
                    self.config.output_dir, 
                    f"{base_name}_best_{self.config.adsorbate}"
                )
                files.extend(best_files)
            
            # Save sites JSON
            sites_file = os.path.join(self.config.output_dir, f"{base_name}_sites.json")
            sites_data = {
                "n_sites": len(sites),
                "adsorbate": self.config.adsorbate,
                "sites": [s.to_dict() for s in sites],
            }
            with open(sites_file, 'w') as f:
                json.dump(sites_data, f, indent=2, default=str)
            files.append(sites_file)
            
            # Save all site structures if requested
            if self.config.save_all_sites and sites:
                sites_dir = os.path.join(self.config.output_dir, "all_sites")
                os.makedirs(sites_dir, exist_ok=True)
                
                for i, site in enumerate(sites):
                    slab_with_ads = self._relaxed_structures.get(id(site))
                    if slab_with_ads is None:
                        slab_with_ads = place_adsorbate_at_site(
                            slab, adsorbate, site.position
                        )
                    site_name = f"{base_name}_site{i}_{site.site_type}"
                    site_files = save_outputs(slab_with_ads, sites_dir, site_name)
                    files.extend(site_files)
            
            return files
            
        except Exception as e:
            self.errors.append(f"Failed to save outputs: {e}")
            return files


def run_quick_analysis(
    material: str,
    miller: Tuple[int, int, int] = (1, 1, 1),
    adsorbate: str = "H",
    output_dir: str = "outputs",
    calculate_energies: bool = False,
) -> WorkflowResult:
    """
    Quick analysis with default parameters.
    
    Args:
        material: Material formula or MP ID
        miller: Miller indices
        adsorbate: Adsorbate species
        output_dir: Output directory
        calculate_energies: Whether to calculate energies
        
    Returns:
        WorkflowResult
    """
    config = AgentConfig(
        material=material,
        miller_indices=miller,
        adsorbate=adsorbate,
        output_dir=output_dir,
        calculate_energies=calculate_energies,
    )
    
    workflow = AdsorptionWorkflow(config)
    return workflow.run()
