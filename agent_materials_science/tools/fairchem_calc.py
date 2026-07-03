"""
FairChem (Meta FAIR Chemistry) integration for ML-based energy calculations.

This module wraps FairChem's Universal Model for Atoms (UMA) for fast,
ML-accelerated DFT-quality calculations of adsorption energies.

Requires fairchem-core >= 2.x (Python >= 3.11, torch ~= 2.8). UMA checkpoints
are gated on Hugging Face: create an account, request access to the
facebook/UMA repository, and run `huggingface-cli login` once.

Adsorption-energy convention
----------------------------
    E_ads = E(slab+ads) - E(slab) - E_gas(adsorbate)

Following the OC20 dataset convention (Chanussot et al., ACS Catal. 2021,
Table 5) and the official FairChem UMA tutorial, the gas-phase reference
E_gas is a *linear combination of per-element reference energies* derived
from N2, H2O, CO and H2 at the OC20 (RPBE) level of theory:

    H: -3.477 eV    C: -7.282 eV    O: -7.204 eV    N: -8.083 eV

Because the UMA 'oc20' task head predicts RPBE *total* energies for
slab+adsorbate systems, combining those totals with these tabulated atomic
references yields adsorption energies in the standard OC20 convention. The
FairChem documentation explicitly recommends this scheme and advises against
computing gas-phase references with the 'omol' head (different level of
theory, no error cancellation).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.formula import Formula

logger = logging.getLogger(__name__)

# Try to import fairchem (v2 API).
try:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    FAIRCHEM_AVAILABLE = True
    FAIRCHEM_IMPORT_ERROR: Optional[str] = None
except ImportError as _err:  # pragma: no cover - depends on environment
    FAIRCHEM_AVAILABLE = False
    FAIRCHEM_IMPORT_ERROR = str(_err)
    FAIRChemCalculator = None  # type: ignore[assignment]
    pretrained_mlip = None  # type: ignore[assignment]

# Kept for backward compatibility with code that imported this flag; the
# legacy (v1) API is no longer supported.
FAIRCHEM_NEW_API = FAIRCHEM_AVAILABLE

DEFAULT_MODEL = "uma-s-1p2"

# Per-element gas-phase reference energies (eV) in the OC20 convention.
# Source: OC20 paper (arXiv:2010.09990) Table 5; reproduced in the official
# FairChem UMA tutorial for adsorption energies with task_name='oc20'.
OC20_ATOMIC_REFERENCE_ENERGIES: Dict[str, float] = {
    "H": -3.477,
    "C": -7.282,
    "O": -7.204,
    "N": -8.083,
}


def get_adsorbate_reference_energy(
    adsorbate: Union[str, Atoms],
    overrides: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Gas-phase reference energy for an adsorbate in the OC20 convention.

    E_gas = sum over atoms of the per-element reference energy. Works for any
    species composed of C, H, O and N (plus any elements supplied via
    ``overrides``).

    Args:
        adsorbate: Species name (e.g. 'CO', 'OH', 'H2O') or an ASE Atoms
            object whose composition will be used.
        overrides: Optional {element: energy_eV} entries that extend or
            replace the built-in OC20 per-element references (e.g. to add
            'S' from your own reference calculation).

    Returns:
        (reference_energy_eV, details) where details records the composition,
        per-element contributions and the scheme name.

    Raises:
        ValueError: If the adsorbate contains an element with no reference
            energy available.
    """
    refs = dict(OC20_ATOMIC_REFERENCE_ENERGIES)
    if overrides:
        refs.update(overrides)

    if isinstance(adsorbate, Atoms):
        symbols = adsorbate.get_chemical_symbols()
        composition = {el: symbols.count(el) for el in sorted(set(symbols))}
        name = adsorbate.get_chemical_formula()
    else:
        name = str(adsorbate)
        composition = dict(Formula(name).count())

    missing = sorted(el for el in composition if el not in refs)
    if missing:
        raise ValueError(
            f"No OC20 gas-phase reference energy for element(s) {missing} in "
            f"'{name}'. Built-in references cover "
            f"{sorted(OC20_ATOMIC_REFERENCE_ENERGIES)}; supply values for "
            f"other elements via `overrides` (computed with settings "
            f"consistent with the oc20 task)."
        )

    per_element = {el: n * refs[el] for el, n in composition.items()}
    energy = float(sum(per_element.values()))
    return energy, {
        "adsorbate": name,
        "composition": composition,
        "per_element_energy_eV": per_element,
        "scheme": "oc20_linear_atomic_reference",
    }


def _legacy_reference_table() -> Dict[str, float]:
    """Molecule-keyed reference table derived from the OC20 atomic scheme."""
    table: Dict[str, float] = {}
    for species in (
        "H", "H2", "O", "O2", "C", "N", "N2", "CO", "CO2",
        "OH", "H2O", "CH4", "NH3", "NO", "HCOO",
    ):
        try:
            table[species], _ = get_adsorbate_reference_energy(species)
        except ValueError:  # pragma: no cover - all listed species are CHON
            pass
    return table


# Backward-compatible name. Values are now derived from the OC20 atomic
# scheme instead of the previous ad-hoc constants (the old table disagreed
# with the OC20 convention by up to ~2.3 eV for O, and its OH entry was an
# acknowledged copy-paste placeholder).
ADSORBATE_REFERENCE_ENERGIES: Dict[str, float] = _legacy_reference_table()


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto'/'cpu'/'cuda' to a concrete torch device string."""
    device = (device or "auto").lower()
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"
    return device


class FairchemCalculator:
    """
    Wrapper around FairChem's ASE calculator for ML-accelerated energies,
    forces and structure relaxations at near-DFT accuracy.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        task_name: str = "oc20",
        cpu: Optional[bool] = None,
        inference_settings: str = "default",
    ):
        """
        Initialize FairChem calculator.

        Args:
            model_name: Pretrained model name (see
                ``fairchem.core.calculate.pretrained_mlip.available_models``),
                e.g. 'uma-s-1p2' (default), 'uma-s-1p1', 'uma-m-1p1'.
                NOTE: 'uma-s-1' was removed from the FairChem registry.
            checkpoint_path: Optional path to a local checkpoint file. Takes
                precedence over ``model_name``.
            device: 'auto' (CUDA if available), 'cpu' or 'cuda'.
            task_name: UMA task head. For surface adsorption / catalysis use
                'oc20'. Use 'omat' only for bulk inorganic materials and
                'omol' for isolated molecules; mixing task heads between the
                slab and the adsorbate reference makes the resulting
                adsorption energies physically meaningless.
            cpu: Deprecated boolean alias for ``device`` (True -> 'cpu',
                False -> 'cuda'). Ignored when ``device`` is not 'auto'.
            inference_settings: FairChem inference settings, 'default' or
                'turbo' (faster, but requires a fixed atomic composition
                across calls - suitable for relaxing one system, not for
                looping over different sites/adsorbates).
        """
        if not FAIRCHEM_AVAILABLE:
            raise ImportError(
                "fairchem-core is not installed (or failed to import: "
                f"{FAIRCHEM_IMPORT_ERROR}).\n"
                "Install it with:  pip install fairchem-core\n"
                "Requirements: Python >= 3.11, torch ~= 2.8. UMA checkpoints "
                "additionally require Hugging Face access to facebook/UMA "
                "(`huggingface-cli login`)."
            )

        if cpu is not None:
            warnings.warn(
                "FairchemCalculator(cpu=...) is deprecated; use device='cpu'/'cuda'.",
                DeprecationWarning,
                stacklevel=2,
            )
            if device == "auto":
                device = "cpu" if cpu else "cuda"

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.task_name = task_name
        self.inference_settings = inference_settings
        self._calculator: Optional[Calculator] = None

    def get_calculator(self) -> Calculator:
        """
        Get or create the underlying ASE calculator (cached).

        Returns:
            ASE Calculator backed by the requested FairChem model.
        """
        if self._calculator is not None:
            return self._calculator

        device = resolve_device(self.device)
        name_or_path = self.checkpoint_path or self.model_name

        try:
            if hasattr(FAIRChemCalculator, "from_model_checkpoint"):
                # Modern entry point (fairchem-core >= ~2.2): accepts either a
                # registry model name or a local checkpoint path.
                self._calculator = FAIRChemCalculator.from_model_checkpoint(
                    name_or_path,
                    task_name=self.task_name,
                    inference_settings=self.inference_settings,
                    device=device,
                )
            else:  # pragma: no cover - older 2.x fallback
                if self.checkpoint_path:
                    from fairchem.core.units.mlip_unit import load_predict_unit

                    predictor = load_predict_unit(self.checkpoint_path, device=device)
                else:
                    predictor = pretrained_mlip.get_predict_unit(
                        self.model_name, device=device
                    )
                self._calculator = FAIRChemCalculator(
                    predictor, task_name=self.task_name
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize FairChem calculator for "
                f"'{name_or_path}' (task='{self.task_name}', device='{device}'): {e}\n"
                f"Available models: {get_available_models()}\n"
                f"If this is a gated UMA model, make sure you have requested "
                f"access on Hugging Face and run `huggingface-cli login`."
            ) from e

        return self._calculator

    # ------------------------------------------------------------------
    # Single-point properties
    # ------------------------------------------------------------------
    def calculate_energy(self, atoms: Atoms) -> float:
        """
        Potential energy of a periodic (slab or bulk) structure, in eV.

        The structure is treated as fully periodic, matching how OC20
        slab+adsorbate systems are represented (the vacuum region prevents
        spurious interactions along the surface normal, provided it exceeds
        the model's 6 A graph cutoff).
        """
        atoms_copy = atoms.copy()
        atoms_copy.set_pbc([True, True, True])
        atoms_copy.calc = self.get_calculator()
        return float(atoms_copy.get_potential_energy())

    def calculate_gas_energy(self, atoms: Atoms) -> float:
        """
        Potential energy of an isolated (non-periodic) molecule or atom, eV.

        fairchem-core >= 2.21 natively handles single isolated atoms
        (pbc=False) via bundled DFT atomic references. Note that for
        adsorption energies with the 'oc20' task you should normally use
        :func:`get_adsorbate_reference_energy` instead of computing gas-phase
        molecules directly - isolated molecules are out-of-domain for the
        oc20 head.
        """
        atoms_copy = atoms.copy()
        atoms_copy.set_pbc(False)
        atoms_copy.calc = self.get_calculator()
        return float(atoms_copy.get_potential_energy())

    def calculate_forces(self, atoms: Atoms) -> np.ndarray:
        """Forces on atoms, shape (N, 3), in eV/A."""
        atoms_copy = atoms.copy()
        atoms_copy.set_pbc([True, True, True])
        atoms_copy.calc = self.get_calculator()
        return atoms_copy.get_forces()

    # ------------------------------------------------------------------
    # Relaxation
    # ------------------------------------------------------------------
    def relax_structure(
        self,
        atoms: Atoms,
        fmax: float = 0.05,
        steps: int = 200,
        optimizer: str = "LBFGS",
    ) -> Atoms:
        """
        Relax a structure with the FairChem calculator.

        Constraints on the input (e.g. FixAtoms on the bottom slab layers)
        are preserved and honored by the optimizer.

        Args:
            atoms: Structure to relax.
            fmax: Force convergence criterion (eV/A).
            steps: Maximum optimizer steps.
            optimizer: 'LBFGS', 'BFGS' or 'FIRE'.

        Returns:
            Relaxed Atoms (a copy; the input is not modified). The relaxed
            energy is available via ``atoms.get_potential_energy()`` and
            convergence info in ``atoms.info['relaxation']``.
        """
        from ase.optimize import BFGS, FIRE, LBFGS

        optimizers = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}
        if optimizer not in optimizers:
            raise ValueError(
                f"Unknown optimizer: {optimizer}. Use one of {list(optimizers)}"
            )

        atoms_copy = atoms.copy()
        atoms_copy.set_pbc([True, True, True])
        atoms_copy.calc = self.get_calculator()

        opt = optimizers[optimizer](atoms_copy, logfile=None)
        converged = opt.run(fmax=fmax, steps=steps)
        atoms_copy.info["relaxation"] = {
            "converged": bool(converged),
            "steps": int(opt.get_number_of_steps()),
            "fmax_target": float(fmax),
        }
        if not converged:
            logger.warning(
                "Relaxation did not converge to fmax=%.3f eV/A within %d steps.",
                fmax,
                steps,
            )
        return atoms_copy

    # ------------------------------------------------------------------
    # Adsorption energy
    # ------------------------------------------------------------------
    def calculate_adsorption_energy(
        self,
        slab_with_adsorbate: Atoms,
        clean_slab: Atoms,
        adsorbate: Optional[Atoms] = None,
        adsorbate_name: Optional[str] = None,
        relax: bool = False,
        fmax: float = 0.05,
        steps: int = 200,
        use_reference_energy: bool = True,
        reference_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Adsorption energy  E_ads = E(slab+ads) - E(slab) - E_gas(ads).

        A negative value indicates favorable binding. When ``relax=True``
        BOTH the combined system and the clean slab are relaxed, so the two
        totals are computed consistently.

        Args:
            slab_with_adsorbate: Slab with the adsorbate placed on it.
            clean_slab: Clean slab without adsorbate.
            adsorbate: Isolated adsorbate (used to infer composition when
                ``adsorbate_name`` is not given, or for a direct gas-phase
                calculation when ``use_reference_energy=False``).
            adsorbate_name: Species name for the reference lookup
                (e.g. 'H', 'CO', 'H2O').
            relax: Relax both systems before the energy evaluation.
            fmax: Relaxation force criterion (eV/A).
            steps: Maximum relaxation steps.
            use_reference_energy: Use the OC20 linear atomic reference for
                E_gas (recommended for the 'oc20' task). When False, E_gas is
                computed directly from ``adsorbate`` with the current model
                (out-of-domain for 'oc20'; interpret with care).
            reference_overrides: Extra {element: energy_eV} references for
                elements outside C/H/O/N.

        Returns:
            Dictionary with 'adsorption_energy', 'e_combined', 'e_slab',
            'e_adsorbate', 'reference_used', 'reference_details', and, when
            ``relax=True``, the relaxed structures under
            'relaxed_slab_with_adsorbate' / 'relaxed_clean_slab'.
        """
        relaxed_combined = relaxed_clean = None
        if relax:
            relaxed_combined = self.relax_structure(
                slab_with_adsorbate, fmax=fmax, steps=steps
            )
            relaxed_clean = self.relax_structure(clean_slab, fmax=fmax, steps=steps)
            e_combined = float(relaxed_combined.get_potential_energy())
            e_slab = float(relaxed_clean.get_potential_energy())
        else:
            e_combined = self.calculate_energy(slab_with_adsorbate)
            e_slab = self.calculate_energy(clean_slab)

        # Gas-phase reference
        reference_used = False
        reference_details: Dict[str, Any] = {}
        e_ads: Optional[float] = None

        ref_target: Optional[Union[str, Atoms]] = adsorbate_name or adsorbate
        if use_reference_energy and ref_target is not None:
            try:
                e_ads, reference_details = get_adsorbate_reference_energy(
                    ref_target, overrides=reference_overrides
                )
                reference_used = True
            except ValueError as err:
                logger.warning("%s Attempting direct gas-phase calculation.", err)

        if e_ads is None:
            if adsorbate is None:
                raise ValueError(
                    "Provide an adsorbate Atoms object or an adsorbate_name "
                    "composed of C/H/O/N (or supply reference_overrides)."
                )
            e_ads = self.calculate_gas_energy(adsorbate)
            reference_details = {
                "scheme": "direct_ml_gas_phase",
                "note": (
                    "Computed with the current model/task on an isolated "
                    "molecule; out-of-domain for the 'oc20' head."
                ),
            }

        result: Dict[str, Any] = {
            "adsorption_energy": e_combined - e_slab - e_ads,
            "e_combined": e_combined,
            "e_slab": e_slab,
            "e_adsorbate": e_ads,
            "reference_used": reference_used,
            "reference_details": reference_details,
        }
        if relax:
            result["relaxed_slab_with_adsorbate"] = relaxed_combined
            result["relaxed_clean_slab"] = relaxed_clean
        return result


def get_available_models() -> List[str]:
    """
    List available FairChem model names.

    Uses the live registry when fairchem-core is installed, so the list
    tracks upstream additions/removals (e.g. 'uma-s-1' was removed).
    """
    if FAIRCHEM_AVAILABLE:
        try:
            return list(pretrained_mlip.available_models)
        except Exception:  # pragma: no cover
            pass
    return [
        "uma-s-1p2",
        "uma-s-1p1",
        "uma-m-1p1",
        "esen-md-direct-all-omol",
        "esen-sm-conserving-all-omol",
        "esen-sm-direct-all-omol",
        "esen-sm-conserving-all-oc25",
        "esen-md-direct-all-oc25",
    ]


def check_fairchem_installation() -> Dict[str, Any]:
    """Check FairChem installation status."""
    status: Dict[str, Any] = {
        "fairchem_available": FAIRCHEM_AVAILABLE,
        "new_api": FAIRCHEM_NEW_API,
        "error": FAIRCHEM_IMPORT_ERROR,
    }

    try:
        from importlib.metadata import version

        status["fairchem_version"] = version("fairchem-core")
    except Exception:
        status["fairchem_version"] = None

    try:
        import torch

        status["cuda_available"] = torch.cuda.is_available()
        status["torch_version"] = torch.__version__
    except ImportError:
        status["cuda_available"] = False
        status["torch_version"] = None

    return status


def create_calculator(
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    task_name: str = "oc20",
) -> Optional[Calculator]:
    """
    Create a FairChem ASE calculator instance, or None if unavailable.
    """
    if not FAIRCHEM_AVAILABLE:
        return None
    try:
        calc = FairchemCalculator(model_name=model, device=device, task_name=task_name)
        return calc.get_calculator()
    except Exception as e:  # pragma: no cover - depends on environment
        logger.warning("Could not create FairChem calculator: %s", e)
        return None
