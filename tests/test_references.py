"""Tests for the OC20-convention gas-phase reference energies."""

import pytest
from ase import Atoms

from agent_materials_science.tools.fairchem_calc import (
    ADSORBATE_REFERENCE_ENERGIES,
    OC20_ATOMIC_REFERENCE_ENERGIES,
    get_adsorbate_reference_energy,
)


class TestCompositionalReferences:
    def test_atomic_values_match_oc20_paper(self):
        assert OC20_ATOMIC_REFERENCE_ENERGIES["H"] == pytest.approx(-3.477)
        assert OC20_ATOMIC_REFERENCE_ENERGIES["C"] == pytest.approx(-7.282)
        assert OC20_ATOMIC_REFERENCE_ENERGIES["O"] == pytest.approx(-7.204)
        assert OC20_ATOMIC_REFERENCE_ENERGIES["N"] == pytest.approx(-8.083)

    def test_molecule_is_sum_of_atoms(self):
        e_co2, details = get_adsorbate_reference_energy("CO2")
        assert e_co2 == pytest.approx(-7.282 + 2 * (-7.204))
        assert details["composition"] == {"C": 1, "O": 2}
        assert details["scheme"] == "oc20_linear_atomic_reference"

    def test_oh_is_not_a_placeholder(self):
        """The old table used the N value (-8.32) as an admitted placeholder
        for OH; the compositional scheme computes it properly."""
        e_oh, _ = get_adsorbate_reference_energy("OH")
        assert e_oh == pytest.approx(-7.204 + -3.477)
        assert e_oh != pytest.approx(OC20_ATOMIC_REFERENCE_ENERGIES["N"])

    def test_accepts_atoms_object(self):
        water = Atoms("H2O")
        e, details = get_adsorbate_reference_energy(water)
        assert e == pytest.approx(2 * (-3.477) + (-7.204))
        assert details["composition"] == {"H": 2, "O": 1}

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="Cl"):
            get_adsorbate_reference_energy("HCl")

    def test_overrides_extend_coverage(self):
        e, _ = get_adsorbate_reference_energy("HCl", overrides={"Cl": -5.0})
        assert e == pytest.approx(-3.477 - 5.0)

    def test_legacy_table_is_consistent(self):
        # The backward-compatible molecule table must agree with the scheme.
        for species, energy in ADSORBATE_REFERENCE_ENERGIES.items():
            expected, _ = get_adsorbate_reference_energy(species)
            assert energy == pytest.approx(expected), species


class TestModuleImportsWithoutFairchem:
    def test_flags_present(self):
        from agent_materials_science.tools import fairchem_calc

        # Module must import cleanly whether or not fairchem is installed.
        assert isinstance(fairchem_calc.FAIRCHEM_AVAILABLE, bool)

    def test_calculator_raises_helpful_error_when_missing(self):
        from agent_materials_science.tools.fairchem_calc import (
            FAIRCHEM_AVAILABLE,
            FairchemCalculator,
        )

        if FAIRCHEM_AVAILABLE:
            pytest.skip("fairchem installed; error path not applicable")
        with pytest.raises(ImportError, match="fairchem-core"):
            FairchemCalculator()
