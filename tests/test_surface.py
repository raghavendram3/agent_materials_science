"""Tests for slab generation (no network or ML dependencies needed)."""

import pytest
from pymatgen.core import Lattice, Structure

from agent_materials_science.core.surface import (
    SurfaceBuilder,
    miller_to_string,
    string_to_miller,
)


@pytest.fixture(scope="module")
def si_bulk():
    return Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
    )


class TestSurfaceBuilder:
    def test_terminations(self, si_bulk):
        builder = SurfaceBuilder(si_bulk, (1, 1, 1))
        terms = builder.get_available_terminations(10.0, 15.0)
        assert len(terms) >= 1
        assert terms[0].n_atoms > 0

    def test_cache_is_keyed_by_parameters(self, si_bulk):
        """A thicker slab request must not return stale thin-slab results."""
        builder = SurfaceBuilder(si_bulk, (1, 1, 1))
        thin = builder.build_slab(min_slab_size=10.0)
        thick = builder.build_slab(min_slab_size=30.0)
        assert len(thick) > len(thin)

    def test_in_unit_planes(self, si_bulk):
        builder = SurfaceBuilder(si_bulk, (1, 1, 1))
        few = builder.build_slab(min_slab_size=4, in_unit_planes=True)
        many = builder.build_slab(min_slab_size=8, in_unit_planes=True)
        assert len(many) > len(few)

    def test_supercell(self, si_bulk):
        builder = SurfaceBuilder(si_bulk, (1, 1, 1))
        unit = builder.build_slab(supercell=(1, 1))
        double = builder.build_slab(supercell=(2, 2))
        assert len(double) == 4 * len(unit)

    def test_invalid_termination_warns_and_falls_back(self, si_bulk):
        builder = SurfaceBuilder(si_bulk, (1, 1, 1))
        with pytest.warns(UserWarning, match="Termination"):
            slab = builder.build_slab(termination=99)
        assert len(slab) > 0

    def test_fix_layers_adds_constraints(self, si_bulk):
        builder = SurfaceBuilder(si_bulk, (1, 1, 1))
        slab = builder.build_slab(fix_layers=2)
        assert slab.constraints, "FixAtoms constraint expected"


class TestMillerParsing:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("111", (1, 1, 1)),
            ("110", (1, 1, 0)),
            ("1,0,0", (1, 0, 0)),
            ("1,-1,0", (1, -1, 0)),
            ("1m10", (1, -1, 0)),
            ("1-10", (1, -1, 0)),
        ],
    )
    def test_parse(self, s, expected):
        assert string_to_miller(s) == expected

    def test_round_trip(self):
        for miller in [(1, 1, 1), (1, -1, 0), (2, 1, 0)]:
            assert string_to_miller(miller_to_string(miller)) == miller

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            string_to_miller("11")
