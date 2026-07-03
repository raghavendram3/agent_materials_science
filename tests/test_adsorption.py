"""Tests for adsorption-site finding."""

import pytest
from pymatgen.core import Lattice, Structure

from agent_materials_science.core.adsorption import AdsorptionSiteFinder
from agent_materials_science.core.surface import SurfaceBuilder
from agent_materials_science.tools.converters import (
    create_adsorbate,
    place_adsorbate_at_site,
)


@pytest.fixture(scope="module")
def si111_slab():
    si = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
    )
    builder = SurfaceBuilder(si, (1, 1, 1))
    return builder.build_slab(termination=0, supercell=(2, 2))


class TestAutoCutoff:
    def test_si111_finds_bridge_sites(self, si111_slab):
        """
        Si(111) surface NN distance is 3.84 A. The old fixed 3.5 A cutoff
        silently found ZERO bridge sites here; the auto cutoff must fix that.
        """
        finder = AdsorptionSiteFinder(si111_slab)  # neighbor_cutoff=None -> auto
        sites = finder.remove_duplicates(finder.find_all_sites())
        types = {s.site_type for s in sites}
        assert "bridge" in types, f"no bridge sites found (types: {types})"
        assert finder.neighbor_cutoff > 3.84  # must exceed the NN distance

    def test_explicit_cutoff_respected(self, si111_slab):
        finder = AdsorptionSiteFinder(si111_slab, neighbor_cutoff=3.0)
        assert finder.neighbor_cutoff == 3.0

    def test_auto_cutoff_bounds(self, si111_slab):
        finder = AdsorptionSiteFinder(si111_slab)
        assert 2.0 <= finder.neighbor_cutoff <= 6.0


class TestSites:
    def test_sites_above_surface(self, si111_slab):
        finder = AdsorptionSiteFinder(si111_slab, height_offset=2.0)
        sites = finder.remove_duplicates(finder.find_all_sites())
        assert sites
        z_top = si111_slab.get_positions()[:, 2].max()
        for site in sites:
            assert site.position[2] > z_top

    def test_top_sites_match_top_layer(self, si111_slab):
        finder = AdsorptionSiteFinder(si111_slab)
        tops = [s for s in finder.find_all_sites() if s.site_type == "top"]
        assert len(tops) == len(finder.top_layer_indices)


class TestPlacement:
    def test_anchor_lands_on_site(self, si111_slab):
        finder = AdsorptionSiteFinder(si111_slab, height_offset=2.0)
        site = finder.find_all_sites()[0]
        co = create_adsorbate("CO", height=0)
        combined = place_adsorbate_at_site(si111_slab, co, site.position)
        assert len(combined) == len(si111_slab) + len(co)
        # The anchor (first adsorbate atom, C for "CO") sits at the site.
        anchor = combined.get_positions()[len(si111_slab)]
        assert abs(anchor[2] - site.position[2]) < 1e-6

    def test_constraints_survive_placement(self, si111_slab):
        slab = si111_slab.copy()
        from agent_materials_science.tools.ase_tools import fix_bottom_layers

        slab = fix_bottom_layers(slab, 2)
        h = create_adsorbate("H", height=0)
        finder = AdsorptionSiteFinder(slab)
        site = finder.find_all_sites()[0]
        combined = place_adsorbate_at_site(slab, h, site.position)
        assert combined.constraints, "FixAtoms lost during placement"
