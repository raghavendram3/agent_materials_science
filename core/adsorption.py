"""
Adsorption site identification and analysis for surface slabs.

This module provides algorithms to identify different types of adsorption sites
on surface slabs:
- Top sites: Directly above surface atoms
- Bridge sites: Between two neighboring surface atoms
- Hollow sites: At the center of 3+ coordinating atoms (fcc, hcp)
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from ase.atoms import Atoms

try:
    from scipy.spatial import Voronoi, Delaunay
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    # pymatgen ships a mature, symmetry-aware adsorption-site finder. When it
    # is available we prefer it over the lightweight built-in geometric finder
    # because it deduplicates symmetry-equivalent sites (far fewer energy
    # evaluations) and handles periodicity and reoriented slabs robustly.
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder as _PmgASF
    from pymatgen.io.ase import AseAtomsAdaptor as _AseAdaptor
    PYMATGEN_ASF_AVAILABLE = True
except ImportError:
    PYMATGEN_ASF_AVAILABLE = False


@dataclass
class AdsorptionSite:
    """
    Represents an adsorption site on a surface.
    
    Attributes:
        position: 3D coordinates (x, y, z) in Angstrom
        site_type: Type of site ('top', 'bridge', 'hollow', 'fcc', 'hcp')
        coordinating_atoms: Indices of surface atoms coordinating this site
        surface_normal: Normal vector of the surface (default: [0, 0, 1])
        energy: Adsorption energy in eV (None if not calculated)
        metadata: Additional information about the site
    """
    position: np.ndarray
    site_type: str
    coordinating_atoms: List[int]
    surface_normal: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    energy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize surface normal vector."""
        self.position = np.array(self.position)
        self.surface_normal = np.array(self.surface_normal)
        norm = np.linalg.norm(self.surface_normal)
        if norm > 0:
            self.surface_normal = self.surface_normal / norm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position": self.position.tolist(),
            "site_type": self.site_type,
            "coordinating_atoms": self.coordinating_atoms,
            "surface_normal": self.surface_normal.tolist(),
            "energy": self.energy,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdsorptionSite":
        """Create from dictionary."""
        return cls(
            position=np.array(data["position"]),
            site_type=data["site_type"],
            coordinating_atoms=data["coordinating_atoms"],
            surface_normal=np.array(data.get("surface_normal", [0, 0, 1])),
            energy=data.get("energy"),
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self):
        energy_str = f", energy={self.energy:.3f} eV" if self.energy is not None else ""
        return (f"AdsorptionSite(type='{self.site_type}', "
                f"position=[{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}], "
                f"coordination={len(self.coordinating_atoms)}{energy_str})")


class AdsorptionSiteFinder:
    """
    Find and classify adsorption sites on a surface slab.
    
    This class analyzes a slab structure to identify potential adsorption
    sites based on surface geometry. It can find:
    - Top sites (on-top of surface atoms)
    - Bridge sites (between two neighboring atoms)
    - Hollow sites (fcc, hcp - center of 3+ atoms)
    """
    
    def __init__(
        self,
        slab: Atoms,
        height_offset: float = 2.0,
        neighbor_cutoff: float = 3.5,
        top_layer_threshold: float = 1.5,
    ):
        """
        Initialize the adsorption site finder.
        
        Args:
            slab: ASE Atoms object (surface slab)
            height_offset: Height above surface for placing adsorbates (Å)
            neighbor_cutoff: Cutoff distance for identifying neighbors (Å)
            top_layer_threshold: Z-distance threshold for top layer atoms (Å)
        """
        self.slab = slab
        self.height_offset = height_offset
        self.neighbor_cutoff = neighbor_cutoff
        self.top_layer_threshold = top_layer_threshold

        # In-plane (surface) cell vectors used for the minimum-image
        # convention. We assume the surface lies in the xy-plane and the
        # vacuum is along the third cell vector (the convention produced by
        # ASE/pymatgen slab generators).
        cell = np.array(slab.get_cell())
        self._cell = cell
        self._a = cell[0]
        self._b = cell[1]
        pbc = slab.get_pbc()
        # Whether the two in-plane directions are periodic. A clean slab is
        # periodic in-plane and non-periodic along the surface normal.
        self._periodic_inplane = bool(pbc[0]) and bool(pbc[1])

        # Identify surface atoms
        self._top_layer_indices = self._find_top_layer()
        self._top_layer_positions = slab.get_positions()[self._top_layer_indices]
        
    @property
    def top_layer_indices(self) -> List[int]:
        """Indices of atoms in the top layer."""
        return self._top_layer_indices
    
    @property
    def top_layer_positions(self) -> np.ndarray:
        """Positions of top layer atoms."""
        return self._top_layer_positions
    
    def _find_top_layer(self) -> List[int]:
        """
        Identify atoms in the top (surface) layer.
        
        Returns:
            List of atom indices in the top layer
        """
        positions = self.slab.get_positions()
        z_coords = positions[:, 2]
        max_z = z_coords.max()
        
        # Find atoms within threshold of maximum z
        top_mask = (max_z - z_coords) < self.top_layer_threshold
        top_indices = np.where(top_mask)[0].tolist()
        
        # Additional check: filter out atoms that have neighbors above them
        filtered = []
        for idx in top_indices:
            z = positions[idx, 2]
            is_surface = True
            
            for other_idx in range(len(self.slab)):
                if other_idx == idx:
                    continue
                other_z = positions[other_idx, 2]
                
                # Check if there's an atom significantly above
                if other_z > z + 0.3:
                    lateral_dist = np.linalg.norm(
                        positions[idx, :2] - positions[other_idx, :2]
                    )
                    if lateral_dist < self.neighbor_cutoff * 0.8:
                        is_surface = False
                        break
            
            if is_surface:
                filtered.append(idx)
        
        return filtered if filtered else top_indices

    def _inplane_images(self):
        """
        Generate in-plane periodic images of the top-layer atoms.

        For a periodic slab, neighbouring atoms (and therefore bridge and
        hollow sites) frequently sit across a cell boundary. To find them we
        tile the top-layer atoms over the 3x3 in-plane neighbourhood and keep
        a mapping back to the original ("parent") atom index.

        Returns:
            Tuple of (image_positions [M, 3], parent_indices [M],
            is_central [M] bool) where ``is_central`` marks the images that
            live in the original (0, 0) cell.
        """
        positions = self._top_layer_positions
        parents = np.asarray(self._top_layer_indices)

        if not self._periodic_inplane:
            # Non-periodic: the only "images" are the atoms themselves.
            central = np.ones(len(positions), dtype=bool)
            return positions.copy(), parents.copy(), central

        shifts = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 0), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

        img_pos = []
        img_parent = []
        img_central = []
        for (i, j) in shifts:
            offset = i * self._a + j * self._b
            img_pos.append(positions + offset)
            img_parent.append(parents)
            img_central.append(np.full(len(positions), (i == 0 and j == 0)))

        return (
            np.vstack(img_pos),
            np.concatenate(img_parent),
            np.concatenate(img_central),
        )

    def _wrap_xy(self, position: np.ndarray) -> np.ndarray:
        """
        Wrap a Cartesian point back into the in-plane unit cell.

        Only the x and y components are wrapped (using the two in-plane cell
        vectors); the z component is left untouched so adsorbate height is
        preserved.
        """
        position = np.array(position, dtype=float)
        if not self._periodic_inplane:
            return position

        # 2x2 matrix whose columns are the in-plane projections of a and b.
        m = np.array([[self._a[0], self._b[0]],
                      [self._a[1], self._b[1]]])
        det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        if abs(det) < 1e-9:
            return position  # Degenerate in-plane cell; nothing sensible to do.

        frac = np.linalg.solve(m, position[:2])
        frac = frac % 1.0
        position[:2] = m @ frac
        return position

    def find_all_sites(self) -> List[AdsorptionSite]:
        """
        Find all types of adsorption sites on the surface.
        
        Returns:
            List of AdsorptionSite objects
        """
        sites = []
        
        # Find each type of site
        sites.extend(self.find_top_sites())
        sites.extend(self.find_bridge_sites())
        sites.extend(self.find_hollow_sites())
        
        return sites
    
    def find_top_sites(self) -> List[AdsorptionSite]:
        """
        Find on-top adsorption sites (directly above surface atoms).
        
        Returns:
            List of top sites
        """
        sites = []
        positions = self.slab.get_positions()
        
        for idx in self._top_layer_indices:
            pos = positions[idx].copy()
            pos[2] += self.height_offset
            
            site = AdsorptionSite(
                position=pos,
                site_type="top",
                coordinating_atoms=[idx],
                metadata={"element": self.slab.get_chemical_symbols()[idx]}
            )
            sites.append(site)
        
        return sites
    
    def find_bridge_sites(self) -> List[AdsorptionSite]:
        """
        Find bridge sites (midpoint between two neighboring surface atoms).

        Neighbours are detected using the minimum-image convention so that
        pairs spanning a periodic cell boundary are included.

        Returns:
            List of bridge sites
        """
        sites = []
        symbols = self.slab.get_chemical_symbols()

        # Tile the surface atoms so we can catch neighbours across boundaries.
        img_pos, img_parent, img_central = self._inplane_images()
        central_pos = self._top_layer_positions
        central_parent = np.asarray(self._top_layer_indices)

        seen = []  # wrapped midpoints already emitted (for de-duplication)
        for ci, c_idx in enumerate(central_parent):
            pos_i = central_pos[ci]
            for k in range(len(img_pos)):
                j_idx = int(img_parent[k])
                pos_j = img_pos[k]

                dist = np.linalg.norm(pos_i - pos_j)
                if dist < 0.1 or dist >= self.neighbor_cutoff:
                    continue

                mid = (pos_i + pos_j) / 2.0
                mid[2] = max(pos_i[2], pos_j[2]) + self.height_offset
                mid = self._wrap_xy(mid)

                # Skip if we already have an equivalent (wrapped) midpoint.
                if any(np.linalg.norm(mid[:2] - s[:2]) < 0.3 for s in seen):
                    continue
                seen.append(mid)

                site = AdsorptionSite(
                    position=mid,
                    site_type="bridge",
                    coordinating_atoms=sorted({int(c_idx), j_idx}),
                    metadata={
                        "elements": [symbols[int(c_idx)], symbols[j_idx]],
                        "bond_length": float(dist),
                    },
                )
                sites.append(site)

        return sites
    
    def find_hollow_sites(self) -> List[AdsorptionSite]:
        """
        Find hollow sites using Delaunay triangulation.

        Hollow sites are at the center of triangles formed by surface atoms.
        These include fcc and hcp sites for close-packed surfaces. The
        triangulation is performed over periodic images of the surface atoms
        so that hollow sites at the cell boundary are recovered.

        Returns:
            List of hollow sites
        """
        if len(self._top_layer_indices) < 1:
            return []

        if not SCIPY_AVAILABLE:
            # Fallback: simple hollow site detection
            return self._find_hollow_sites_simple()

        symbols = self.slab.get_chemical_symbols()
        img_pos, img_parent, img_central = self._inplane_images()

        # Need at least three points to triangulate.
        if len(img_pos) < 3:
            return []

        try:
            tri = Delaunay(img_pos[:, :2])
        except Exception as e:
            print(f"Warning: Delaunay triangulation failed: {e}")
            return self._find_hollow_sites_simple()

        sites = []
        seen = []
        for simplex in tri.simplices:
            # Keep only triangles that touch the central cell, otherwise the
            # same physical site is generated many times by the periodic tiling.
            if not np.any(img_central[simplex]):
                continue

            triangle_pos = img_pos[simplex]
            edges = [
                float(np.linalg.norm(triangle_pos[0] - triangle_pos[1])),
                float(np.linalg.norm(triangle_pos[1] - triangle_pos[2])),
                float(np.linalg.norm(triangle_pos[2] - triangle_pos[0])),
            ]
            if max(edges) > self.neighbor_cutoff * 1.5:
                continue  # Skip large (non-physical) triangles

            centroid = triangle_pos.mean(axis=0)
            centroid[2] = triangle_pos[:, 2].max() + self.height_offset
            centroid = self._wrap_xy(centroid)

            if any(np.linalg.norm(centroid[:2] - s[:2]) < 0.3 for s in seen):
                continue
            seen.append(centroid)

            atom_indices = [int(img_parent[i]) for i in simplex]
            site = AdsorptionSite(
                position=centroid,
                site_type="hollow",
                coordinating_atoms=atom_indices,
                metadata={
                    "elements": [symbols[i] for i in atom_indices],
                    "edge_lengths": edges,
                },
            )
            sites.append(site)

        return sites

    def _find_hollow_sites_simple(self) -> List[AdsorptionSite]:
        """
        Simple hollow site finder without scipy.

        Considers triplets over periodic images of the surface atoms so that
        boundary-spanning hollow sites are not missed.

        Returns:
            List of hollow sites
        """
        symbols = self.slab.get_chemical_symbols()
        img_pos, img_parent, img_central = self._inplane_images()

        sites = []
        seen = []
        m = len(img_pos)
        for i in range(m):
            # At least one vertex must be in the central cell.
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    if not (img_central[i] or img_central[j] or img_central[k]):
                        continue

                    pos = img_pos[[i, j, k]]
                    d_ij = np.linalg.norm(pos[0] - pos[1])
                    d_jk = np.linalg.norm(pos[1] - pos[2])
                    d_ki = np.linalg.norm(pos[2] - pos[0])

                    if max(d_ij, d_jk, d_ki) > self.neighbor_cutoff * 1.5:
                        continue
                    if min(d_ij, d_jk, d_ki) < 0.1:
                        continue  # Degenerate / coincident image points

                    centroid = pos.mean(axis=0)
                    centroid[2] = pos[:, 2].max() + self.height_offset
                    centroid = self._wrap_xy(centroid)

                    if any(np.linalg.norm(centroid[:2] - s[:2]) < 0.3 for s in seen):
                        continue
                    seen.append(centroid)

                    atom_indices = [int(img_parent[i]), int(img_parent[j]),
                                    int(img_parent[k])]
                    sites.append(AdsorptionSite(
                        position=centroid,
                        site_type="hollow",
                        coordinating_atoms=atom_indices,
                        metadata={"elements": [symbols[a] for a in atom_indices]},
                    ))

        return sites
    
    def filter_by_type(
        self,
        sites: List[AdsorptionSite],
        site_types: List[str]
    ) -> List[AdsorptionSite]:
        """
        Filter sites by type.

        "fcc" and "hcp" are accepted as aliases for "hollow" (the built-in
        finder does not distinguish the two), so requesting them does not
        silently return an empty list.

        Args:
            sites: List of sites to filter
            site_types: Types to keep (e.g., ['top', 'bridge'])

        Returns:
            Filtered list of sites
        """
        wanted = set(site_types)
        if "fcc" in wanted or "hcp" in wanted:
            wanted.add("hollow")
        return [s for s in sites if s.site_type in wanted]
    
    def remove_duplicates(
        self,
        sites: List[AdsorptionSite],
        tolerance: float = 0.5
    ) -> List[AdsorptionSite]:
        """
        Remove duplicate sites that are too close together.

        Distances are evaluated under the in-plane minimum-image convention so
        that two sites separated only by a lattice translation are treated as
        duplicates.

        Args:
            sites: List of sites
            tolerance: Distance tolerance (Å)

        Returns:
            List of unique sites
        """
        if not sites:
            return []

        unique = [sites[0]]

        for site in sites[1:]:
            is_unique = True
            for existing in unique:
                dist = self._mic_distance(site.position, existing.position)
                if dist < tolerance:
                    is_unique = False
                    break

            if is_unique:
                unique.append(site)

        return unique

    def _mic_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Distance between two points using the in-plane minimum image."""
        delta = np.array(p1, dtype=float) - np.array(p2, dtype=float)
        if self._periodic_inplane:
            m = np.array([[self._a[0], self._b[0]],
                          [self._a[1], self._b[1]]])
            det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
            if abs(det) > 1e-9:
                frac = np.linalg.solve(m, delta[:2])
                frac -= np.round(frac)
                delta[:2] = m @ frac
        return float(np.linalg.norm(delta))
    
    def rank_by_coordination(
        self,
        sites: List[AdsorptionSite]
    ) -> List[AdsorptionSite]:
        """
        Rank sites by coordination number (higher = more favorable).
        
        Args:
            sites: Sites to rank
            
        Returns:
            Sorted list (highest coordination first)
        """
        return sorted(sites, key=lambda s: len(s.coordinating_atoms), reverse=True)
    
    def rank_by_energy(
        self,
        sites: List[AdsorptionSite]
    ) -> List[AdsorptionSite]:
        """
        Rank sites by adsorption energy (lowest = most favorable).
        
        Args:
            sites: Sites with energy values
            
        Returns:
            Sorted list (lowest energy first)
        """
        # Sites without energy go to the end
        sites_with_e = [s for s in sites if s.energy is not None]
        sites_without_e = [s for s in sites if s.energy is None]
        
        sorted_sites = sorted(sites_with_e, key=lambda s: s.energy)
        return sorted_sites + sites_without_e
    
    def get_site_summary(self, sites: List[AdsorptionSite]) -> Dict[str, Any]:
        """
        Get a summary of found sites.
        
        Args:
            sites: List of sites
            
        Returns:
            Summary dictionary
        """
        type_counts = {}
        for site in sites:
            type_counts[site.site_type] = type_counts.get(site.site_type, 0) + 1
        
        energies = [s.energy for s in sites if s.energy is not None]
        
        summary = {
            "total_sites": len(sites),
            "sites_by_type": type_counts,
            "surface_atoms": len(self._top_layer_indices),
        }
        
        if energies:
            summary["energy_range"] = {
                "min": min(energies),
                "max": max(energies),
                "mean": np.mean(energies),
            }
        
        return summary

# Mapping between this package's site-type vocabulary and pymatgen's.
_PMG_TO_LOCAL = {"ontop": "top", "bridge": "bridge", "hollow": "hollow"}
_LOCAL_TO_PMG = {"top": "ontop", "bridge": "bridge", "hollow": "hollow",
                 "fcc": "hollow", "hcp": "hollow"}


def find_sites_pymatgen(
    slab: Atoms,
    height_offset: float = 2.0,
    site_types: Optional[List[str]] = None,
    symm_reduce: float = 0.01,
    near_reduce: float = 0.01,
    neighbor_cutoff: float = 3.5,
) -> List[AdsorptionSite]:
    """
    Find adsorption sites using pymatgen's ``AdsorbateSiteFinder``.

    This is the recommended backend when pymatgen is installed: it is
    symmetry-aware (so symmetry-equivalent sites are collapsed, dramatically
    reducing the number of energy evaluations), periodicity-correct, and able
    to handle reoriented slabs. The returned objects use this package's
    ``AdsorptionSite`` type so the rest of the workflow is unchanged.

    Args:
        slab: ASE Atoms slab (surface normal assumed along z).
        height_offset: Distance to place sites above the surface (Å).
        site_types: Subset of {"top", "bridge", "hollow"} to return
            (None = all). "fcc"/"hcp" are treated as "hollow".
        symm_reduce: Symmetry tolerance for collapsing equivalent sites; set
            to 0 to keep every site.
        near_reduce: Tolerance for removing near-duplicate sites.
        neighbor_cutoff: Cutoff (Å) used to attribute coordinating surface
            atoms to each site (for ranking/metadata only).

    Returns:
        List of AdsorptionSite objects.
    """
    if not PYMATGEN_ASF_AVAILABLE:
        raise RuntimeError(
            "pymatgen's AdsorbateSiteFinder is not available. Install pymatgen "
            "or use the built-in AdsorptionSiteFinder backend."
        )

    # Translate requested types into pymatgen's vocabulary.
    if site_types:
        pmg_positions = sorted({_LOCAL_TO_PMG.get(t, t) for t in site_types})
    else:
        pmg_positions = ["ontop", "bridge", "hollow"]

    structure = _AseAdaptor().get_structure(slab)
    asf = _PmgASF(structure)
    found = asf.find_adsorption_sites(
        distance=height_offset,
        symm_reduce=symm_reduce,
        near_reduce=near_reduce,
        positions=pmg_positions,
    )

    # Pre-compute top-layer atom positions for coordinating-atom attribution.
    positions = slab.get_positions()
    z = positions[:, 2]
    top_mask = (z.max() - z) < 1.5
    top_indices = np.where(top_mask)[0]
    symbols = slab.get_chemical_symbols()

    expected_coord = {"top": 1, "bridge": 2, "hollow": 3}

    sites: List[AdsorptionSite] = []
    for pmg_type, local_type in _PMG_TO_LOCAL.items():
        for pos in found.get(pmg_type, []):
            pos = np.asarray(pos, dtype=float)
            # Attribute the nearest surface atoms (in-plane) as coordinating.
            if len(top_indices):
                lateral = np.linalg.norm(positions[top_indices, :2] - pos[:2], axis=1)
                order = np.argsort(lateral)
                n_coord = expected_coord.get(local_type, 1)
                coord = [int(top_indices[i]) for i in order[:n_coord]]
            else:
                coord = []
            sites.append(AdsorptionSite(
                position=pos,
                site_type=local_type,
                coordinating_atoms=coord,
                metadata={
                    "backend": "pymatgen",
                    "elements": [symbols[i] for i in coord],
                },
            ))
    return sites


def summarize_sites(sites: List[AdsorptionSite], n_surface_atoms: int = 0) -> Dict[str, Any]:
    """Build a site-summary dict (mirrors AdsorptionSiteFinder.get_site_summary)."""
    type_counts: Dict[str, int] = {}
    for s in sites:
        type_counts[s.site_type] = type_counts.get(s.site_type, 0) + 1
    summary: Dict[str, Any] = {
        "total_sites": len(sites),
        "sites_by_type": type_counts,
        "surface_atoms": n_surface_atoms,
    }
    energies = [s.energy for s in sites if s.energy is not None]
    if energies:
        summary["energy_range"] = {
            "min": float(min(energies)),
            "max": float(max(energies)),
            "mean": float(np.mean(energies)),
        }
    return summary
