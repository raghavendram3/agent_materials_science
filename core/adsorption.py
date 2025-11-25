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
        
        Returns:
            List of bridge sites
        """
        sites = []
        pairs = self._find_neighbor_pairs()
        positions = self.slab.get_positions()
        
        for i, j in pairs:
            pos_i = positions[i]
            pos_j = positions[j]
            
            # Midpoint
            mid = (pos_i + pos_j) / 2
            mid[2] = max(pos_i[2], pos_j[2]) + self.height_offset
            
            site = AdsorptionSite(
                position=mid,
                site_type="bridge",
                coordinating_atoms=[i, j],
                metadata={
                    "elements": [
                        self.slab.get_chemical_symbols()[i],
                        self.slab.get_chemical_symbols()[j]
                    ],
                    "bond_length": np.linalg.norm(pos_i - pos_j)
                }
            )
            sites.append(site)
        
        return sites
    
    def find_hollow_sites(self) -> List[AdsorptionSite]:
        """
        Find hollow sites using Delaunay triangulation.
        
        Hollow sites are at the center of triangles formed by surface atoms.
        These include fcc and hcp sites for close-packed surfaces.
        
        Returns:
            List of hollow sites
        """
        sites = []
        
        if len(self._top_layer_indices) < 3:
            return sites
        
        if not SCIPY_AVAILABLE:
            # Fallback: simple hollow site detection
            return self._find_hollow_sites_simple()
        
        # Use Delaunay triangulation
        positions_2d = self._top_layer_positions[:, :2]
        
        try:
            tri = Delaunay(positions_2d)
            
            for simplex in tri.simplices:
                # Get the three atom indices (in full slab indexing)
                atom_indices = [self._top_layer_indices[i] for i in simplex]
                
                # Get positions
                triangle_pos = self._top_layer_positions[simplex]
                
                # Check if triangle is reasonable size
                edges = [
                    np.linalg.norm(triangle_pos[0] - triangle_pos[1]),
                    np.linalg.norm(triangle_pos[1] - triangle_pos[2]),
                    np.linalg.norm(triangle_pos[2] - triangle_pos[0]),
                ]
                
                if max(edges) > self.neighbor_cutoff * 1.5:
                    continue  # Skip large triangles
                
                # Calculate centroid
                centroid = triangle_pos.mean(axis=0)
                centroid[2] = triangle_pos[:, 2].max() + self.height_offset
                
                # Determine if fcc or hcp (simplified)
                site_type = "hollow"
                
                site = AdsorptionSite(
                    position=centroid,
                    site_type=site_type,
                    coordinating_atoms=atom_indices,
                    metadata={
                        "elements": [self.slab.get_chemical_symbols()[i] for i in atom_indices],
                        "edge_lengths": edges,
                    }
                )
                sites.append(site)
                
        except Exception as e:
            print(f"Warning: Delaunay triangulation failed: {e}")
            return self._find_hollow_sites_simple()
        
        return sites
    
    def _find_hollow_sites_simple(self) -> List[AdsorptionSite]:
        """
        Simple hollow site finder without scipy.
        
        Returns:
            List of hollow sites
        """
        sites = []
        positions = self.slab.get_positions()
        n = len(self._top_layer_indices)
        
        # Check all triplets
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    idx_i = self._top_layer_indices[i]
                    idx_j = self._top_layer_indices[j]
                    idx_k = self._top_layer_indices[k]
                    
                    pos = self._top_layer_positions[[i, j, k]]
                    
                    # Check distances
                    d_ij = np.linalg.norm(pos[0] - pos[1])
                    d_jk = np.linalg.norm(pos[1] - pos[2])
                    d_ki = np.linalg.norm(pos[2] - pos[0])
                    
                    # Only accept reasonable triangles
                    if max(d_ij, d_jk, d_ki) > self.neighbor_cutoff * 1.5:
                        continue
                    
                    # Centroid
                    centroid = pos.mean(axis=0)
                    centroid[2] = pos[:, 2].max() + self.height_offset
                    
                    site = AdsorptionSite(
                        position=centroid,
                        site_type="hollow",
                        coordinating_atoms=[idx_i, idx_j, idx_k],
                    )
                    sites.append(site)
        
        return sites
    
    def _find_neighbor_pairs(self) -> List[Tuple[int, int]]:
        """
        Find pairs of neighboring atoms in the top layer.
        
        Returns:
            List of (i, j) index pairs
        """
        pairs = []
        n = len(self._top_layer_indices)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(
                    self._top_layer_positions[i] - self._top_layer_positions[j]
                )
                
                if dist < self.neighbor_cutoff:
                    pairs.append((
                        self._top_layer_indices[i],
                        self._top_layer_indices[j]
                    ))
        
        return pairs
    
    def filter_by_type(
        self,
        sites: List[AdsorptionSite],
        site_types: List[str]
    ) -> List[AdsorptionSite]:
        """
        Filter sites by type.
        
        Args:
            sites: List of sites to filter
            site_types: Types to keep (e.g., ['top', 'bridge'])
            
        Returns:
            Filtered list of sites
        """
        return [s for s in sites if s.site_type in site_types]
    
    def remove_duplicates(
        self,
        sites: List[AdsorptionSite],
        tolerance: float = 0.5
    ) -> List[AdsorptionSite]:
        """
        Remove duplicate sites that are too close together.
        
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
                dist = np.linalg.norm(site.position - existing.position)
                if dist < tolerance:
                    is_unique = False
                    break
            
            if is_unique:
                unique.append(site)
        
        return unique
    
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
