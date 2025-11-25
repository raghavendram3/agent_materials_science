"""
Converters and utilities for adsorbate creation and structure manipulation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ase.atoms import Atoms
from ase import Atom


# Predefined adsorbate geometries (positions relative to binding site)
ADSORBATE_GEOMETRIES = {
    # Single atoms
    "H": [("H", (0, 0, 0))],
    "O": [("O", (0, 0, 0))],
    "N": [("N", (0, 0, 0))],
    "C": [("C", (0, 0, 0))],
    "S": [("S", (0, 0, 0))],
    "F": [("F", (0, 0, 0))],
    "Cl": [("Cl", (0, 0, 0))],
    
    # Diatomic homonuclear molecules
    "H2": [("H", (0, 0, 0)), ("H", (0, 0, 0.74))],  # H-H bond ~0.74 Å
    "O2": [("O", (0, 0, 0)), ("O", (0, 0, 1.21))],  # O=O bond ~1.21 Å
    "N2": [("N", (0, 0, 0)), ("N", (0, 0, 1.10))],  # N≡N bond ~1.10 Å
    
    # Diatomic heteronuclear molecules (binding atom at origin)
    "CO": [("C", (0, 0, 0)), ("O", (0, 0, 1.128))],  # C-down
    "OC": [("O", (0, 0, 0)), ("C", (0, 0, 1.128))],  # O-down
    "OH": [("O", (0, 0, 0)), ("H", (0, 0, 0.97))],
    "HO": [("H", (0, 0, 0)), ("O", (0, 0, 0.97))],  # H-down (for dissociation studies)
    "NO": [("N", (0, 0, 0)), ("O", (0, 0, 1.15))],
    "ON": [("O", (0, 0, 0)), ("N", (0, 0, 1.15))],
    "HF": [("H", (0, 0, 0)), ("F", (0, 0, 0.92))],
    "HCl": [("H", (0, 0, 0)), ("Cl", (0, 0, 1.27))],
    
    # Water and hydroxyl
    "H2O": [
        ("O", (0, 0, 0)),
        ("H", (0.757, 0.587, 0)),
        ("H", (-0.757, 0.587, 0))
    ],
    
    # Linear molecules
    "CO2": [
        ("C", (0, 0, 0)),
        ("O", (0, 0, 1.16)),
        ("O", (0, 0, -1.16))
    ],
    "N2O": [
        ("N", (0, 0, 0)),
        ("N", (0, 0, 1.13)),
        ("O", (0, 0, 2.31))
    ],
    
    # Ammonia (N-down)
    "NH3": [
        ("N", (0, 0, 0)),
        ("H", (0, 0.942, 0.38)),
        ("H", (0.816, -0.471, 0.38)),
        ("H", (-0.816, -0.471, 0.38))
    ],
    
    # Methane
    "CH4": [
        ("C", (0, 0, 0)),
        ("H", (0.629, 0.629, 0.629)),
        ("H", (-0.629, -0.629, 0.629)),
        ("H", (-0.629, 0.629, -0.629)),
        ("H", (0.629, -0.629, -0.629))
    ],
    
    # Ethylene
    "C2H4": [
        ("C", (-0.665, 0, 0)),
        ("C", (0.665, 0, 0)),
        ("H", (-1.237, 0.929, 0)),
        ("H", (-1.237, -0.929, 0)),
        ("H", (1.237, 0.929, 0)),
        ("H", (1.237, -0.929, 0))
    ],
    
    # Acetylene
    "C2H2": [
        ("C", (-0.6, 0, 0)),
        ("C", (0.6, 0, 0)),
        ("H", (-1.67, 0, 0)),
        ("H", (1.67, 0, 0))
    ],
    
    # Formate (HCOO)
    "HCOO": [
        ("C", (0, 0, 0)),
        ("O", (-1.12, 0.37, 0)),
        ("O", (1.12, 0.37, 0)),
        ("H", (0, -1.1, 0))
    ],
    
    # Methanol
    "CH3OH": [
        ("C", (0, 0, 0)),
        ("O", (1.43, 0, 0)),
        ("H", (-0.39, 1.03, 0)),
        ("H", (-0.39, -0.51, 0.89)),
        ("H", (-0.39, -0.51, -0.89)),
        ("H", (1.83, 0.89, 0))
    ],
}


def create_adsorbate(
    species: str,
    height: float = 2.0,
    cell_size: float = 20.0
) -> Atoms:
    """
    Create an adsorbate molecule/atom.
    
    Args:
        species: Chemical species name (e.g., 'H', 'CO', 'OH', 'H2O')
        height: Height offset in z-direction (Angstrom)
        cell_size: Size of cubic cell for isolated molecule (Angstrom)
        
    Returns:
        ASE Atoms object with the adsorbate
    """
    species_upper = species.upper()
    
    # Check if we have a predefined geometry
    if species in ADSORBATE_GEOMETRIES:
        geometry = ADSORBATE_GEOMETRIES[species]
    elif species_upper in ADSORBATE_GEOMETRIES:
        geometry = ADSORBATE_GEOMETRIES[species_upper]
    else:
        # Treat as single atom
        geometry = [(species, (0, 0, 0))]
    
    # Create atoms
    atoms_list = []
    for symbol, pos in geometry:
        # Shift positions so binding atom is at height offset
        new_pos = (pos[0], pos[1], pos[2] + height)
        atoms_list.append(Atom(symbol, new_pos))
    
    adsorbate = Atoms(atoms_list)
    
    # Set up a large cell for isolated molecule calculations
    adsorbate.set_cell([cell_size, cell_size, cell_size])
    adsorbate.center()
    adsorbate.set_pbc([True, True, True])
    
    return adsorbate


def place_adsorbate_at_site(
    slab: Atoms,
    adsorbate: Atoms,
    site_position: np.ndarray,
    height_offset: float = 0.0
) -> Atoms:
    """
    Place an adsorbate at a specific position on the slab.
    
    Args:
        slab: ASE Atoms slab (clean)
        adsorbate: ASE Atoms adsorbate molecule
        site_position: 3D position (x, y, z) for placement
        height_offset: Additional height offset (Angstrom)
        
    Returns:
        Combined ASE Atoms (slab + adsorbate)
    """
    combined = slab.copy()
    ads_copy = adsorbate.copy()
    
    # Get adsorbate center of mass (or first atom for single atoms)
    if len(ads_copy) == 1:
        ads_center = ads_copy.get_positions()[0]
    else:
        ads_center = ads_copy.get_center_of_mass()
    
    # Calculate translation vector
    target_pos = np.array(site_position)
    target_pos[2] += height_offset
    translation = target_pos - ads_center
    
    # Translate adsorbate
    ads_copy.translate(translation)
    
    # Combine slab and adsorbate
    combined.extend(ads_copy)
    
    return combined


def place_adsorbate_on_surface(
    slab: Atoms,
    species: str,
    position_xy: Tuple[float, float],
    height: float = 2.0
) -> Atoms:
    """
    Place an adsorbate at a specific x,y position on the slab surface.
    
    Args:
        slab: ASE Atoms slab
        species: Adsorbate species name
        position_xy: (x, y) position on surface
        height: Height above the highest surface atom
        
    Returns:
        Combined ASE Atoms (slab + adsorbate)
    """
    # Find the highest z-coordinate
    z_max = slab.get_positions()[:, 2].max()
    
    # Create adsorbate
    adsorbate = create_adsorbate(species, height=0)
    
    # Position on surface
    site_position = np.array([position_xy[0], position_xy[1], z_max + height])
    
    return place_adsorbate_at_site(slab, adsorbate, site_position)


def get_adsorbate_info(species: str) -> Dict[str, Any]:
    """
    Get information about an adsorbate species.
    
    Args:
        species: Adsorbate species name
        
    Returns:
        Dictionary with adsorbate information
    """
    species_check = species.upper() if species.upper() in ADSORBATE_GEOMETRIES else species
    
    if species_check in ADSORBATE_GEOMETRIES:
        geometry = ADSORBATE_GEOMETRIES[species_check]
        elements = [atom[0] for atom in geometry]
        
        return {
            "species": species,
            "n_atoms": len(geometry),
            "elements": list(set(elements)),
            "composition": {el: elements.count(el) for el in set(elements)},
            "predefined": True,
        }
    else:
        return {
            "species": species,
            "n_atoms": 1,
            "elements": [species],
            "composition": {species: 1},
            "predefined": False,
        }


def list_available_adsorbates() -> List[str]:
    """
    List all available predefined adsorbate species.
    
    Returns:
        List of adsorbate species names
    """
    return list(ADSORBATE_GEOMETRIES.keys())


def rotate_adsorbate(
    adsorbate: Atoms,
    angle: float,
    axis: str = 'z'
) -> Atoms:
    """
    Rotate an adsorbate around an axis.
    
    Args:
        adsorbate: ASE Atoms adsorbate
        angle: Rotation angle in degrees
        axis: Rotation axis ('x', 'y', or 'z')
        
    Returns:
        Rotated ASE Atoms
    """
    ads_copy = adsorbate.copy()
    ads_copy.rotate(angle, axis, center='COM')
    return ads_copy


def tilt_adsorbate(
    adsorbate: Atoms,
    tilt_angle: float,
    tilt_direction: str = 'x'
) -> Atoms:
    """
    Tilt an adsorbate from vertical.
    
    Args:
        adsorbate: ASE Atoms adsorbate
        tilt_angle: Tilt angle in degrees from vertical
        tilt_direction: Direction to tilt ('x' or 'y')
        
    Returns:
        Tilted ASE Atoms
    """
    ads_copy = adsorbate.copy()
    
    # Get center of first atom (binding site)
    binding_pos = ads_copy.get_positions()[0].copy()
    
    # Translate to origin
    ads_copy.translate(-binding_pos)
    
    # Rotate around x or y axis
    if tilt_direction.lower() == 'x':
        ads_copy.rotate(tilt_angle, 'x')
    else:
        ads_copy.rotate(tilt_angle, 'y')
    
    # Translate back
    ads_copy.translate(binding_pos)
    
    return ads_copy


def create_custom_adsorbate(
    atoms_data: List[Tuple[str, Tuple[float, float, float]]],
    height: float = 2.0
) -> Atoms:
    """
    Create a custom adsorbate from atom positions.
    
    Args:
        atoms_data: List of (element, (x, y, z)) tuples
        height: Height offset for the binding atom
        
    Returns:
        ASE Atoms adsorbate
    """
    atoms_list = []
    for i, (symbol, pos) in enumerate(atoms_data):
        # First atom is at height offset, others relative to it
        if i == 0:
            new_pos = (pos[0], pos[1], pos[2] + height)
        else:
            new_pos = (pos[0], pos[1], pos[2] + height)
        atoms_list.append(Atom(symbol, new_pos))
    
    adsorbate = Atoms(atoms_list)
    adsorbate.set_cell([20.0, 20.0, 20.0])
    adsorbate.center()
    adsorbate.set_pbc([True, True, True])
    
    return adsorbate
