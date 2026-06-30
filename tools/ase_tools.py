"""
ASE (Atomic Simulation Environment) utilities for structure manipulation.
"""

import os
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
from ase.atoms import Atoms
from ase.build import surface
from ase.io import write, read
from ase.constraints import FixAtoms

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor


def structure_to_atoms(struct: Structure) -> Atoms:
    """
    Convert a pymatgen Structure to an ASE Atoms object.
    
    Args:
        struct: pymatgen Structure object
        
    Returns:
        ASE Atoms object
    """
    adaptor = AseAtomsAdaptor()
    return adaptor.get_atoms(struct)


def atoms_to_structure(atoms: Atoms) -> Structure:
    """
    Convert an ASE Atoms object to a pymatgen Structure.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        pymatgen Structure object
    """
    adaptor = AseAtomsAdaptor()
    return adaptor.get_structure(atoms)


def build_slab(
    bulk: Atoms,
    miller: Tuple[int, int, int],
    layers: int = 6,
    vacuum: float = 15.0,
    center: bool = True,
    periodic: bool = True,
) -> Atoms:
    """
    Build a surface slab from a bulk Atoms object using ASE.
    
    Args:
        bulk: ASE Atoms bulk structure
        miller: Miller indices (h, k, l)
        layers: Number of atomic layers
        vacuum: Vacuum thickness in Angstrom
        center: Whether to center the slab in the cell
        periodic: Whether to apply periodic boundary conditions
        
    Returns:
        ASE Atoms slab
    """
    h, k, l = miller
    slab = surface(bulk, (h, k, l), layers=layers, vacuum=vacuum, periodic=periodic)
    
    if center:
        # Center slab in the z-direction
        slab.center(vacuum=vacuum / 2, axis=2)
    
    return slab


def build_slab_with_termination(
    structure: Structure,
    miller: Tuple[int, int, int],
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 15.0,
    termination_index: int = 0,
    center_slab: bool = True,
) -> Tuple[Atoms, List[Dict[str, Any]]]:
    """
    Build a surface slab with specific termination using pymatgen's SlabGenerator.
    
    This method provides better control over surface terminations compared
    to the simple ASE surface() function.
    
    Args:
        structure: pymatgen Structure (bulk)
        miller: Miller indices (h, k, l)
        min_slab_size: Minimum slab thickness in Angstrom
        min_vacuum_size: Minimum vacuum thickness in Angstrom
        termination_index: Index of termination to use (0 = most stable)
        center_slab: Whether to center the slab
        
    Returns:
        Tuple of (ASE Atoms slab, list of available terminations info)
    """
    h, k, l = miller
    
    # Generate all possible slabs with different terminations
    slabgen = SlabGenerator(
        initial_structure=structure,
        miller_index=(h, k, l),
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=center_slab,
        in_unit_planes=False,
        primitive=True,
        max_normal_search=max(abs(h), abs(k), abs(l)) + 1,
    )
    
    slabs = slabgen.get_slabs()
    
    if not slabs:
        raise ValueError(
            f"Could not generate slabs for Miller indices ({h},{k},{l}). "
            "Try different Miller indices or check the bulk structure."
        )
    
    # Collect termination information
    terminations = []
    for i, slab in enumerate(slabs):
        term_info = {
            "index": i,
            "formula": slab.composition.reduced_formula,
            "n_atoms": len(slab),
            "is_symmetric": slab.is_symmetric(),
            "is_polar": slab.is_polar(),
            "surface_area": slab.surface_area,
        }
        terminations.append(term_info)
    
    # Select the requested termination
    if termination_index >= len(slabs):
        print(f"Warning: Requested termination {termination_index} not available. "
              f"Using termination 0 (of {len(slabs)} available).")
        termination_index = 0
    
    selected_slab = slabs[termination_index]
    
    # Convert to ASE Atoms
    atoms = structure_to_atoms(selected_slab)
    
    return atoms, terminations


def apply_supercell(atoms: Atoms, nx: int, ny: int, nz: int = 1) -> Atoms:
    """
    Apply supercell repetition to the structure.
    
    Args:
        atoms: ASE Atoms object
        nx: Repetitions in x direction
        ny: Repetitions in y direction
        nz: Repetitions in z direction (default 1 for slabs)
        
    Returns:
        Repeated ASE Atoms object
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Supercell repeats must be positive integers.")
    return atoms.repeat((nx, ny, nz))


def fix_bottom_layers(
    atoms: Atoms,
    n_layers: int = 2,
    tolerance: float = 0.5
) -> Atoms:
    """
    Fix the bottom layers of a slab to simulate bulk behavior.
    
    Args:
        atoms: ASE Atoms slab
        n_layers: Number of bottom layers to fix
        tolerance: Tolerance for identifying layers (Angstrom)
        
    Returns:
        ASE Atoms with constraints
    """
    atoms_copy = atoms.copy()
    positions = atoms_copy.get_positions()
    z_coords = positions[:, 2]

    # Cluster atoms into layers: two atoms belong to the same layer if their
    # z-coordinates are within ``tolerance``. Using the raw set of unique
    # floating-point z-values (as a naive implementation does) treats almost
    # every atom as its own "layer" because of numerical noise.
    order = np.argsort(z_coords)
    layers = []  # list of representative z for each layer (ascending)
    layer_of_atom = {}
    for idx in order:
        z = z_coords[idx]
        if layers and abs(z - layers[-1]) < tolerance:
            layer_id = len(layers) - 1
        else:
            layers.append(z)
            layer_id = len(layers) - 1
        layer_of_atom[idx] = layer_id

    # Fix the atoms that belong to the lowest ``n_layers`` layers.
    n_fix = min(n_layers, len(layers))
    fixed_indices = [int(i) for i, lid in layer_of_atom.items() if lid < n_fix]

    if fixed_indices:
        constraint = FixAtoms(indices=sorted(fixed_indices))
        atoms_copy.set_constraint(constraint)

    return atoms_copy


def save_outputs(
    atoms: Atoms,
    outdir: str,
    base_name: str,
    formats: List[str] = None
) -> List[str]:
    """
    Save structure to various file formats.
    
    Args:
        atoms: ASE Atoms object
        outdir: Output directory
        base_name: Base name for output files
        formats: List of output formats (default: ['cif', 'vasp'])
        
    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ['cif', 'vasp']
    
    os.makedirs(outdir, exist_ok=True)
    paths = []
    
    for fmt in formats:
        if fmt == 'cif':
            path = os.path.join(outdir, f"{base_name}.cif")
            write(path, atoms, format='cif')
        elif fmt == 'vasp':
            path = os.path.join(outdir, f"POSCAR_{base_name}")
            write(path, atoms, format='vasp')
        elif fmt == 'xyz':
            path = os.path.join(outdir, f"{base_name}.xyz")
            write(path, atoms, format='xyz')
        elif fmt == 'json':
            path = os.path.join(outdir, f"{base_name}.json")
            write(path, atoms, format='json')
        else:
            path = os.path.join(outdir, f"{base_name}.{fmt}")
            write(path, atoms, format=fmt)
        
        paths.append(path)
    
    return paths


def get_slab_info(atoms: Atoms) -> Dict[str, Any]:
    """
    Get information about a slab structure.
    
    Args:
        atoms: ASE Atoms slab
        
    Returns:
        Dictionary with slab information
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    
    # Calculate slab thickness
    z_coords = positions[:, 2]
    thickness = z_coords.max() - z_coords.min()
    
    # Get chemical composition
    symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    composition = {el: symbols.count(el) for el in unique_elements}
    
    # Estimate number of layers
    z_unique = sorted(set(round(z, 2) for z in z_coords))
    n_layers_estimate = len(z_unique)
    
    # Get surface area as the magnitude of the cross product of the two
    # in-plane cell vectors. Using only the z-component (a_x*b_y - a_y*b_x)
    # is wrong whenever the cell vectors are not aligned with the xy-plane,
    # which is common for pymatgen-generated slabs.
    surface_area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    
    return {
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "elements": unique_elements,
        "composition": composition,
        "thickness": round(thickness, 3),
        "n_layers_estimate": n_layers_estimate,
        "surface_area": round(surface_area, 3),
        "cell": cell.tolist(),
        "pbc": atoms.pbc.tolist(),
    }


def get_surface_atoms(
    atoms: Atoms,
    tolerance: float = 1.0
) -> List[int]:
    """
    Identify surface atoms (atoms at the top of the slab).
    
    Args:
        atoms: ASE Atoms slab
        tolerance: Tolerance for identifying top layer (Angstrom)
        
    Returns:
        List of atom indices at the surface
    """
    positions = atoms.get_positions()
    z_coords = positions[:, 2]
    max_z = z_coords.max()
    
    surface_indices = [
        i for i, z in enumerate(z_coords)
        if abs(z - max_z) < tolerance
    ]
    
    return surface_indices


def load_structure(filepath: str) -> Atoms:
    """
    Load a structure from file.
    
    Args:
        filepath: Path to structure file (CIF, POSCAR, XYZ, etc.)
        
    Returns:
        ASE Atoms object
    """
    return read(filepath)
