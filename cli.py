"""
Command Line Interface for the Materials Science Agent.

Usage:
    python -m agent_materials_science.cli --material Si --miller 1,1,1 --adsorbate H
"""

import argparse
import sys
import json
from typing import Tuple

from .config import AgentConfig, get_available_adsorbates, get_available_models
from .agent import MaterialsScienceAgent, run_analysis
from .core.surface import string_to_miller


def parse_miller(s: str) -> Tuple[int, int, int]:
    """Parse Miller indices from string."""
    try:
        return string_to_miller(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_supercell(s: str) -> Tuple[int, int]:
    """Parse supercell dimensions from string."""
    try:
        parts = s.replace(" ", "").split(",")
        if len(parts) == 1:
            n = int(parts[0])
            return (n, n)
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        else:
            raise ValueError("Supercell must be 'N' or 'N,M'")
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid supercell: {s}")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent_materials_science",
        description="Materials Science Adsorption Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python -m agent_materials_science.cli --material Si --miller 1,1,1 --adsorbate H

  # Using Materials Project ID
  python -m agent_materials_science.cli --mp-id mp-149 --miller 1,1,0 --adsorbate CO

  # Full workflow with energy calculations
  python -m agent_materials_science.cli \\
      --material Pt \\
      --miller 1,1,1 \\
      --adsorbate CO \\
      --supercell 2,2 \\
      --calculate-energies \\
      --output-dir outputs/pt_analysis

  # List available adsorbates
  python -m agent_materials_science.cli --list-adsorbates

  # Check installation
  python -m agent_materials_science.cli --check-install
""",
    )
    
    # Material specification
    material_group = parser.add_argument_group("Material")
    material_group.add_argument(
        "--material", "-m",
        type=str,
        help="Chemical formula (e.g., 'Si', 'SrTiO3', 'Pt')"
    )
    material_group.add_argument(
        "--mp-id",
        type=str,
        help="Materials Project ID (e.g., 'mp-149')"
    )
    
    # Surface parameters
    surface_group = parser.add_argument_group("Surface")
    surface_group.add_argument(
        "--miller", "-k",
        type=parse_miller,
        default=(1, 1, 1),
        help="Miller indices (e.g., '1,1,1' or '110'). Default: 1,1,1"
    )
    surface_group.add_argument(
        "--termination", "-t",
        type=int,
        default=0,
        help="Surface termination index (0 = most stable). Default: 0"
    )
    surface_group.add_argument(
        "--layers", "-l",
        type=int,
        default=6,
        help="Number of atomic layers. Default: 6"
    )
    surface_group.add_argument(
        "--vacuum", "-v",
        type=float,
        default=15.0,
        help="Vacuum thickness in Angstrom. Default: 15.0"
    )
    surface_group.add_argument(
        "--supercell", "-s",
        type=parse_supercell,
        default=(1, 1),
        help="In-plane supercell (e.g., '2,2' or '3'). Default: 1,1"
    )
    
    # Adsorbate parameters
    adsorbate_group = parser.add_argument_group("Adsorbate")
    adsorbate_group.add_argument(
        "--adsorbate", "-a",
        type=str,
        default="H",
        help="Adsorbate species (e.g., 'H', 'CO', 'OH'). Default: H"
    )
    adsorbate_group.add_argument(
        "--height",
        type=float,
        default=2.0,
        help="Height offset above surface in Angstrom. Default: 2.0"
    )
    adsorbate_group.add_argument(
        "--site-types",
        type=str,
        nargs="+",
        default=["top", "bridge", "hollow"],
        help="Site types to consider. Default: top bridge hollow"
    )
    adsorbate_group.add_argument(
        "--site-finder",
        type=str,
        choices=["auto", "builtin", "pymatgen"],
        default="auto",
        help="Adsorption-site backend. 'auto' prefers pymatgen's "
             "symmetry-aware AdsorbateSiteFinder when installed. Default: auto"
    )
    adsorbate_group.add_argument(
        "--no-symm-reduce",
        action="store_true",
        help="Keep all symmetry-equivalent sites (pymatgen backend only)"
    )
    
    # Calculation parameters
    calc_group = parser.add_argument_group("Calculations")
    calc_group.add_argument(
        "--calculate-energies", "-e",
        action="store_true",
        help="Calculate adsorption energies using FairChem"
    )
    calc_group.add_argument(
        "--relax",
        action="store_true",
        help="Relax structures before energy calculations"
    )
    calc_group.add_argument(
        "--model",
        type=str,
        default="uma-s-1p1",
        help="FairChem model name. Default: uma-s-1p1"
    )
    calc_group.add_argument(
        "--task",
        type=str,
        default="oc20",
        help="UMA task head ('oc20' for adsorption/catalysis, 'omat' for "
             "bulk materials, 'omol' for molecules). Default: oc20"
    )
    calc_group.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for calculations (if available)"
    )
    
    # Output parameters
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs",
        help="Output directory. Default: outputs"
    )
    output_group.add_argument(
        "--save-all-sites",
        action="store_true",
        help="Save structures for all sites (not just best)"
    )
    output_group.add_argument(
        "--json",
        type=str,
        metavar="FILE",
        help="Save results to JSON file"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    # Utility commands
    utility_group = parser.add_argument_group("Utility")
    utility_group.add_argument(
        "--list-adsorbates",
        action="store_true",
        help="List available adsorbate species and exit"
    )
    utility_group.add_argument(
        "--list-models",
        action="store_true",
        help="List available FairChem models and exit"
    )
    utility_group.add_argument(
        "--check-install",
        action="store_true",
        help="Check installation status and exit"
    )
    utility_group.add_argument(
        "--list-terminations",
        action="store_true",
        help="List available surface terminations and exit"
    )
    
    return parser


def main(argv=None):
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Handle utility commands
    if args.list_adsorbates:
        print("\nAvailable adsorbates:")
        print("-" * 40)
        for ads in get_available_adsorbates():
            print(f"  {ads}")
        print()
        return 0
    
    if args.list_models:
        print("\nAvailable FairChem models:")
        print("-" * 40)
        for model in get_available_models():
            print(f"  {model}")
        print()
        return 0
    
    if args.check_install:
        print("\nInstallation status:")
        print("-" * 40)
        status = MaterialsScienceAgent.check_installation()
        
        # Check core dependencies
        print(f"  Materials Project API: {'✓' if status.get('mp_api') else '✗'}")
        print(f"  Pymatgen: {'✓' if status.get('pymatgen') else '✗'}")
        print(f"  ASE: {'✓' if status.get('ase') else '✗'}")
        
        # FairChem status
        fc = status.get("fairchem", {})
        print(f"  FairChem: {'✓' if fc.get('fairchem_available') else '✗'}")
        if fc.get('cuda_available'):
            print(f"  CUDA: ✓ (torch {fc.get('torch_version', 'unknown')})")
        else:
            print(f"  CUDA: ✗ (CPU only)")
        
        print()
        return 0
    
    if args.list_terminations:
        if not args.material and not args.mp_id:
            print("Error: --material or --mp-id required for --list-terminations")
            return 1
        
        from .tools.materials_project import MaterialsProjectTool
        from .core.surface import SurfaceBuilder
        
        mp = MaterialsProjectTool()
        
        if args.mp_id:
            structure = mp.get_structure_by_mp_id(args.mp_id)
            mat_id = args.mp_id
        else:
            structure, mat_id = mp.get_structure_by_formula(args.material)
        
        builder = SurfaceBuilder(structure, args.miller)
        builder.print_terminations()
        return 0
    
    # Validate required arguments for analysis
    if not args.material and not args.mp_id:
        parser.print_help()
        print("\nError: --material or --mp-id is required")
        return 1
    
    # Create configuration
    config = AgentConfig(
        material=args.material or "",
        mp_id=args.mp_id,
        miller_indices=args.miller,
        termination=args.termination,
        n_layers=args.layers,
        vacuum=args.vacuum,
        supercell=args.supercell,
        adsorbate=args.adsorbate,
        height_offset=args.height,
        site_types=args.site_types,
        site_finder=args.site_finder,
        symm_reduce=not args.no_symm_reduce,
        calculate_energies=args.calculate_energies,
        relax_structures=args.relax,
        fairchem_model=args.model,
        fairchem_task=args.task,
        use_gpu=args.gpu,
        output_dir=args.output_dir,
        save_all_sites=args.save_all_sites,
        verbose=not args.quiet,
    )
    
    # Run analysis
    try:
        agent = MaterialsScienceAgent(config)
        result = agent.run()
        
        # Print summary
        if not args.quiet:
            print("\n" + result.summary())
        
        # Save JSON if requested
        if args.json:
            result.to_json(args.json)
            if not args.quiet:
                print(f"\nResults saved to: {args.json}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
