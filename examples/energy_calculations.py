"""
Advanced example: Adsorption energy calculations with FairChem UMA.

This example demonstrates:
1. Using FairChem ML potentials for energy calculations
2. Analyzing multiple adsorbates
3. Comparing adsorption sites

Requirements:
- Set MP_API_KEY environment variable
- FairChem must be installed: pip install fairchem-core
"""

from agent_materials_science import MaterialsScienceAgent, AgentConfig
from agent_materials_science.tools.fairchem_calc import FAIRCHEM_AVAILABLE, check_fairchem_installation


def analyze_single_adsorbate():
    """Analyze CO adsorption on Pt(111)."""
    
    print("=" * 60)
    print("CO Adsorption on Pt(111)")
    print("=" * 60)
    
    config = AgentConfig(
        material="Pt",
        miller_indices=(1, 1, 1),
        adsorbate="CO",
        n_layers=6,
        vacuum=15.0,
        supercell=(2, 2),
        calculate_energies=True,      # Enable energy calculations
        fairchem_model="uma-s-1p2",   # UMA-Small v1.2 (uma-s-1 was removed upstream)
        device="auto",                # "auto" uses CUDA when available
        output_dir="outputs/pt_co",
        verbose=True,
    )
    
    agent = MaterialsScienceAgent(config)
    result = agent.run()
    
    print("\n" + result.summary())
    return result


def compare_adsorbates():
    """Compare different adsorbates on the same surface."""
    
    print("\n" + "=" * 60)
    print("Comparing Adsorbates on Pt(111)")
    print("=" * 60)
    
    adsorbates = ["H", "O", "CO", "OH"]
    results = {}
    
    for ads in adsorbates:
        print(f"\n--- Analyzing {ads} ---")
        
        config = AgentConfig(
            material="Pt",
            miller_indices=(1, 1, 1),
            adsorbate=ads,
            n_layers=6,
            vacuum=15.0,
            supercell=(2, 2),
            calculate_energies=True,
            fairchem_model="uma-s-1p2",
            device="auto",
            output_dir=f"outputs/pt_{ads.lower()}",
            verbose=False,  # Quieter output for comparison
        )
        
        try:
            agent = MaterialsScienceAgent(config)
            result = agent.run()
            results[ads] = result
            
            if result.best_site and result.best_site.get("energy") is not None:
                print(f"  Best site: {result.best_site['site_type']}")
                print(f"  Adsorption energy: {result.best_site['energy']:.3f} eV")
            else:
                print(f"  Best site: {result.best_site['site_type'] if result.best_site else 'None'}")
                
        except Exception as e:
            print(f"  Error: {e}")
            results[ads] = None
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary: Adsorption Energies on Pt(111)")
    print("=" * 60)
    print(f"{'Adsorbate':<12} {'Best Site':<10} {'E_ads (eV)':<12}")
    print("-" * 36)
    
    for ads, result in results.items():
        if result and result.best_site:
            site_type = result.best_site.get("site_type", "?")
            energy = result.best_site.get("energy")
            energy_str = f"{energy:.3f}" if energy is not None else "N/A"
            print(f"{ads:<12} {site_type:<10} {energy_str:<12}")
        else:
            print(f"{ads:<12} {'Error':<10} {'N/A':<12}")
    
    return results


def main():
    # Check FairChem installation
    print("Checking FairChem installation...")
    status = check_fairchem_installation()
    
    if not status["fairchem_available"]:
        print("\n⚠️  FairChem is not installed!")
        print("Install with: pip install fairchem-core")
        print("Continuing without energy calculations...\n")
    else:
        print("✓ FairChem is available")
        if status.get("cuda_available"):
            print(f"✓ CUDA available (torch {status.get('torch_version')})")
        else:
            print("  CUDA not available (using CPU)")
    
    print()
    
    # Run single analysis
    if FAIRCHEM_AVAILABLE:
        result = analyze_single_adsorbate()
        
        # Optionally compare adsorbates (can take a while)
        # compare_adsorbates()
    else:
        print("Skipping energy calculations - FairChem not available")
        
        # Run without energy calculations
        config = AgentConfig(
            material="Pt",
            miller_indices=(1, 1, 1),
            adsorbate="CO",
            calculate_energies=False,
            output_dir="outputs/pt_co_no_energy",
            verbose=True,
        )
        
        agent = MaterialsScienceAgent(config)
        result = agent.run()
        print("\n" + result.summary())


if __name__ == "__main__":
    main()
