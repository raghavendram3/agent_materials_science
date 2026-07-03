"""
Basic usage example for the Materials Science Agent.

This example demonstrates:
1. Creating an agent configuration
2. Running the analysis workflow
3. Accessing results

Requirements:
- Set MP_API_KEY environment variable with your Materials Project API key
"""

from agent_materials_science import MaterialsScienceAgent, AgentConfig


def main():
    # Create configuration for analyzing Si(111) surface with H adsorbate
    config = AgentConfig(
        material="Si",                    # Chemical formula
        miller_indices=(1, 1, 1),         # Miller indices for surface
        adsorbate="H",                    # Adsorbate species
        n_layers=6,                       # Number of slab layers
        vacuum=15.0,                      # Vacuum thickness (Å)
        supercell=(2, 2),                 # In-plane supercell
        output_dir="outputs/si_example",  # Output directory
        verbose=True,                     # Print progress
    )
    
    # Create and run agent
    agent = MaterialsScienceAgent(config)
    result = agent.run()
    
    # Print summary
    print("\n" + result.summary())
    
    # Access specific results
    print(f"\nMaterial ID: {result.material_id}")
    print(f"Formula: {result.formula}")
    print(f"Number of sites found: {len(result.sites)}")
    
    if result.best_site:
        print(f"\nBest adsorption site:")
        print(f"  Type: {result.best_site['site_type']}")
        print(f"  Position: {result.best_site['position']}")
        if result.best_site.get('energy') is not None:
            print(f"  Energy: {result.best_site['energy']:.3f} eV")
    
    print(f"\nOutput files:")
    for f in result.output_files:
        print(f"  - {f}")
    
    return result


if __name__ == "__main__":
    main()
