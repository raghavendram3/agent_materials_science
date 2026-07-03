"""
Interactive workflow example using the step-by-step agent.

This example demonstrates:
1. Step-by-step control over the analysis
2. Inspecting intermediate results
3. Choosing surface terminations
4. Manual site selection

Requirements:
- Set MP_API_KEY environment variable
"""

from agent_materials_science.agent import InteractiveAgent
from agent_materials_science.tools.ase_tools import save_outputs


def main():
    print("=" * 60)
    print("Interactive Materials Science Workflow")
    print("=" * 60)
    
    # Initialize the interactive agent
    agent = InteractiveAgent()
    
    # Step 1: Fetch material
    print("\n[Step 1] Fetching material from Materials Project...")
    material = "SrTiO3"  # Perovskite oxide
    props = agent.fetch_material(material)
    
    print(f"\nMaterial: {props['formula']} ({props['material_id']})")
    print(f"Space group: {props.get('space_group', 'N/A')}")
    print(f"Crystal system: {props.get('crystal_system', 'N/A')}")
    print(f"Band gap: {props.get('band_gap', 'N/A')} eV")
    print(f"Elements: {', '.join(props.get('elements', []))}")
    
    # Step 2: List available terminations
    print("\n[Step 2] Checking available surface terminations...")
    miller = (1, 0, 0)
    terminations = agent.list_terminations(miller=miller)
    
    print(f"\nAvailable terminations for {material} {miller}:")
    print("-" * 50)
    for t in terminations:
        polar_str = " (polar)" if t.get("is_polar") else ""
        symm_str = " (symmetric)" if t.get("is_symmetric") else ""
        print(f"  [{t['index']}] {t['formula']}: {t['n_atoms']} atoms{polar_str}{symm_str}")
    
    # Step 3: Create slab with chosen termination
    print("\n[Step 3] Creating surface slab...")
    selected_termination = 0  # Choose most stable (first)
    
    slab_info = agent.create_slab(
        miller=miller,
        termination=selected_termination,
        min_slab_size=10.0,
        vacuum=15.0,
        supercell=(2, 2),
    )
    
    print(f"\nSlab created:")
    print(f"  Formula: {slab_info['formula']}")
    print(f"  Atoms: {slab_info['n_atoms']}")
    print(f"  Thickness: {slab_info['thickness']:.2f} Å")
    print(f"  Surface area: {slab_info['surface_area']:.2f} Å²")
    
    # Step 4: Find adsorption sites
    print("\n[Step 4] Finding adsorption sites...")
    sites = agent.find_sites(height_offset=2.0)
    
    print(f"\nFound {len(sites)} adsorption sites:")
    site_types = {}
    for s in sites:
        st = s["site_type"]
        site_types[st] = site_types.get(st, 0) + 1
    
    for st, count in site_types.items():
        print(f"  - {st}: {count}")
    
    # Step 5: Select adsorbate
    print("\n[Step 5] Selecting adsorbate...")
    ads_info = agent.select_adsorbate("O")
    
    print(f"\nAdsorbate: {ads_info['species']}")
    print(f"  Atoms: {ads_info['n_atoms']}")
    print(f"  Elements: {', '.join(ads_info['elements'])}")
    
    # Step 6: Examine specific sites
    print("\n[Step 6] Examining adsorption sites...")
    print("\nTop 5 sites (by coordination):")
    print("-" * 60)
    print(f"{'#':<4} {'Type':<10} {'Position (x,y,z)':<30} {'Coord':<6}")
    print("-" * 60)
    
    for i, site in enumerate(sites[:5]):
        pos = site["position"]
        pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        coord = len(site["coordinating_atoms"])
        print(f"{i:<4} {site['site_type']:<10} {pos_str:<30} {coord:<6}")
    
    # Step 7: Save outputs
    print("\n[Step 7] Saving structures...")
    slab = agent.get_slab()
    output_dir = "outputs/interactive_example"
    
    files = save_outputs(slab, output_dir, f"srtio3_100_slab")
    
    print(f"\nSaved files:")
    for f in files:
        print(f"  - {f}")
    
    print("\n" + "=" * 60)
    print("Workflow completed!")
    print("=" * 60)
    
    return agent


if __name__ == "__main__":
    main()
