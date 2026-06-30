# Materials Science Adsorption Agent
![MSA](ML_ads.png)
A Python agent for automated materials science workflows, focusing on surface adsorption analysis using Materials Project data, ASE/pymatgen tools, and FairChem UMA machine learning potentials.

## Features

1. **Materials Project Integration**: Retrieve crystal structures using MP API
2. **Structure Conversion**: Convert pymatgen structures to ASE Atoms objects
3. **Surface Cleaving**: Generate surface slabs with user-specified Miller indices and terminations
4. **Adsorbate Selection**: Choose from common adsorbates (H, O, CO, OH, N, etc.)
5. **Adsorption Site Analysis**: Find optimal adsorption sites using pymatgen algorithms
6. **ML-Accelerated DFT**: Calculate adsorption energies using FairChem UMA models
7. **Results Export**: Generate CIF files and comprehensive JSON reports

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up Materials Project API key
export MP_API_KEY="your_api_key_here"
# Or create a .env file with: MP_API_KEY=your_api_key_here
```

## Quick Start

### Command Line Interface

```bash
# Basic usage - analyze silicon (111) surface with H adsorbate
python -m agent_materials_science.cli --material Si --miller 1,1,1 --adsorbate H

# Using Materials Project ID
python -m agent_materials_science.cli --mp-id mp-149 --miller 1,1,0 --adsorbate CO

# Full workflow with energy calculations
python -m agent_materials_science.cli \
    --material SrTiO3 \
    --miller 1,0,0 \
    --adsorbate O \
    --layers 6 \
    --vacuum 15 \
    --supercell 2,2 \
    --calculate-energies \
    --output-dir outputs/srtio3_analysis
```

### Python API

```python
from agent_materials_science import MaterialsScienceAgent, AgentConfig

# Create agent configuration
config = AgentConfig(
    material="Si",
    miller_indices=(1, 1, 1),
    adsorbate="H",
    n_layers=6,
    vacuum=15.0,
    supercell=(2, 2),
    calculate_energies=True,
    output_dir="outputs"
)

# Run agent
agent = MaterialsScienceAgent(config)
result = agent.run()

# Access results
print(f"Material: {result.formula} ({result.material_id})")
print(f"Best adsorption site: {result.best_site}")
print(f"Output files: {result.output_files}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MP_API_KEY` | Materials Project API key | Required |
| `FAIRCHEM_MODEL` | FairChem model name | `uma-s-1p1` |
| `FAIRCHEM_DEVICE` | Device for ML calculations | `cpu` |

### Supported Adsorbates

- **Single atoms**: H, O, N, C, S
- **Molecules**: CO, OH, H2O, NH3, CO2, CH4

### FairChem Models

UMA models require **FairChem v2** and a Hugging Face account with access
granted to the UMA model repository (`huggingface-cli login`).

- `uma-s-1p2` - UMA small v1.2 (latest small model; recommended)
- `uma-s-1p1` - UMA small v1.1
- `uma-s-1` - UMA small v1
- `uma-m-1p1` - UMA medium v1.1 (higher accuracy, slower)
- `esen-*` - ESEN models

Pick the **task head** to match the problem: `oc20` for surface adsorption /
catalysis (the default here), `omat` for bulk inorganic materials, `omol` for
isolated molecules. Mixing task heads between the slab and the reference makes
adsorption energies meaningless.

### Adsorption-site backend

Site finding uses **pymatgen's `AdsorbateSiteFinder`** when pymatgen is
installed (`--site-finder auto`, the default). It is symmetry-aware, so it
returns only the distinct sites and avoids redundant energy evaluations. A
self-contained, periodicity-correct geometric finder is used as a fallback
(`--site-finder builtin`). Pass `--no-symm-reduce` to keep every site.

## Project Structure

```
agent_materials_science/
├── __init__.py           # Package exports
├── agent.py              # Main agent orchestration
├── cli.py                # Command line interface
├── config.py             # Configuration management
├── core/
│   ├── __init__.py
│   ├── adsorption.py     # Adsorption site finding
│   ├── surface.py        # Surface/slab generation
│   └── workflow.py       # Workflow orchestration
├── tools/
│   ├── __init__.py
│   ├── materials_project.py  # MP API wrapper
│   ├── ase_tools.py      # ASE utilities
│   ├── fairchem_calc.py  # FairChem calculator
│   └── converters.py     # Structure converters
├── outputs/              # Default output directory
├── requirements.txt
└── README.md
```

## Output Files

The agent generates the following output files:

- `{material_id}_slab_{hkl}.cif` - Clean slab structure
- `{material_id}_slab_{hkl}.vasp` - VASP POSCAR format
- `{material_id}_sites.json` - Adsorption sites data
- `{material_id}_best_site.cif` - Structure with adsorbate at best site
- `{material_id}_results.json` - Complete analysis results

## Methodology notes & limitations

- **Adsorption energies are approximate.** `E_ads = E(slab+ads) − E(slab) −
  E_ref(ads)` uses tabulated gas-phase reference energies that are *not*
  recomputed with the active model/task, so absolute values are indicative
  rather than publication-grade. For rigorous numbers, compute the references
  with the same model and task.
- **Single fixed placement per site.** Each site is evaluated at one initial
  geometry (optionally relaxed). The state-of-the-art recipe is
  [AdsorbML](https://www.nature.com/articles/s41524-023-01121-5): generate many
  initial configurations, ML-relax each, and take the minimum. Consider that
  workflow (or higher-level libraries such as
  [`quacc`](https://quacc.readthedocs.io) `slab_to_ads_flow` and `atomate2`)
  for production studies.
- **Surface-normal assumption.** Site finding assumes the surface lies in the
  xy-plane with vacuum along z (the convention produced by the slab builder).
- **`OH` reference energy is a placeholder** and should be verified before use.

## License

MIT License
