# Materials Science Adsorption Agent

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
| `FAIRCHEM_MODEL` | FairChem model name | `uma-s-1` |
| `FAIRCHEM_DEVICE` | Device for ML calculations | `cpu` |

### Supported Adsorbates

- **Single atoms**: H, O, N, C, S
- **Molecules**: CO, OH, H2O, NH3, CO2, CH4

### FairChem Models

- `uma-s-1` (default) - Universal Materials Accelerator small
- `uma-s-1p1` - UMA small v1.1
- `uma-m-1p1` - UMA medium v1.1
- `esen-md-direct-all-omol` - ESEN model for dynamics

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

## License

MIT License
