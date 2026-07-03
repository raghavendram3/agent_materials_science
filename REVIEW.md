# Code Review: agent_materials_science

Review of https://github.com/raghavendram3/agent_materials_science (main branch, July 2026), with fixes applied in the accompanying `agent_materials_science_fixed` package. Every issue below was verified by running the code against the current libraries (pymatgen 2026.5.4, ase 3.29.0, mp-api 0.46.4) and by inspecting the fairchem-core 2.21.0 source from PyPI.

## Critical bugs (verified)

### 1. The package is uninstallable
`pyproject.toml` declares `packages = ["agent_materials_science", ...]`, but the modules live at the repository root — there is no `agent_materials_science/` directory. `pip install .` fails with a build error (verified), so the console script and the README's install instructions cannot work. `python -m agent_materials_science.cli` only works by accident when the clone happens to be named `agent_materials_science` and you run from its parent directory.

**Fix:** moved all modules into an `agent_materials_science/` package directory. `pip install .` now works, the `agent-materials-science` console script is installed, and the tests import the installed package.

### 2. Physics bug: inconsistent relaxation over-estimates binding
In `workflow._calculate_energies`, when `relax_structures=True` only the slab+adsorbate system was relaxed; `E_slab` was a single-point energy of the *unrelaxed* clean slab. The relaxed combined system gains the slab-relaxation energy, which is never subtracted, so every adsorption energy is systematically too negative. (Ironically, `FairchemCalculator.calculate_adsorption_energy` handled this correctly — the workflow re-implemented the logic inline and got it wrong.)

**Fix:** with `--relax`, the clean slab is now relaxed once *before* site finding (so sites are also located on the relaxed surface geometry), and each slab+adsorbate system is relaxed; both totals in `E_ads = E(slab+ads) − E(slab) − E_gas` come from consistent relaxed references.

### 3. Wrong gas-phase reference energies (up to 2.3 eV off)
`ADSORBATE_REFERENCE_ENERGIES` hardcoded "typical DFT values" that disagree with the OC20 convention the UMA `oc20` head is meant to pair with: O was −4.93 eV vs the official −7.204 eV (a 2.27 eV error propagating 1:1 into every O adsorption energy), and the OH entry was an acknowledged copy-paste of the N value.

**Fix:** replaced with the official OC20 linear atomic reference scheme (H −3.477, C −7.282, O −7.204, N −8.083 eV — OC20 paper Table 5, reproduced in the FairChem UMA adsorption-energy tutorial). `get_adsorbate_reference_energy()` computes E_gas compositionally for *any* C/H/O/N adsorbate (27 supported species instead of 13 tabulated ones), accepts per-element overrides for other elements, and reports a per-element breakdown in the output JSON. The FairChem docs explicitly recommend this scheme and advise against computing gas references with the `omol` head; the module docstring now documents that.

### 4. Custom checkpoints crash: nonexistent API
`pretrained_mlip.get_predict_unit(checkpoint_path=...)` — there is no `checkpoint_path` keyword in fairchem-core 2.x (verified against the 2.21.0 source), so any user passing a local checkpoint got a `TypeError`.

**Fix:** uses `FAIRChemCalculator.from_model_checkpoint(name_or_path, task_name=..., device=..., inference_settings=...)`, the modern single entry point that accepts both registry names and local paths, with a `load_predict_unit`/`get_predict_unit` fallback for older 2.x releases.

### 5. Stale slab cache returns wrong structures
`SurfaceBuilder` cached slabs/terminations without keying on parameters: requesting a 30 Å slab after a 10 Å terminations call silently returned the stale 10 Å slabs (verified — same atom count).

**Fix:** caches keyed by `(min_slab_size, min_vacuum_size, in_unit_planes, center_slab)`; regression test included.

### 6. Hardcoded neighbor cutoff misses whole site classes
`neighbor_cutoff=3.5 Å` found **zero bridge sites on Si(111)** (surface NN distance 3.84 Å) — verified with the built-in finder. Any surface with spacing above 3.5 Å silently loses bridge/hollow sites.

**Fix:** `neighbor_cutoff=None` (new default) auto-estimates the cutoff as 1.2× the top-layer nearest-neighbor distance (minimum-image), clipped to [2.0, 6.0] Å. Regression test asserts bridges are found on Si(111).

### 7. Environment variables silently override explicit arguments
`AgentConfig(fairchem_model="uma-m-1p1")` returned `uma-s-1p1` if `FAIRCHEM_MODEL=uma-s-1p1` was set (verified) — inverted precedence, same for task and device.

**Fix:** sentinel `None` defaults with standard precedence: explicit argument > environment variable > built-in default. Tested.

## Other bugs and staleness

- **`uma-s-1` no longer exists.** It was removed from the FairChem registry (current: `uma-s-1p2`, `uma-s-1p1`, `uma-m-1p1`, `esen-*`, …); the repo listed it as available and the examples used it. Fixed: default is now `uma-s-1p2`, `get_available_models()` queries the live registry when fairchem is installed, and the examples were updated.
- **Wrong dependency claims.** The repo advertises Python ≥3.8 and `torch>=2.0`; fairchem-core 2.21 requires **Python ≥3.11,<3.14** and **torch ~=2.8** (verified from wheel metadata). `pyproject.toml`/`requirements.txt` updated; fairchem is now an optional `[ml]` extra so the geometry workflow (slabs + sites) installs without a ~2 GB torch download.
- **CLI hid more than half the supported adsorbates.** `--list-adsorbates` read `config.ADSORBATES` (13 species) while the geometry table in `converters` supports 27 (verified mismatch: C2H2, C2H4, CH3OH, Cl, F, H2, HCOO, HCl, HF, HO, N2, N2O, O2, OC, ON missing). Fixed: single source of truth (the geometry table), with descriptions.
- **Full-database footgun.** `search_materials()` with no criteria queried the entire MP summary collection (~150k documents). Fixed: requires at least one criterion and bounds the download with `num_chunks=1` and a chunk size matched to `max_results`.
- **`n_layers` didn't mean layers.** Documented as "atomic layers" but converted via a 2.5 Å/layer heuristic. Fixed: `layers_in_unit_planes=True` (new default) passes `n_layers` as (hkl) planes via pymatgen's `in_unit_planes`; the heuristic remains available as an opt-out.
- **Dead/missing config wiring.** `center_slab` existed in config but was never passed to the slab generator; `fix_bottom_layers` existed in the tools but was not exposed, so relaxations moved the entire slab including "bulk" layers. Fixed: both wired through, plus new `--fix-layers` and `--device {auto,cpu,cuda}` CLI flags (`--gpu` kept as a deprecated alias).
- **Saved "best" structure didn't match its energy.** With relaxation on, the reported energy came from the relaxed geometry but the saved structure was the unrelaxed placement. Fixed: relaxed structures are stored and saved (also for `--save-all-sites`), with convergence info recorded in the site metadata and output JSON.
- **`get_structure_by_mp_id`** now uses the canonical `MPRester.get_structure_by_material_id` (handles deprecated/re-mapped IDs).
- **POSCAR output** now passes `sort=True` so species are grouped into single blocks.
- **Version mismatch** (`pyproject` 0.1.0 vs `__init__` 1.0.0) resolved to 0.2.0 in both; `run_analysis` and `InteractiveAgent` are now exported from the package root as the README/examples imply.
- **Repo hygiene:** committed `__pycache__/` and `outputs/` removed; `.gitignore` added. (The upstream repo also deleted its tests and CI at some point — a `tests/` suite with 44 passing tests is restored here.)

## Upgrade opportunities with current tools

Beyond the fixes, worth knowing about:

1. **Single-atom energies now work natively.** fairchem-core ≥2.21 handles isolated single atoms (`pbc=False`) via bundled DFT atomic references — the repo's central "FairChem cannot calculate single atoms" premise is outdated. A `calculate_gas_energy()` method was added; the OC20 tabulated scheme remains the default because it is the convention-consistent choice for the `oc20` head.
2. **AdsorbML is now a fairchem-core extra** (`pip install "fairchem-core[adsorbml]"`). For production adsorption energies, the AdsorbML recipe (many placements → ML-relax all → take the minimum) is the state of the art; this package's one-placement-per-site approach is a fast screen by comparison. Exposed as the `[adsorbml]` extra and documented.
3. **Turbo inference.** `inference_settings="turbo"` is passed through for users relaxing a single system repeatedly (it requires a fixed composition across calls, so it's off by default).
4. **Dynamic model registry.** `--list-models` now reflects `pretrained_mlip.available_models` at runtime, so future model additions/removals won't go stale.
5. **Device auto-detection.** `device="auto"` picks CUDA when available; env `FAIRCHEM_DEVICE` respected with correct precedence.

## Verification

- 44 pytest tests pass (config precedence, cache keying, Si(111) bridge-site regression, OC20 reference arithmetic, placement/constraint preservation, module import without fairchem).
- `pip install .` succeeds; console script and all utility commands run.
- End-to-end workflow smoke test (mocked MP fetch, Si(111) + OH, fix_layers=2) produces sites, structures, and JSON with no errors.
- FairChem-dependent paths verified against the fairchem-core 2.21.0 source; energy calculations themselves were not executed here (no GPU/torch in the review environment), so the ML code path is API-verified rather than run-verified.
