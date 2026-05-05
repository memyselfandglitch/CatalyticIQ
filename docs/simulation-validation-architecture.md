# Simulation Validation Architecture

CatalyticIQ uses one simulation-validation engine with reaction-family YAML files.
The goal is to avoid hardcoded reaction logic while keeping the hackathon demo
deployable.

## Current Stage

```text
generated shortlist
  -> chemical validation gate
  -> YAML reaction config
  -> condition sweep from YAML ranges
  -> analytic thermodynamic equilibrium
  -> Cantera equilibrium cross-check
  -> catalyst descriptor score
  -> simulation_validation.csv / simulation sweep CSV
  -> simulation surrogate
  -> dashboard
```

The active CO2-to-methanol config is:

```text
config/reactions/co2_methanol.yaml
```

The validator is:

```text
services/simulation/cantera_validator.py
```

Run it with:

```bash
conda run -n catdrx python scripts/validate_shortlist_simulation.py \
  --candidates dataset/co2_methanol/output_0_<timestamp>/generated_candidates_clean.csv \
  --reaction-config config/reactions/co2_methanol.yaml
```

When Cantera is installed and the mechanism contains the required species, the
backend column becomes:

```text
yaml_cantera_plus_analytic_microkinetic
```

If Cantera is unavailable or the selected mechanism lacks required species, the
same engine falls back to:

```text
yaml_analytic_thermo_microkinetic
```

## Why YAML

Each reaction family differs in stoichiometry, feed, target species,
byproducts, catalyst families, and temperature-pressure windows. These should be
configuration, not separate code paths.

Each YAML file defines:

- reaction id and family
- Cantera mechanism YAML
- default temperature and pressure
- sweep ranges for temperature, pressure, feed ratios, GHSV, and time-on-stream
- feed composition
- stoichiometry
- thermodynamic delta H / delta S
- target and byproduct species
- catalyst descriptor weights and synergy rules

Existing configs:

```text
config/reactions/co2_methanol.yaml
config/reactions/syngas_ethanol.yaml
config/reactions/ethanol_to_hydrocarbons.yaml
```

## Model Design

The desired platform design is:

```text
shared reaction-conditioned generative backbone
  + reaction-specific dataset
  + reaction-specific ranking head
  + reaction-specific YAML simulation config
```

This gives us one platform and one validation engine, while letting each
reaction family keep its own chemistry.

## Next Stages

Stage A: Current MVP

- YAML-driven thermodynamic validation.
- Cantera equilibrium check.
- Catalyst descriptor score.
- Dashboard merge via `simulation_validation.csv`.
- YAML-defined condition sweeps via `scripts/generate_cantera_sweep.py`.
- Lightweight surrogate training via `scripts/train_simulation_surrogate.py`.

Run a smoke sweep:

```bash
conda run -n catdrx python scripts/generate_cantera_sweep.py \
  --reaction-config config/reactions/co2_methanol.yaml \
  --candidates dataset/co2_methanol/output_0_20260503_190505/generated_candidates_clean.csv \
  --output dataset/simulation/co2_methanol_sweep_smoke.csv \
  --n-samples 3 \
  --limit-candidates 3
```

Train a surrogate from that sweep:

```bash
conda run -n catdrx python scripts/train_simulation_surrogate.py \
  --input dataset/simulation/co2_methanol_sweep_smoke.csv \
  --target simulated_sty_g_h_gcat \
  --output-dir dataset/simulation/surrogates/co2_methanol_smoke
```

Important design point: the sweep does not generate YAML. YAML defines the
reaction family and sweep bounds; the sweep generates CSV training data.

Stage B: CRECK mechanism support

- Use `creck_to_cantera_selected.py` to convert selected CRECK CHEMKIN files.
- Point `cantera_mechanism` in each YAML to the converted mechanism.
- Use broader mechanisms for syngas-to-ethanol and ethanol-to-hydrocarbons.

Stage C: Reduced reactor simulation

- Adapt `cantera_catdrx_training_data.py`.
- Build reduced catalytic reaction networks from YAML.
- Run a reactor-chain/PFR simulation over condition sweeps.
- Generate simulation-backed training data for a fast surrogate.

Stage D: Surrogate-assisted validation

- Train a lightweight regressor on Cantera sweep outputs.
- Use Cantera for top candidates and the surrogate for fast shortlist scoring.
- Feed simulation-backed outcomes into the ActivityHead retraining loop.
