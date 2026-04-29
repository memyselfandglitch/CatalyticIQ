# CatalyticIQ

CatalyticIQ is an AI platform prototype for **Theme 4: AI Platform for Molecular Discovery in Chemical Catalysis and Synthetic Biology**.

It supports an end-to-end discovery workflow:

1. Researcher enters a target reaction.
2. Platform uses structured catalyst datasets and trained models.
3. Generative module proposes candidate catalyst designs.
4. Predictive models rank candidates.
5. Results are exported for lab testing.
6. Experimental outcomes are fed back for retraining.

## Theme 4 Alignment

This repository is designed around the hackathon requirements:

- **Chemical catalysis direction:** CO2 + green H2 -> methanol workflow implemented.
- **Generative + predictive loop:** candidate generation plus ranking.
- **Feedback loop readiness:** dataset update and retraining path included.
- **Pilot readiness:** architecture and data model are designed to extend to syngas->ethanol and ethanol->hydrocarbons in a longer GPS Renewables pilot.

## Current Scope (Hackathon MVP)

- Focus reaction family: **CO2 hydrogenation to methanol**.
- Public data sources integrated:
  - TheMeCat
  - Suvarna methanol catalysis dataset
- Main artifacts:
  - merged training data in `dataset/co2_methanol.csv`
  - fine-tuned model checkpoints under `dataset/co2_methanol/output_*`
  - generated candidate list under `generated_mol_*.csv`

## Environment Setup

### Apple Silicon (M1/M2/M3)

```bash
conda env create -f catalyticiq-osx-arm64.yml
conda activate catalyticiq
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
python -m pip install torch-geometric==2.5.2
```

If `conda activate` fails in a fresh shell:

```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda init zsh
```

## Dataset Preparation

### 1) Download raw files

Expected locations:

- `dataset/raw/TheMeCat_v1.csv`
- `dataset/raw/Suvarna_2022.xlsx`

### 2) Build merged CatalyticIQ dataset

```bash
python3 scripts/prepare_co2_methanol_dataset.py \
  --themecat "dataset/raw/TheMeCat_v1.csv" \
  --suvarna "dataset/raw/Suvarna_2022.xlsx" \
  --output "dataset/co2_methanol.csv"
```

### 3) Optional target scaling for demo-readable ranking

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("dataset/co2_methanol.csv")
df["methanol_sty"] = df["methanol_sty"] * 100.0
df.to_csv("dataset/co2_methanol_scaled.csv", index=False)
PY
```

## Fine-tuning

Use pretrained ORD checkpoint in:
`dataset/ord/output_0_ord_pretrained_aug5`

```bash
MPLCONFIGDIR="/Users/deveshi/catalyst_gen/.cache/matplotlib" \
XDG_CACHE_HOME="/Users/deveshi/catalyst_gen/.cache" \
/opt/homebrew/Caskroom/miniconda/base/envs/catalyticiq/bin/python main_finetune.py \
  --file co2_methanol \
  --pretrained_file ord \
  --pretrained_time ord_pretrained_aug5 \
  --epochs 30 \
  --lr 0.0005 \
  --class_weight enabled
```

Output goes to:
`dataset/co2_methanol/output_<seed>_<timestamp>`

## Generation

```bash
MPLCONFIGDIR="/Users/deveshi/catalyst_gen/.cache/matplotlib" \
XDG_CACHE_HOME="/Users/deveshi/catalyst_gen/.cache" \
/opt/homebrew/Caskroom/miniconda/base/envs/catalyticiq/bin/python generation.py \
  --file co2_methanol \
  --pretrained_file co2_methanol \
  --pretrained_time <timestamp> \
  --correction enabled \
  --from_around_mol enabled
```

Generated files:

- `generated_mol_*.csv`
- `generated_stats_*.txt`

## Demo Checklist

- Enter target reaction (CO2 + H2 -> methanol).
- Show ranked candidate list from generated outputs.
- Export shortlist for hypothetical lab validation.
- Show retraining path with newly appended results.

## Longer-Term Pilot Path (GPS Renewables)

- Add syngas->ethanol and ethanol->hydrocarbons datasets.
- Add constrained generation for inorganic catalyst components.
- Add lab-result ingestion API with provenance/versioning.
- Add collaboration, role-based review, and experiment attribution.
- Integrate simulation and process-economic screening modules.
