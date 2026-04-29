# CatalyticIQ

CatalyticIQ is an AI research platform for **Theme 4: AI Platform for Molecular Discovery in Chemical Catalysis and Synthetic Biology**.

It is designed for GPS Renewables-style workflows where researchers must rapidly discover and optimize catalysts and biological pathways for sustainable fuel production.

## Hackathon Submission Snapshot

CatalyticIQ demonstrates an end-to-end discovery loop for **CO2 + green H2 -> methanol**:

1. Ingest and normalize scientific catalyst datasets.
2. Fine-tune a reaction-conditioned generative + predictive model.
3. Generate novel candidate catalyst designs.
4. Rank candidates by predicted performance signals.
5. Export candidate lists for lab testing.
6. Feed outcomes back for retraining in subsequent rounds.

## What This Repository Covers

### Implemented in this prototype

- CO2->methanol data pipeline using:
  - `dataset/raw/TheMeCat_v1.csv`
  - `dataset/raw/Suvarna_2022.xlsx`
- Merged training dataset:
  - `dataset/co2_methanol.csv`
  - `dataset/co2_methanol_scaled.csv`
- Fine-tuning and training outputs:
  - `dataset/co2_methanol/output_*`
- Candidate generation and export:
  - `generated_mol_*.csv`
  - `generated_stats_*.txt`

### Prepared as roadmap for pilot extension

- Syngas->ethanol and ethanol->hydrocarbons reaction families.
- Synthetic biology direction (enzyme/pathway design).
- Multi-user collaboration and lab system integration.

## Quick Start

### 1) Environment (Apple Silicon)

```bash
conda env create -f catalyticiq-osx-arm64.yml
conda activate catalyticiq
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
python -m pip install torch-geometric==2.5.2
```

### 2) Build the merged CO2 methanol dataset

```bash
python3 scripts/prepare_co2_methanol_dataset.py \
  --themecat "dataset/raw/TheMeCat_v1.csv" \
  --suvarna "dataset/raw/Suvarna_2022.xlsx" \
  --output "dataset/co2_methanol.csv"
```

### 3) Scale target for demo-friendly ranking (recommended)

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("dataset/co2_methanol.csv")
df["methanol_sty"] = df["methanol_sty"] * 100.0
df.to_csv("dataset/co2_methanol_scaled.csv", index=False)
PY
```

### 4) Fine-tune

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

### 5) Generate candidates

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

## Recommended Training Timelines

### Fast demo iteration (1-3 hours)

- 1 to 5 epochs to validate pipeline and produce sample outputs.

### Strong hackathon run (6-12 hours)

- 20 to 30 epochs with periodic monitoring of `report.txt` and `loss.txt`.

### Extended quality run (overnight, 12+ hours)

- 40+ epochs plus candidate post-filtering and reranking for stronger shortlist quality.
