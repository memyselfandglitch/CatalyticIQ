# Round 2 Demo Script (7 minutes)

This walkthrough drives the CatalyticIQ Streamlit dashboard end-to-end. Copy / paste the commands as given; the run timestamps used here are the ones committed in this prototype.

## 0. Pre-flight (offline, ~30 s)

```bash
conda activate catalyticiq
streamlit run app.py
```

Open the dashboard URL printed by Streamlit. Sidebar:
- **Run folder** = `output_0_20260428_212044` (the 30-epoch fine-tune)
- **Generated candidates file** = the most-recent `generated_mol_*.csv`
- **Top-N shortlist size** = 20
- **Require metal-containing candidates** = on

## 1. Discover (~90 s) — credible shortlist

Point at the **Discover** tab. Highlight:

- Top three candidates: `Cu/K/ZnO`, `Cu/Pd/ZrO2`, `Cu/ZnO`. These are real CO2-to-methanol catalyst families straight out of the literature; the generative model has rediscovered them on its own.
- Calibrated STY column (g MeOH / h / g cat) is rank-mapped against the training distribution — say "we don't claim absolute units, but the rank order is what drives synthesis decisions."
- Selectivity prior and stability proxy are labelled as priors / proxies, not measurements.
- 2D depictions render via RDKit for the top eight rows.

Click the **Activity R^2** card (top right): held-out 0.755 from `dataset/co2_methanol/property_heads/metrics.json`.

## 2. Pathway (~75 s) — chemistry credibility

Switch to **Pathway**. Default: HCOO mechanism, heuristic_scaling backend.

- Walk through the six steps — `CO2(g) + 3H2(g) -> *HCOO -> *H2COO -> *H2COOH -> *H3CO -> CH3OH(g) + H2O(g)`.
- For Cu/Zn-based candidates the rate-limiting step (~ -0.78 eV barrier on RWGS, mid-pathway dip on HCOO) matches Behrens et al. 2012.
- Toggle to RWGS mechanism — note the deeper *CO+*O step.
- Toggle backend to `xtb_topn` — the UI flags that xtb-python is not installed and the diagram stays at Tier A. Make the point: "we never lie about the backend."

## 3. Compare (~60 s) — known + novel on one chart

Switch to **Compare**. Show:

- Pareto plot of activity vs selectivity prior, sized by stability proxy.
- Materials Project entries (`mp-30 Cu`, `mp-2133 ZnO`, `mp-22598 In2O3`, etc.) appear in a different colour than the CatalyticIQ-novel candidates — the generative ones live next to their inspiration.

## 4. Knowledge Base (~45 s) — retrieval is real

Switch to **Knowledge Base**. Highlight:

- Nine curated MP entries with formation energy, band gap, and role hints.
- Probe Cu/Zn — four matching OCP binding energies (`*CO`, `*H`, `*OH`, `*HCOO`) come back from the cache with full citations.
- Mention the cache is a single-file DuckDB — `cache/retrieval.duckdb`. mp-api / fairchem live calls happen automatically when keys / packages are present, otherwise we fall through to this cache.

## 5. Validation (~75 s) — the encoder is doing real work

Switch to **Validation**. Numbers to read out loud:

- Held-out R^2 = 0.755, 90% interval coverage = 93% — the activity head is well calibrated.
- Latent-neighbour Jaccard = 0.92 — the encoder strongly clusters by elemental composition.
- Top-decile coherence = 48% — top STY catalysts have nearly half their nearest latent neighbours also in the top decile (chance is 10%).
- Active-learning recovery: 8 / 20 of held-out top STY rows return in the top-50 of a freshly trained head.
- Pareto bars: CVAE p95 beats random p95; GA wins on absolute extrapolation but is fed by the same activity head — point being the encoder *is* the prior that makes GA work.

Download the PDF report so judges can take it offline.

## 6. Feedback (~75 s) — the loop closes

Switch to **Feedback**.

- Pick `Cu/K/ZnO` from the dropdown.
- Enter measured STY = `0.61`, MeOH selectivity = `82`, stability = `120`, T = `240`, P = `50`, H2/CO2 = `3.0`. User = `demo`. Notes = `Round-2 live demo`. Submit.
- Show the row appearing in the **Recent experiments** table.
- Drop into a terminal:

```bash
python scripts/retrain_with_feedback.py --mode heads --epochs 60 --promote
```

Read the JSON output: new `version` id, `psi` 0.0 (not enough rows yet), `parent_test_r2` vs `new_test_r2`. Note the `model_versions` table now has the new entry with parent pointer and delta-R2.

## 7. Wrap (~15 s)

> "CatalyticIQ already runs the loop the brief asks for: retrieve known catalysts, generate novel ones, rank them by activity / selectivity / stability, ground them in real reaction-energy diagrams, validate the encoder is actually doing work, and close the loop with versioned retrains. The architecture is ready for syngas->ethanol, ethanol->jet, and Direction 2 as soon as the data shows up."

## Backup commands

If something goes sideways, these regenerate every artifact the dashboard reads:

```bash
python scripts/prepare_co2_methanol_dataset.py \
  --themecat dataset/raw/TheMeCat_v1.csv \
  --suvarna dataset/raw/Suvarna_2022.xlsx \
  --output dataset/co2_methanol.csv

python scripts/postprocess_candidates.py \
  --candidates dataset/co2_methanol/output_0_20260428_212044/generated_mol_lat_con_20260429_122817.csv

python scripts/train_property_heads.py --pretrained_time 20260428_212044 --epochs 100
python scripts/validate_encoder.py
```
