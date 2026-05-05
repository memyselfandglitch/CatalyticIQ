# Round 2 Demo Script: CO2-to-Methanol Closed Loop

This walkthrough keeps the demo focused on **Direction 1: Chemical Catalysis** for
`CO2 + green H2 -> methanol`. Syngas-to-ethanol and ethanol-to-jet are mentioned
only as pilot extensions.

## 0. Pre-flight

Refresh the committed artifacts:

```bash
conda run -n catdrx python scripts/run_co2_demo.py --sweep-samples 200
```

Launch the dashboard:

```bash
conda run --no-capture-output -n catdrx streamlit run app.py \
  --server.port 8501 \
  --server.address 127.0.0.1
```

Sidebar:

- **Reaction** = `CO2 + H2 -> methanol`
- **Run folder** = `output_0_20260503_190505`
- **Top-N shortlist size** = 20
- **Require metal-containing candidates** = on

## 1. Discover: Generate And Rank

Show the closed-loop status panel:

- known catalyst baseline count from Materials Project/OCP cache
- simulation validation rows
- YAML sweep rows
- simulation-surrogate test R2

Then show the top ranked candidates. Explain the ranking simply:

```text
CVAE generates candidates
-> ActivityHead predicts methanol STY
-> validation gate filters chemical relevance
-> Cantera/thermo simulation and surrogate support final ranking
```

Say clearly that `raw_score` is the generation score, while
`predicted_sty_g_h_gcat` is the ActivityHead-aligned productivity score.

## 2. Knowledge Base: Known Catalyst Baseline

Open **Knowledge Base**.

Show that the platform is not only a generator: it first retrieves or loads a
known catalyst baseline from Materials Project/OCP cache. Live adapters activate
when API keys and optional packages are available; the committed cache keeps the
demo deterministic.

## 3. Validation: Simulation Layer

Open **Validation**.

Show:

- `simulation_validation.csv`
- backend: `yaml_cantera_plus_analytic_microkinetic`
- equilibrium conversion
- catalyst descriptor score
- simulated STY

Then show the sweep surrogate:

- 3,000 rows from `15 candidates x 200 condition samples`
- test R2 around `0.9945`
- test MAE around `0.0038`

Phrase this carefully:

> The surrogate is validated against our simulation layer, not wet-lab data yet.
> In a GPS pilot, internal experiments replace or augment these labels.

## 4. Pathway: Reaction-Energy Diagram

Open **Pathway**.

Pick a high-ranked candidate and show HCOO vs RWGS. Keep the wording honest:

- Tier A heuristic scaling is always available.
- xTB / DFT tiers are optional and labelled by the actual backend used.
- The diagram gives chemists mechanistic context before export.

## 5. Compare: Known vs Novel

Open **Compare**.

Show the plot combining:

- known catalysts from the knowledge base
- generated CatalyticIQ candidates

This satisfies the required flow:

```text
retrieve known catalysts -> generate novel candidates -> rank and compare
```

## 6. Feedback: Close The Loop

Open **Feedback**.

Show the import command:

```bash
conda run -n catdrx python scripts/import_feedback_csv.py \
  --input dataset/feedback/co2_methanol_lab_results_example.csv
```

Then show the heads-only retraining command:

```bash
conda run -n catdrx python scripts/retrain_with_feedback.py \
  --file co2_methanol \
  --pretrained_time 20260503_190505 \
  --mode heads
```

Explain:

- small feedback batches update the ranking head first
- full CVAE fine-tuning waits for enough validated rows and PSI drift checks
- every experiment and model version is append-only and auditable

## 7. Wrap

Closing line:

> CatalyticIQ implements the chemical catalysis loop requested in Theme 4:
> known catalyst baseline, novel catalyst generation, predictive ranking,
> thermodynamic/Cantera simulation validation, visual comparison, export, and
> feedback-driven retraining. The demo is CO2-to-methanol today; the pilot path
> extends the same backbone to syngas-to-ethanol and ethanol-to-jet as data
> becomes available.
