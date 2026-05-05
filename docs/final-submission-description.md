# CatalyticIQ — Round 2 Submission Description

CatalyticIQ is an end-to-end **chemical catalysis** prototype for CO2-to-methanol catalyst discovery. A reaction-conditioned generative VAE proposes novel catalyst compositions, the ActivityHead ranks them by predicted methanol productivity, a chemistry validation gate filters weak candidates, and a YAML-driven Cantera/thermodynamic layer generates simulation labels for validation and surrogate training. Lab feedback flows into a versioned retrain loop. The Streamlit dashboard surfaces every step — discovery, pathway, compare, knowledge-base, validation, feedback — for a researcher to drive the loop from a single screen.

## What we ship

- **Generative model**: reaction-conditioned CVAE fine-tuned on 1,946 CO2-to-methanol rows merged from TheMeCat and Suvarna. Latest selected checkpoint: `dataset/co2_methanol/output_0_20260503_190505/`, best validation loss 4.2088, 100% validity in the latest generation run.
- **Predictive heads**: latent-MLP `ActivityHead` (R^2 = 0.755, MAE 0.10 g MeOH / h / g_cat on a 196-row test split), `SelectivityHead` (joint MeOH / CO selectivity, R^2 = 0.461 on TheMeCat-only rows), `StabilityHead` (descriptor proxy from Tammann / Hüttig temperatures + redox class).
- **Simulation surrogate**: 3,000 YAML-sweep CO2 validation rows generated from 15 shortlisted catalysts x 200 condition samples. Random Forest surrogate fit to simulation labels: test R^2 0.9945, test MAE 0.0038 g/h/g_cat.
- **Reaction-energy module**: HCOO and RWGS pathway graphs with three pluggable backends (heuristic literature scaling, GFN2-xTB cluster, DFT / OCP IS2RE). The dashboard always labels the actual backend used.
- **External knowledge base**: Materials Project adapter (mp-api with offline cache fallback), Open Catalyst Project adapter (fairchem with offline binding-energy seed), DuckDB cache with append-only provenance.
- **Encoder validation suite**: held-out R^2 0.775 / 93.9% 90% interval coverage, latent-neighbour Jaccard 0.92, top-decile coherence 48%, active-learning recovery 8/20 in top-50, Pareto comparison vs random and GA baselines. Output: `encoder_report.pdf` + `encoder_report.json`.
- **Feedback loop**: append-only DuckDB experiments log, model-versions table with parent / delta-R2 / N / PSI, retrain orchestrator with heads-only and full-CVAE modes plus PSI > 0.25 / N < 25 drift guard.
- **Dashboard**: six-tab Streamlit app (Discover / Pathway / Compare / Knowledge Base / Validation / Feedback).

## How a researcher uses it

1. Open the dashboard, pick the latest run.
2. **Discover** tab shows the post-processed shortlist with calibrated STY, selectivity prior, stability proxy, and 2D structures. Top three for the current artifacts: `Cu/K/ZnO`, `Cu/Pd/ZrO2`, `Cu/ZnO`.
3. **Pathway** tab plots the free-energy diagram for the chosen candidate; mechanism toggle compares HCOO vs RWGS routes; backend toggle slides between heuristic / xTB / DFT tiers.
4. **Compare** tab plots activity vs selectivity, merging Materials Project known catalysts with the generative novel candidates, sized by stability proxy.
5. **Knowledge Base** tab lists nine curated MP entries (Cu, ZnO, Al2O3, ZrO2, In2O3, CeO2, Pd, Ni, Pt) with provenance plus the OCP binding-energy seed for adsorbate / surface / energy.
6. **Validation** tab shows R^2, neighbour Jaccard, top-decile coherence, AL recovery, Pareto bars; downloadable encoder report PDF.
7. **Feedback** tab logs the next lab run; retrain command updates the activity head and writes a new `model_versions` row.

## What is intentionally still in roadmap

- Tier B (xTB) and Tier C (DFT/OCP) are wired but degrade gracefully when their optional packages are absent. We label this in the UI rather than hide it.
- Stability head is descriptor-based today — the residual MLP is parameterised but unused until TOS data arrives.
- Stage B (syngas -> ethanol) and Stage C (ethanol -> hydrocarbons / jet-range products) re-use the same pipeline with new dataset keys, reaction-specific heads, and reaction-specific YAML simulation configs; they are pilot extensions, not required for the CO2 demo.
- Direction 2 (synthetic biology) is reserved for a parallel `catcvae/protein_*` track once the SME lands.
- Multi-user collaboration, lab-system integrations, and higher-fidelity Cantera/CatMAP/FairChem reactor validation are scoped for the pilot.

## Pilot intent

We are interested in piloting CatalyticIQ with GPS Renewables. The feedback loop is designed so that even 50-100 internal datapoints unlock Stage C end-to-end, with full data provenance and version history out of the box.
