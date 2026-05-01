# Theme 4 — CatalyticIQ Solution Document

This document maps the CatalyticIQ Round 2 prototype to every requirement of Theme 4 (AI Platform for Molecular Discovery in Chemical Catalysis and Synthetic Biology). Sections are ordered to match the Round 1 brief.

## 1. Understanding of the problem

GPS Renewables is building India's first Ethanol-to-Jet plant. Their fuel-chemistry stack stitches three reactions: **CO2 + H2 -> methanol** (front-end carbon capture leverage), **syngas -> ethanol** (mid-stream), and **ethanol -> jet hydrocarbons** (terminal product). Each reaction has its own catalyst landscape with thousands of literature data points scattered across compositions, supports, promoters, and process windows. Discovery today is dominated by trial-and-error, so the bottleneck is not catalysis chemistry per se but the cost-of-iteration.

CatalyticIQ replaces brute-force search with an AI loop that proposes novel candidates, ranks them on the same axes a chemist cares about (activity / selectivity / stability / energy profile), and tightens predictions every time a lab result returns. This document describes the platform as it stands at Round 2 (CO2->methanol, Direction 1) and how each layer extends to the rest of the pilot.

## 2. SME inclusion

The CatalyticIQ team includes domain advisors covering chemical engineering and heterogeneous catalysis, with named SME involvement at three checkpoints:

- **Reaction selection and dataset curation** — confirming that TheMeCat + Suvarna are the right anchor corpora for CO2->methanol and that pseudo-SMILES preserves enough chemistry for the encoder.
- **Stability descriptor table** — sanity-checking the Tammann / Hüttig / redox-class values used in `catcvae/stability_descriptors.py` against published deactivation literature on Cu/ZnO, In2O3, ZrO2 catalysts.
- **Reaction-mechanism review** — vetting the HCOO and RWGS pathway graphs and the binding-energy table in `catcvae/reaction_energy.py`.

For Round 2 onwards we plan to formalise this as a weekly review with a senior catalysis engineer plus, for the synthetic-biology direction, a metabolic-pathway researcher.

## 3. System architecture

The platform is split across four layers; each maps to concrete code.

### 3.1 Data layer

- `scripts/prepare_co2_methanol_dataset.py` ingests TheMeCat (CSV) and Suvarna (XLSX) and emits two CSVs:
  - `dataset/co2_methanol.csv` (legacy schema, used by the existing CVAE pipeline).
  - `dataset/co2_methanol_full.csv` (extended schema with `selectivity_meoh_pct`, `co2_conversion_pct`, `yield_meoh_pct`).
- Catalyst compositions are converted to a pseudo-SMILES representation (`[Cu].[Zn].[Al]`) gated against the elemental whitelist in `catcvae/utils/elements_ord.csv`.
- External knowledge: `services/retrieval/materials_project.py` (mp-api with offline cache) and `services/retrieval/open_catalyst.py` (fairchem with offline cache), backed by `services/retrieval/cache.py` DuckDB store with append-only provenance.
- Lab-feedback storage: `services/feedback/store.py` DuckDB store with append-only `experiments` and versioned `model_versions` tables.

### 3.2 AI/ML layer

- **Generative**: reaction-conditioned VAE (the existing CatDRX architecture, fine-tuned in `dataset/co2_methanol/output_0_20260428_212044/`).
- **Predictive heads** (latent-MLP on the frozen encoder embedding):
  - `ActivityHead` — methanol STY, R^2 = 0.755 on a held-out 196-row test split, MAE 0.10 g/h/g_cat.
  - `SelectivityHead` — joint MeOH / CO selectivity, R^2 = 0.43 on the 605-row TheMeCat-only test slice.
  - `StabilityHead` — composition-weighted Tammann / Hüttig / redox-class proxy in `catcvae/stability_descriptors.py`.
- **Validation suite** (`scripts/validate_encoder.py`):
  - Held-out predictor R^2 + 90% interval coverage.
  - Latent-neighbour Jaccard similarity (mean 0.92).
  - Top-decile coherence: 48% of nearest neighbours of high-STY rows are themselves high-STY (chance is 10%).
  - Pareto comparison: CVAE's predicted-STY distribution vs random uniform sampling vs a small genetic algorithm.
  - Active-learning recovery: holding out the top 20 STY rows and asking how many return in the top 50 of the retrained head (8 / 20 = 40% in the current artifact set).
- **Feedback retrain** (`scripts/retrain_with_feedback.py`) supports two modes:
  - `heads` (default): refits only the property heads on cached embeddings. Safe at small N.
  - `cvae`: schedules a full CVAE fine-tune. Refuses to run when `PSI > 0.25` or `N < 25` unless `--force` is passed.

### 3.3 Simulation layer

- `catcvae/reaction_energy.py` exposes a single `estimate_pathway(components, mechanism, backend)` interface backed by three tiers:
  - **Tier A (`heuristic_scaling`)** — composition-weighted pure-element binding energies (eV) fed through piecewise scaling relations to a six-step HCOO or RWGS profile. Always available.
  - **Tier B (`xtb_topn`)** — GFN2-xTB single-point on a 19-atom icosahedral cluster surrogate of the dominant active metal. Activates when `xtb-python` is importable; otherwise degrades to Tier A and labels the result honestly in the UI.
  - **Tier C (`dft_topk`)** — Open Catalyst Project IS2RE composition match via `fairchem`, falling through to ASE+GPAW for unmatched compositions. Activates when `fairchem` is importable.
- COMSOL .mph reactor surrogate is in the pilot roadmap; the architecture leaves a clean integration point at `services/simulation/` (not yet implemented in code).

### 3.4 User interface

`app.py` Streamlit dashboard with six tabs:

- **Discover** — KPIs, post-processed shortlist with calibrated STY / selectivity-prior / stability-proxy / 2D depictions, training trends, generation metrics.
- **Pathway** — per-candidate free-energy diagram with HCOO / RWGS toggle and Tier A / B / C selector.
- **Compare** — activity vs selectivity scatter merging known catalysts (MP-coloured) with CatalyticIQ-novel candidates; bubble size = stability proxy.
- **Knowledge Base** — full Materials Project + OCP retrieval results with provenance log.
- **Validation** — encoder validation metrics + downloadable PDF.
- **Feedback** — experiment-log form, recent experiments table, retrain trigger guidance, model-version history.

## 4. Data feedback loops

The feedback loop is the load-bearing differentiator and is implemented end-to-end:

1. The researcher exports the shortlist from the Discover tab and synthesises / tests one or more candidates.
2. Results are logged through the Feedback tab, which writes to `cache/feedback.duckdb` (append-only) with timestamp, user, conditions, and current model version.
3. `scripts/retrain_with_feedback.py` is invoked in `heads` mode — this loads cached embeddings + the latest checkpoint, retrains the activity head with the additional rows, computes the test-set delta-R2 vs the parent head, and writes a new entry into the `model_versions` table.
4. The PSI drift guard refuses to escalate to a full CVAE fine-tune until at least 25 rows have arrived and the PSI on activity targets is below 0.25, both configurable.
5. The Validation tab is regenerated by re-running `scripts/validate_encoder.py`, so judges (or GPS Renewables' chemists) can see the validation metrics shift after each feedback round.

## 5. Justification of technology choices

- **Reaction-conditioned VAE over diffusion / pure GA**: VAE gives a continuous latent space we can interrogate (round-trip, neighbour-search, GA seeded from the latent prior) and supports conditional generation on reaction context.
- **Latent-MLP heads over re-running the joint NN**: lets us iterate on activity / selectivity / stability prediction without re-running the costly graph encoder pass; also enables cheap heads-only feedback retrains.
- **DuckDB for caches and feedback**: zero-server SQL, single-file, easy to commit alongside the code, and trivial to swap for Postgres in the pilot.
- **Streamlit for the prototype UI**: ships a working multi-user-friendly dashboard in days; the architecture (Pure-Python service modules, no Streamlit-specific business logic) makes a Next.js / FastAPI rewrite a straightforward Round 3 step.
- **Heuristic + xTB + DFT tiered energy estimation** rather than DFT-only: keeps the bulk shortlist tractable on commodity hardware while reserving DFT for the final 5-10. The tier label is shown in the UI so we never overclaim.

## 6. Risks and trade-offs

- **Pseudo-SMILES loses oxide structure.** Mitigated by the post-processing oxide map and the descriptor-based stability score. Long-term we want a graph-of-elements representation rather than dot-disconnected SMILES.
- **Stability proxy is descriptor-based, not data-fit.** Explicitly labelled as a proxy in every UI surface; learnable residual is in place in `StabilityHead` for when TOS data lands.
- **Tier B / C are gated by optional dependencies.** The fallback is honest: the UI shows the actual backend used, never the requested backend, when degradation happens.
- **Feedback retrain on small N risks overfitting.** Default policy is `heads`-only with PSI < 0.25 and N >= 25 before the CVAE itself is touched. PSI safeguard is configurable per deployment.

## 7. Roadmap

| Stage | Reaction | Dataset | Module gate |
|-------|----------|---------|-------------|
| A (now) | CO2 -> methanol | TheMeCat + Suvarna (1,946 rows) | shipped in this prototype |
| B | syngas -> ethanol | PNNL Active-Learning (Zenodo:11113829) | requires re-running phases 1-7 with new dataset key |
| C | ethanol -> jet | GPS Renewables proprietary lab data | requires the pilot data agreement |
| D | enzyme / pathway design | BRENDA + UniProt + AlphaFold | adds a parallel `catcvae/protein_*` track |

The pilot work plan is to deliver Stage B end-to-end within 2 weeks of pilot kick-off, Stage C as soon as proprietary data lands, and Stage D in parallel once the synthetic-biology SME is on board.

## 8. Pilot intent

We are actively interested in a pilot engagement with GPS Renewables. CatalyticIQ is designed so that even 50-100 datapoints from an internal lab unlock Stage C end-to-end; the feedback loop becomes the institutional knowledge system the brief asks for, with full append-only provenance and per-user audit ready to extend into a multi-user collaboration platform.
