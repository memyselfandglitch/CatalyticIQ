# Build And Positioning Playbook

## Public Demo Positioning

CatalyticIQ should be presented as a **CO2-to-methanol chemical catalysis MVP**,
not as the full Theme 4 platform. The strongest demo claim is:

```text
known catalyst baseline
-> generative catalyst proposal
-> ActivityHead ranking
-> validation gate
-> Cantera/thermodynamic simulation
-> sweep-trained surrogate
-> feedback discrepancy analysis
-> heads-only retraining
```

Syngas-to-ethanol and ethanol-to-hydrocarbons / jet-range products should be
described as pilot extensions that reuse the same architecture after
reaction-specific data, validation configs, and SME review are promoted.

## What To Say

- The shipped demo is Direction 1: Chemical Catalysis.
- The implemented reaction is `CO2 + green H2 -> methanol`.
- The simulation surrogate is validated against simulation labels, not wet-lab
  experiments.
- Lab outcomes can be logged and compared against predicted STY, selectivity,
  yield, and stability.
- Small feedback batches retrain the ActivityHead first; full CVAE fine-tuning
  waits for enough validated rows and drift checks.

## What Not To Overclaim

- Do not claim industrial catalyst validation yet.
- Do not claim syngas-to-ethanol or ethanol-to-hydrocarbons / jet-range products
  are demo-ready.
- Do not claim optional xTB / DFT / OCP live calls are always active.
- Do not present the simulation surrogate R2 as wet-lab accuracy.

## Video Flow

1. Launch the dashboard.
2. Show Knowledge Base known catalyst baseline.
3. Show Discover generated candidates and ActivityHead ranking.
4. Show Validation simulation backend and surrogate metrics.
5. Show Pathway HCOO vs RWGS diagram.
6. Show Compare known vs generated candidates.
7. Export the shortlist.
8. Log or import feedback.
9. Show discrepancy flags and heads-only retraining command.

## Deployment Shape

For the hackathon, deploy a deterministic Streamlit artifact viewer. It should
load committed or precomputed artifacts and avoid live training during the demo.

For a GPS pilot, split the system into:

- Streamlit or web UI
- FastAPI backend
- background simulation/retraining workers
- model registry
- versioned data/artifact store
- lab/LIMS integration adapter
