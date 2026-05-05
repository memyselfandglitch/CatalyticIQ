# PR: Align Public Demo Scope With Shipped Artifacts And Refresh Metrics

## Summary

- Hide pilot reactions from the public dashboard by default.
- Keep pilot reactions registered for extension work behind `CATALYTICIQ_SHOW_PILOTS=1`.
- Refresh stale README/docs metrics from committed JSON artifacts.
- Standardize docs and UI commands on the shipped `catalyticiq` conda env.
- Keep legacy local `catdrx` usable through `CATALYTICIQ_CONDA_ENV=catdrx`.
- Add feedback prediction-vs-actual fields and discrepancy analysis.
- Add the missing build-and-positioning playbook linked from the README.

## Why

The repository now ships a coherent CO2-to-methanol closed-loop chemical
catalysis prototype. Some public-facing docs and UI affordances still implied
broader pilot readiness. This patch makes the demo honest, self-consistent, and
stronger for hackathon review without removing extension code.

## Key Changes

1. `services/reaction_registry.py`
   - adds `has_output_runs()`
   - marks CO2 as the only public demo profile
   - hides pilot profiles unless `CATALYTICIQ_SHOW_PILOTS=1`

2. `app.py`
   - uses `CATALYTICIQ_CONDA_ENV` for command snippets
   - defaults command snippets to `catalyticiq`
   - shows prediction-vs-actual feedback discrepancy analysis

3. Feedback loop
   - stores predicted and measured STY/selectivity/yield/stability fields
   - imports those fields from example feedback CSV
   - flags activity/selectivity/yield discrepancies

4. Docs
   - updates stale selectivity and encoder-validation metrics
   - aligns ethanol pilot wording to hydrocarbons / jet-range products
   - adds `docs/build-and-positioning-playbook.md`

## Validation

- `py_compile` passes for changed Python files.
- Example feedback CSV imports into a fresh DuckDB.
- Metrics cross-checked against:
  - `dataset/co2_methanol/property_heads/metrics.json`
  - `dataset/co2_methanol/validation/encoder_report.json`

## Suggested Review Focus

- Whether public UI should remain CO2-only by default.
- Whether to fully rename local usage from `catdrx` to `catalyticiq`, or keep
  the environment override for developer convenience.

