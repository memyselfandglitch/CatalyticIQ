# CatalyticIQ Audit Notes

## Fixed

1. Public dashboard scope
   - CO2 is now the only public demo reaction by default.
   - Pilot reactions can be shown with `CATALYTICIQ_SHOW_PILOTS=1`.

2. Stale metrics
   - SelectivityHead R2 updated to `0.461`.
   - Encoder validation held-out R2 updated to `0.775`.
   - 90% interval coverage updated to `93.9%`.

3. Environment-name drift
   - Public docs now use the shipped conda env name `catalyticiq`.
   - Existing local `catdrx` workflows can use `CATALYTICIQ_CONDA_ENV=catdrx`.

4. Broken docs link
   - Added `docs/build-and-positioning-playbook.md`.

5. Ethanol pilot wording
   - Updated from `ethanol -> jet` to `ethanol -> hydrocarbons / jet-range
     products` where it describes current repo config scope.

6. Feedback loop clarity
   - Feedback storage now includes predicted and measured outcomes.
   - Dashboard flags prediction-vs-actual discrepancies and surfaces hypotheses.

## Not Changed

- CO2 simulation/surrogate wording remains simulation-label based, not wet-lab
  accuracy.
- Syngas and ethanol configs remain available for pilot work.
- Optional dependency warnings from DeepChem/TensorFlow/JAX/DGL are not blockers
  for the current CO2 demo path.

