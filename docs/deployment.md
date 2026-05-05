# Deployment

The public hackathon deployment should run as an artifact viewer: it loads the
committed CO2-to-methanol demo artifacts and does not train models live.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app from the repo.
3. Set:
   - main file: `app.py`
   - Python: `runtime.txt`
4. Optional secrets:
   - `MP_API_KEY` for live Materials Project lookup
5. Deploy.

The app defaults to the public CO2 demo profile only. To expose pilot reaction
profiles in a private deployment, set:

```text
CATALYTICIQ_SHOW_PILOTS=1
```

If command snippets in the sidebar should use a non-default local conda env, set:

```text
CATALYTICIQ_CONDA_ENV=catdrx
```

## Local Deployment Smoke Test

```bash
conda run --no-capture-output -n catdrx streamlit run app.py \
  --server.port 8502 \
  --server.address 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8502
```

## What Is Included

- Streamlit dashboard
- CO2 generated shortlist artifacts
- simulation validation CSV
- 3,000-row simulation sweep
- trained simulation surrogate artifact
- property-head and encoder-validation metrics
- feedback example CSV

## What Is Not Run Live

- CVAE training
- candidate generation
- ActivityHead training
- full Cantera sweeps
- full feedback retraining

Those commands remain available for local or pilot use.
