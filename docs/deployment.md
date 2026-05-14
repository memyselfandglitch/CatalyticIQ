# Deployment

The public hackathon deployment should run as an artifact viewer: it loads the
committed CO2-to-methanol demo artifacts and does not train models live.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app from the repo.
3. Set:
   - main file: `app.py`
   - Python: `runtime.txt`
4. Optional secrets (Streamlit **Secrets** UI or repo `.streamlit/secrets.toml` - **do not commit**):
   - `MP_API_KEY` - same key as on [Materials Project](https://next-gen.materialsproject.org/) (API dashboard).
     The app copies this into `os.environ` at startup so `mp-api` can read it.
5. Deploy.

Example `.streamlit/secrets.toml` (local or Cloud paste equivalent):

```toml
MP_API_KEY = "your-mp-api-key"
```

## Environment variables (all platforms)

| Variable | Required | Purpose |
|----------|----------|---------|
| `MP_API_KEY` | No (demo works offline) | Live Materials Project queries (`services/retrieval/materials_project.py`). |
| `CATALYTICIQ_SHOW_PILOTS` | No | Set to `1` to show pilot reaction profiles in the UI. |
| `CATALYTICIQ_CONDA_ENV` | No | Conda env name embedded in sidebar command snippets (default `catalyticiq`). |
| `CATALYTICIQ_GENERATION_N_SAMPLES` | No | Overrides default generation sample count in the dashboard. |

**Docker** - inject at runtime (never bake keys into the image):

```yaml
environment:
  - MP_API_KEY=${MP_API_KEY}
```

Or `--env-file .env` with `.env` gitignored.

**Kubernetes** - mount from a Secret:

```yaml
env:
  - name: MP_API_KEY
    valueFrom:
      secretKeyRef:
        name: catalyticiq-secrets
        key: mp-api-key
```

**Fly.io / Railway / Render** - set `MP_API_KEY` in the service's **Environment** / **Variables** UI (same as shell: one string, no quotes needed in UIs that strip them).

**CLI / systemd** - `export MP_API_KEY=...` in the unit or wrapper script that starts `streamlit run app.py`.

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
