# PathSnap — Map-Matching & Route Reconstruction

PES University UE24CS352A Machine Learning hackathon project.

PathSnap matches noisy GPS observations to a road network: a candidate classifier scores
nearby road segments for every GPS point, and an HMM/Viterbi decoder reconstructs a
connected route. A FastAPI server (Laptop 1) runs the pipeline; a Streamlit UI (Laptop 2)
lets evaluators pick a trip and view the predicted route and scores.

> Work in progress — sections below are filled in phase by phase.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Download the dataset as described in [`dataset/README.md`](dataset/README.md).

## Phase 0 — dataset analysis

```bash
python -m scripts.analysis.run_all --partition train
python -m scripts.analysis.run_all --partition student_test
```

Outputs go to `results/analysis/<partition>/` (CSVs + `DATASET_REPORT.md`).

Visualise any trip on an interactive folium map (road network, reference route, GPS points,
far-off points, time gaps; layer control + legend), optionally with a static PNG:

```bash
python -m scripts.analysis.06_visualise_trip --partition train --trip 1 33 54 --static
python -m scripts.analysis.06_visualise_trip --partition student_test --all
```

Maps are written to `results/analysis/<partition>/maps/<trip_id>.html` (and `.png`).

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | core library (I/O, geometry, candidates, features, models, HMM, pipeline) |
| `scripts/analysis/` | Phase 0 dataset description and validation scripts |
| `scripts/` | feature building, training, tuning, evaluation, export |
| `notebooks/` | exploratory data analysis |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Streamlit evaluator interface (Laptop 2) |
| `configs/` | configuration |
| `artifacts/` | exported final model + config used by the API |
| `results/` | CSVs, logs, plots, folium maps |
| `report/` | technical report sources and figures |

## Two-laptop deployment (evaluation)

| Laptop | Runs | Needs |
|---|---|---|
| **Laptop 1 — API** | `api/main.py` (FastAPI): full pipeline + server-side scoring | repo, `requirements.txt`, `artifacts/`, dataset incl. `evaluator_test/` |
| **Laptop 2 — UI** | `ui/app.py` (Streamlit): trip picker, map, scores | `ui/` only, `ui/requirements.txt` |

Both laptops must be on the same network (Wi-Fi or a phone hotspot).

**Laptop 1 (API)**

```bash
source .venv/bin/activate
export PATHSNAP_DATA_DIR=/path/to/dataset          # contains evaluator_test/
export PATHSNAP_EVAL_PARTITION=evaluator_test      # use student_test to rehearse
python -m api.main                                 # listens on 0.0.0.0:8000
```

Find its LAN address (`ip -4 addr` on Linux, `ipconfig` on Windows, `ipconfig getifaddr en0` on macOS) and allow
incoming TCP 8000 in the firewall (Linux `sudo ufw allow 8000/tcp`; Windows: allow Python on private networks).
Every request is printed in this terminal (`<time> <client ip> POST /match trip=<id> status=200 <ms> ms` plus a
metrics line) — show this window to the evaluator.

**Laptop 2 (UI)**

```bash
pip install -r ui/requirements.txt
export PATHSNAP_API_URL=http://<laptop-1-ip>:8000  # or type it in the sidebar
streamlit run ui/app.py
```

In the browser: *Check connection* → pick one of the 10 trips → *Run map-matching*. The map shows the raw GPS track,
the predicted route and matched points coloured by confidence; the cards show the scores computed on Laptop 1.
The reference routes never leave Laptop 1.

Test without model files: `PATHSNAP_USE_MOCK=1 python -m api.main` (nearest-arc stand-in).
