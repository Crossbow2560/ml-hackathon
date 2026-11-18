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
