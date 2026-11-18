"""PathSnap API (Laptop 1) — FastAPI server running the full map-matching pipeline.

Endpoints
    GET  /health            status + model name/version
    GET  /trips             the trips of the evaluation partition (id + number of GPS points)
    POST /match {trip_id}   prediction (pathsnap.pipeline.match_trip) + server-side metrics
                            (pathsnap.pipeline.score_trip). The reference route and labels are
                            never part of the response.

Every request is logged to stdout as
    <timestamp> <client_ip> <METHOD> <path> trip=<id> status=<code> <ms> ms
so the evaluator can watch inference requests arrive on this laptop.

Environment
    PATHSNAP_DATA_DIR        dataset root (default ./dataset)
    PATHSNAP_EVAL_PARTITION  partition served (default evaluator_test; student_test to rehearse)
    PATHSNAP_USE_MOCK=1      use api/mock_pipeline.py (nearest-arc stand-in, no model files)
    PATHSNAP_HOST / PATHSNAP_PORT   bind address (default 0.0.0.0:8000)
Run: python -m api.main
"""
from __future__ import annotations

import datetime as dt
import math
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import pipeline
from pathsnap.io import data_dir

PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "evaluator_test")
USE_MOCK = os.environ.get("PATHSNAP_USE_MOCK", "0") == "1"
if USE_MOCK:
    from api import mock_pipeline as engine
else:
    engine = pipeline

app = FastAPI(title="PathSnap API", version=pipeline.PIPELINE_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MatchRequest(BaseModel):
    trip_id: str


def partition_dir():
    return data_dir() / PARTITION


def log(msg: str):
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def clean(x):
    """JSON cannot carry NaN/inf: replace them with null."""
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ms = (time.perf_counter() - t0) * 1000
        trip = getattr(request.state, "trip_id", "-")
        client = request.client.host if request.client else "-"
        log(f"{client} {request.method} {request.url.path} trip={trip} status={status} {ms:.0f} ms")


@app.on_event("startup")
def startup():
    log(f"PathSnap API starting — partition={PARTITION} data={partition_dir()} mock={USE_MOCK}")
    if not USE_MOCK:
        _, _, cfg = pipeline.load_artifacts()
        log(f"loaded model: {cfg['classifier']['name']} + HMM ({cfg['hmm']['transition']}) v{cfg['version']}")


@app.get("/health")
def health():
    if USE_MOCK:
        model = {"name": "mock nearest-arc", "version": "mock"}
    else:
        _, _, cfg = pipeline.load_artifacts()
        model = {"name": cfg["classifier"]["name"], "hmm": cfg["hmm"]["transition"], "version": str(cfg["version"])}
    return {"status": "ok", "partition": PARTITION, "model": model}


@app.get("/trips")
def trips():
    d = partition_dir()
    if not d.is_dir():
        raise HTTPException(500, f"partition folder not found on the server: {d}")
    out = []
    for p in sorted(x for x in d.iterdir() if x.is_dir()):
        track = p / f"{p.name}.track"
        n = sum(1 for line in open(track) if line.strip()) if track.exists() else 0
        out.append({"trip_id": p.name, "n_points": n})
    return {"partition": PARTITION, "trips": out}


@app.post("/match")
def match(req: MatchRequest, request: Request):
    tid = req.trip_id.strip().zfill(8)
    request.state.trip_id = tid
    tdir = partition_dir() / tid
    if not tid.isdigit() or not tdir.is_dir():
        raise HTTPException(404, f"unknown trip id {req.trip_id!r} in partition {PARTITION}")
    pred = engine.match_trip(tdir)
    pipeline.validate_prediction(pred)
    metrics = engine.score_trip(tdir, pred)
    pipeline.validate_metrics(metrics)
    log(f"  trip {tid}: {metrics['n_points']} points | top1={metrics['top1']:.3f} macro_f1={metrics['macro_f1']:.3f} "
        f"continuity={metrics['route_continuity']:.3f} route_f1={metrics['route_f1']:.3f} | "
        f"total {pred['timing_ms']['total']:.0f} ms")
    return clean({**pred, "metrics": metrics})


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("PATHSNAP_HOST", "0.0.0.0"), port=int(os.environ.get("PATHSNAP_PORT", 8000)),
                access_log=False)
