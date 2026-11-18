# Dataset

The dataset is **not committed** to this repository. Download it from Google Drive:

**https://drive.google.com/drive/folders/1t4igpcX2wB98Y2FrhEKnJ0Ml7TAQzsmd?usp=sharing**

Place it in this folder so the layout is:

```
dataset/
├── train/            80 trips   – training, cross-validation, tuning
│   └── 00000001/
│       ├── 00000001.track   lon lat t            (whitespace, no header, ~1 Hz)
│       ├── 00000001.nodes   lon lat              (row index = node id)
│       ├── 00000001.arcs    from_node to_node    (row index = arc id, directed)
│       └── 00000001.route   arc_id               (ordered reference route)
├── student_test/     10 trips   – development-stage evaluation only
└── evaluator_test/   10 trips   – supplied at evaluation time; read ONLY by the API
```

Alternatively, keep the data anywhere and point the code at it:

```bash
export PATHSNAP_DATA_DIR=/path/to/dataset
```

Notes:
- Each trip has its **own** local road network; node and arc ids are only meaningful inside that trip.
- No code other than `api/` may read `evaluator_test/` (leakage rule).
