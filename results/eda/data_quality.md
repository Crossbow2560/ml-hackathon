# Data quality findings and decisions

Source: `results/analysis/<partition>/*.csv` (Phase 0 scripts) visualised in `notebooks/01_eda.ipynb`.
Partition is stated for every finding. evaluator_test was not opened.

## Organisation of the raw data
- One folder per trip with four header-less, whitespace-separated files: `.track` (lon, lat, t), `.nodes` (lon, lat),
  `.arcs` (from_node, to_node) and `.route` (ordered arc ids). Node id = row index in `.nodes`; arc id = row index in `.arcs`.
- **Each trip has its own local road network**, so node/arc ids only mean something inside that trip.
- All 80 train and 10 student_test trips have all four files; no parse or column-count errors.
- `.route` lists the arcs driven, in order, but **does not say which GPS point is on which arc**. Per-point labels must be
  derived by aligning the GPS sequence to the route (Phase 2).

## GPS tracks
- train: 208,417 points; student_test: 19,160 points. Trips range from 262 to 19,313 points.
- No NaN/inf values, no out-of-range coordinates, no lon/lat swaps, no duplicate or decreasing timestamps (both partitions).
- Sampling is ~1 Hz: 96.5 % (train) / 98.6 % (student_test) of steps are exactly 1 s. 12 train steps have non-integer Δt.
- Gaps: 22 train trips have a gap > 60 s and 11 have a gap > 300 s (largest 153,064 s ≈ 42.5 h in trip 90, which also
  inflates the total train duration). Only 1 student_test trip has a gap > 60 s.
  **Decision:** split the HMM chain at gaps > 60 s instead of forcing a path through them.
- 147 train steps imply speeds > 70 m/s (GPS jumps); 1 in student_test. **Decision:** smooth speed/heading over a window.
- Trips are spread worldwide (latitudes −34.6° to 61.2° in train). **Decision:** distances in a per-trip UTM projection
  (metres), never in degrees; no absolute coordinates as features.

## Road networks
- Median arc length 24.5 m (train) / 31.9 m (student_test): arcs are short straight pieces, so a road is many arcs.
- Two-way share (arc u→v whose reverse v→u exists) is 0.35–1.00, median 0.95 (train). Both directions of a road sit on
  the same line, so distance alone cannot separate them. **Decision:** heading vs arc bearing is a key feature.
- 22 self-loops and 1,775 duplicate arcs in train (2 / 128 in student_test); no invalid node references, no isolated nodes.
- Networks are not fully connected: 79/80 train trips have > 1 weakly connected component (median 20), although the
  largest strongly connected component holds ~96 % of nodes. **Decision:** the HMM must handle unreachable transitions.

## Reference routes
- Route continuity is 0.981–1.000; 10 train trips have 13 breaks in total, 1 student_test trip has 1 break.
- 67 repeated arcs and 46 immediate U-turns in train (255 / 8 in student_test).
- The routes are reference labels, not verified ground truth (see far-off points below).

## GPS noise relative to the route
- Distance from a GPS point to the nearest route arc: median 5.5 m (train) / 5.0 m (student_test); p95 140 m (train,
  driven by a few trips) / 16.8 m (student_test).
- Naive nearest-arc baseline: the closest arc (either direction) is on the route for 86.1 % (train) / 89.7 %
  (student_test) of points — the classifier + HMM must beat this.
- Candidate radius: at 50 m a point has ~12 candidate arcs on average and 90 % of train trips keep ≥ 98 % of their
  points within 50 m of the route. **Decision:** R = 50 m; K cap to be set in Phase 2 (≈ 45 covers the 95th percentile).
- Four train trips have > 5 % of points more than 50 m from the route:
  - **33 and 54 — off-network dwell:** after driving the route, the vehicle stays for 60 % / 87 % of the trip time in a
    ~1 km area with no mapped road (parked / private ground).
  - **2 and 43 — route mismatch / detour:** far points spread along the trip (trip 2 continues ~550 m past the route end).
  **Decision:** points > 50 m from their aligned route arc are marked `unreliable` and excluded from training labels and
  metrics; whether to drop trip 54 entirely is decided after Phase 2 labels are inspected.
