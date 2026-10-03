# TrafficFlowBench 2026 — team Steins: solution write-up (draft)

Public leaderboard: 0.88107 (V13). Code: `trafficflow/tfb/` (Apache-2.0), single entry script: `trafficflow/reproduce.sh`.
CPU only (4 cores, 15 GB RAM). Full experiment log with every accepted and rejected idea: `trafficflow/PLAN.md`.

## Guiding principle: validation and private are new scenarios
The scenario seed of validation (March) and private (April) is redrawn: per-link free-flow plateaus move by sd ≈ 1.3 km/h,
queued-speed levels and ramp demand change, and bottleneck clusters switch on or off. Every model is therefore judged
**out of scenario** on the published layers of validation/private, used for evaluation only:
* Task 1: hide 3% of the observed cells of validation/private, rebuild the features without them, predict them
  (`tfb/t1_shift_eval.py`). It predicted the online deltas of V12→V12b (+0.006 vs +0.0058) and V12b→V13 (+0.0016 vs
  +0.0021); the train-day holdout did not (V10a→V12: +0.005 predicted, −0.0012 online).
* Task 2: onset events and ongoing windows mined from the validation/private masked layer, inputs exactly as at
  inference (`tfb/t2_onset_shift.py`, `tfb/t2_ongoing_shift.py`, `tfb/t2_prod_shift.py`).

## Task 1 (state reconstruction)
* LightGBM residual over temporal interpolation (speed; flow per lane), neighbour, profile and pct features.
* **Corridor-wide common-mode noise**: the generator's free-flow branch is flat, so speed − plateau is measurement noise;
  it shares a corridor-wide component (corr ≈ 0.3 between links 10+ apart). The mean deviation of the other links observed
  at the same slot is a feature. True free-flow error drops to the white-noise floor (1.145 vs ideal 1.152 out of scenario).
* **Scenario-own statistics**: plateaus and ramp/speed profiles are measured on the split's own published masked layer.
  Train-measured plateaus planted a per-link bias in the new scenarios (V12); split-own ones removed ≈ 0.3 km/h of speed
  RMSE out of scenario.
* **Ramp congestion sensor for the Task 2 blackouts**: on-ramp inflow falls to 5–70% of its profile while the mainline
  link it feeds is queued, and the ramp layer stays published inside the 90-minute mainline blackout after each Task 2
  origin. A gap specialist trained on synthetic blackouts uses it (gap speed RMSE 6.3 → 4.9 out of scenario).
* Capacity: the out-of-scenario learning curve kept improving up to all train target cells (≈ 20M rows; matrix filled
  column-wise into one float32 array). V13 uses the all-data model (a 0.75 · all-data + 0.25 · 1.2M-row average was
  measured afterwards at ≈ +0.0002 out of scenario, below the submission bar).

## Task 2 (queue forecasting)
* Onset: queues switch on as whole blocks at the last horizon step; a cluster model + in-cluster link model with
  Monte-Carlo expected-IoU decoding. The origin-row-free refit (the window history ends at T−5) gave +0.034 online.
* Ongoing: 0.5 · LightGBM + 0.5 · mean of CNN seeds (1-D dilated convolutions over links), threshold 0.5. **The masked
  layer publishes the origin row T at ≈ 58% coverage**; models trained and run with it gained +0.010 in scenario and
  +0.021/+0.023 out of scenario.
* Compliance: only data up to the origin T are used (the 60-minute history and same-day pre-origin masked layer).
  Observations after the post-horizon buffer are never read for Task 2.

## Task 3 (physics) and Task 4 (ODME)
* Task 3 follows Task 1 (k = q/v, N = kL); its residual is bounded by flow noise in N at the target cells.
* Task 4: L2 projection of the split's weak prior onto its (noise-free) link counts, equal to the organizers' ridge
  reference within 0.04%; KL, rescaled, per-path and pooled-prior variants were all worse online.

## What did not work (out of scenario)
Physics-based onset location (flat FD: no precursor), ramp demand for onset, prior-day and same-day scenario evidence for
onset, ramp features for the main Task 1 model, queued-speed plateaus, more CNN seeds (saturated), a spatiotemporal CNN
imputer (fits the train scenario's spatial patterns; flow error doubled out of scenario).
