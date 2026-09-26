# TrafficFlowBench (IEEE Big Data Cup 2026): plan for the next session

* Competition: `kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench`
* Deadline: **2026-11-06/07** (5 submissions per day)
* Prizes: Gold $1,500 / Silver $1,000 / Bronze $500 / Student-Newcomer $500, per division.
  The **Newcomer** award may be the most reachable prize if the account holder qualifies.
  Check the rules page for the exact definition.

## Prerequisites (account holder)
1. Join the competition on Kaggle and accept its rules.
2. The environment allows `kaggle.com`, `www.kaggle.com` and `storage.googleapis.com`. Confirmed
   reachable on 2026-09-26.
3. Environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`, or `KAGGLE_API_TOKEN`. These were
   not yet visible in the session on 2026-09-26, so they need a new session.

## Scoring (from the official toolkit `github.com/jacky850/trafficflowbench-public`, docs/SCORING_SPEC.md)
`S = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME`
* **T1 state:** RMSE of speed (scale 25 km/h) and per-lane flow (scale 600) on masked cells,
  regimes R1/R2/R3. It can be scored locally on the train months.
* **T2 queue:** IoU of the binary queue indicator (speed ≤ 0.6·v_free) over T+5…T+30.
  There are two equally weighted conditions, onset and ongoing. It **cannot** be scored locally
  (labels withheld), so derive labels from the train state.
* **T3 physics:** computed from the T1 answer: FD validity plus LWR conservation. Improve it via T1.
* **T4 ODME:** path flows fitted to link counts. The released counts are noise-free.

## State of the art (public, as of 2026-09-25)
* A public entry scores **0.86711** (rank 9 of 140) with a documented pipeline. **Its repository
  has no license, so do not copy its code.** Its write-up describes these ideas:
  * T1: LightGBM residuals on top of an interpolation baseline, fundamental-diagram features,
    separate models for long blackouts, density reconciliation in congestion.
  * T3: total-variation smoothing of density inside runs of target cells.
  * T2: LightGBM onset model on labels derived from train, 2-stage stacking, top-m
    expected-IoU decoding; ongoing = LWR shockwave blend.
  * T4: L2 projection of the weak path prior onto the link counts, solved per split.
* Another public entry reports 0.845.

## Plan (own implementation, Apache-2.0)
1. Download the data. Write a loader and the exact local T1 scorer, and use the official
   scoring code where its license allows.
2. T4 first: a cheap, near-deterministic NNLS / L2 projection per split.
3. T1: interpolation baseline → LightGBM residual model; validate on held-out train days per regime.
4. T2: derive queue labels from the train state. Build onset and ongoing classifiers with
   features only ≤ T, and decode to maximise expected IoU.
5. Submit a baseline early (day 1), then iterate within the daily limit. Fit on train only and
   treat the public LB as a transfer check, since validation and private are different months.

## Progress (2026-09-26)

Code: `trafficflow/tfb/` (run from `trafficflow/`, data in `data_tfb/kaggle_public`, official toolkit
cloned to `tfb_ref/`). Kaggle CLI auth: `export KAGGLE_API_TOKEN=$KAGGLE_KEY` (the key is a new-style token).

| Sub | T1 | T2 | T4 | Public LB |
|---|---|---|---|---|
| v1 | temporal linear interp | persistence | L2 projection | 0.70508 |
| v1-kl | same | same | KL (max-entropy) projection | 0.67482 |
| v2 | interp | onset: static link set at T+30; ongoing: persistence | L2 | 0.81819 |
| v3 | LightGBM residual | onset + ongoing LightGBM, expected-IoU decoding | L2 | 0.85566 |
| v4 | v3 + gap-specialist model on T2 blackout slots | same | L2 | 0.86257 |

Findings
* T1: speed noise is ~1.7 km/h and flow noise ~30 vph/lane, so interp already scores 0.909 locally.
  The LightGBM residual reaches 0.943 locally (`python -m tfb.t1_model`, then `python -m tfb.t1_train`).
* Validation/private have 90-min all-link blackouts after each T2 origin (1.7% of targets); train has
  none. The main model is off-distribution there (flow RMSE 384/lane). A specialist trained on
  synthetic gaps placed at onset/ongoing origins (`tfb/t1_gaps.py`, `tfb/patch_gaps.py`) fixes it.
* T2 onset: in every train onset window the queue first appears exactly at T+30, on a few recurring
  bottleneck links per corridor. So onset reduces to choosing links at the last step. Mining every
  onset event in train (about one per day per corridor) gives held-out onset IoU 0.56 (static set)
  and 0.67 (conditional model).
* T2 ongoing: a per-cell model over the six steps × links near the queue gives held-out IoU 0.81
  against 0.67 for persistence, on windows with persistence IoU ≤ 0.9, as the selector uses.
* T3: organizer fluxes reproduce the published observations, so S_LWR ≈ 1 − Σ|ΔN error|/Σ|ΔN_true|.
  Only the L1 error of density q/v at target cells matters, and white measurement noise floors it.
* T4: L2 beats KL by 0.15 S_ODME. The official ridge baseline is effectively the L2 projection.
  Priors across splits correlate ~0.75, which looks like lognormal noise around a shared base.

Next
1. T4 probes, one per day alongside other changes: `sl2` (prior rescaled, then L2), `wl2`
   (chi-square), and L2 on the geometric mean of the three splits' priors.
2. T1: retrain on all train days with more rows (`python -m tfb.t1_full`, running as of this note),
   then apply the gap patch.
   Tried and dropped: a direct L1 density model (flow = k·v, or a 50/50 mix). Its local S_LWR proxy
   (`tfb/t3_proxy.py`) moves by at most ±0.006, and 0.35·S_state + 0.1·S_LWR is unchanged within
   ±0.0015. S_LWR follows T1 accuracy: interp ≈ 0.52 and LightGBM ≈ 0.60, averaged over regimes.
   Probe files are ready in `out/tfb/`: `odme_sl2.csv`, `odme_wl2.csv`, `odme_geo_l2.csv`.
3. T2: tune the decoding, and add ramp-flow and upstream-demand features to the onset model.

## Local ↔ online calibration (every change is checked against a leaderboard delta)

Local evaluators
* T1: `tfb/score.py` on held-out train days (d % 4 == 0), same formula as `score_task1.py`.
* T2: `tfb/t2_eval.py` scores the windows the organizers actually selected in train (5 onset +
  5 ongoing per corridor). Truth is the observed speed ≤ 0.6·v_free, only eligible cells count, and
  aggregation follows `score_task2.py`. Models are trained 2-fold by day parity.
  An earlier evaluator that used all mined onset events did **not** match the leaderboard, because
  the selector's window distribution is different.
* T3: `tfb/t3_proxy.py` (S_LWR ≈ 1 − Σ|ΔN_pred − ΔN_true|/Σ|ΔN_true|). Only relative changes are
  meaningful; the absolute level is not calibrated yet.

| Change | Local predicted Δtotal | Online Δtotal |
|---|---|---|
| v1→v2 (onset static set) | +0.105 (S_queue 0.39→0.74) | +0.113 |
| v2→v3 (T1 LightGBM, T2 models, with gap bug) | +0.041 (T2 +0.028, T1 +0.012, phys ≈ +0.008, gap bug −0.007) | +0.0375 |
| v3→v4 (gap specialist) | +0.0069 | +0.0069 |

Local T2 (v3/v4): S_queue 0.833 (onset 0.828, ongoing 0.838). Only 40 windows per condition, so the
standard error is about 0.04; improvements under ~0.02 need the leaderboard.

Submission plan for the next day (5/day, resets 00:00 UTC). Each probe changes one factor only.
1. v5 = v4 + full-train T1 model (`state_lgbfull_gap.csv`), for T1 transfer.
2. Probe: v4 with every queue_pred = 0 → gives the online S_queue(v4) exactly
   (S_queue = Δ/0.30), which calibrates the local 0.833.
3–5. T4 probes on top of the best: `odme_sl2`, `odme_wl2`, `odme_geo_l2`. Each changes only S_ODME,
   so ΔS_ODME = Δ/0.20.
