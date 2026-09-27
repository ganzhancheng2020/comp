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
The zips are prebuilt in `out/tfb/`. Submit with `python -m tfb.lb out/tfb/sub_<name>.zip "<msg>"`.
The probes use v4 (0.86257) as the base, so each leaderboard delta maps directly onto one component.
Order (revised 2026-09-26 19:30 UTC): `sub_v5_t2new` (v4 + retuned T2: onset num_leaves 31, ongoing
num_leaves 255 / 1000 rounds), then `sub_v6_t2new_t1full` (v5 + full-train T1), then `sub_p_sl2`,
`sub_p_plen`, `sub_p_geo`. `sub_p_q0` (S_queue calibration) moves to a later day: two leaderboard
deltas already confirm the T2 evaluator.

Large, selector-aligned T2 evaluators (`tfb/t2_eval_big.py`) are built from mined train events with
the selector's filters: the horizon has an eligible queued cell, history coverage is ≥ 0.7, and for
ongoing windows the persistence IoU is ≤ 0.9. IoU counts eligible cells only.
* Onset: 1,900 events. The static set scores 0.713 here, against 0.725 on the organizer's windows
  and ≈ 0.75 implied online, so it is aligned. Model 0.880; num_leaves 31 → 0.898. A 0.5-threshold
  decode drops to 0.776.
* Ongoing: 5,760 windows. Persistence 0.644 and model 0.788, which is about 0.1 below the level on
  the organizer's windows, so use it for relative comparisons only. num_leaves 127 → 0.808;
  num_leaves 255 with min_data 100 → 0.812.
* The earlier event-based evaluator did not align because it ignored eligibility: IoU is scored on
  eligible cells only, and bottleneck clusters often contain ineligible cells.
1. v5 = v4 + full-train T1 model (`state_lgbfull_gap.csv`), for T1 transfer.
2. Probe: v4 with every queue_pred = 0 → gives the online S_queue(v4) exactly
   (S_queue = Δ/0.30), which calibrates the local 0.833.
3–5. T4 probes on top of the best, each testing one hypothesis (ΔS_ODME = Δ/0.20):
   `odme_sl2` (rescale, then L2: is zeroing small ODs good?), `odme_plen_l2` (per-path shift that does
   not grow with path length: is the prior noise additive per path?), and `odme_geo_l2` (pooled prior
   across splits: is the truth closer to the shared structure?).

T4 proxy that was rejected: "similarity to another split's prior". It ranks KL above L2
(0.638 vs 0.592), which contradicts the leaderboard (L2 beats KL by ~0.15 S_ODME). So the truth is not
the structure the priors share; it is sparser and more concentrated, and T4 can only be judged online.
Log prior = gravity fit (R² 0.75) + residual with sd 0.6. The residuals of different splits'
priors are uncorrelated (≈0.03).

## 2026-09-27 submissions

| Sub | Change (one factor) | Public LB | Δ | Reading |
|---|---|---|---|---|
| v5 | v4 + retuned T2 | 0.86784 | +0.0053 vs v4 | Local predicted +0.0065, so the big T2 evaluators align |
| v6 | v5 + full-train T1 | **0.86864** | +0.0008 vs v5 | Best so far; the previous public top was 0.86711 |
| p_sl2 | v4, T4 prior rescaled then L2 | 0.80288 | ΔS_ODME −0.30 | Truth mass is near the raw prior level, not the counts-scaled level |
| p_plen | v4, T4 per-path constant shift | 0.83983 | ΔS_ODME −0.11 | Plain L2's per-link accumulation is better |
| p_geo | v4, T4 pooled prior across splits | 0.81225 | ΔS_ODME −0.25 | Truth follows its own split's prior |

New T4 hypothesis: the prior loads ~2× the counts, but the truth mass is near the prior. So f* may not
satisfy A f* = c exactly, and exact projection trades S_od (0.45) for S_link (0.25).
Test with the one-dimensional family f = (1−α)·f_L2 + α·b on top of v6:
`sub_p_blend25` (α=0.25, local S_link 0.785, at most −0.011 total from S_link) and `sub_p_blend50`
(α=0.5, S_link 0.569, at most −0.022). If either beats v6, move α toward the peak.
Later: `sub_p_q0` (S_queue calibration).

## Leaderboard position (2026-09-27 ~00:30 UTC): rank 19, 0.86864; the top is 0.92406 (gap 0.055)

Gap attribution (estimated):
* S_state ≈ 0.94: aligned, at the noise floor.
* S_queue ≈ 0.79 online. Anchor: the official persistence baseline is 0.3017 on validation; adding
  the v2 and v5 deltas gives ≈ 0.79. Local predicts ≈ 0.85, so there is a 0.06 local/online gap to
  explain.
* S_phys ≈ 0.7: bounded by the noise for everyone.
* S_ODME ≈ 0.85–0.9.
The gap is mostly in T2 and T4.

Next-day submissions (base v6), revised:
1. `sub_p_q0v6`: every queue prediction 0 → S_queue(v6) = (v6 − score) / 0.30.
2. `sub_p_onset0v6`: onset windows zeroed → onset IoU = 2·(v6 − score)/0.30 × (#families
   weighting is equal), then ongoing IoU = 2·S_queue − onset IoU.
3. `sub_p_o0v6`: T4 all zeros. S_ODME(0) = 0.15·S_dev(0) + 0.15·0.5 ≈ 0.085–0.13, so
   S_ODME(L2) ≈ (v6 − score)/0.2 + ~0.1. With probe 1 this also gives S_phys by subtraction
   (S_state ≈ 0.94 from the aligned local evaluator).
4. `sub_p_blend25`: T4 α = 0.25 blend (the "truth ≠ counts" hypothesis).
5. `sub_p_blend50` if blend25 improves; otherwise the best T2 improvement of the day.

Findings from the toolkit's git history (MIT):
* The first versions scored T4 against a ridge solve (NNLS, λ=0.05) over the prior and counts.
  The docs now say the leaderboard truth is organizer-held.
* The "naive baseline" S_ODME of 0.8359 uses TRAIN counts. Our L2 uses the split's own counts, and
  it equals the split's ridge solve to within 0.04% L1.
* So if the leaderboard truth is that solve, our T4 is already about 1.0, and the gap to the top lies
  in T2 and physics. Probe 3 decides this.
* `base_od.csv` (removed on 2026-09-11) was the private month's prior. We have it anyway as
  `task4/<panel>/private/synthetic_weak_prior.csv`.

T2 ongoing, shockwave/queue-geometry features (distance to nearest queue up/down at lags 0/3/6/12,
run length, up/downstream demand): 0.8122 → 0.8136 on the big evaluator, which is within noise.
T2 local level is aligned absolutely too: official-style persistence scores 0.3074 locally against
0.3017 online. The earlier "0.06 local/online gap" came from a wrong anchor and does not exist.
Online S_queue(v6) ≈ 0.855.

Physics, local estimate (`tfb/t3_proxy.py`: official S_FD plus the S_LWR proxy):

| Prediction | S_FD | S_LWR proxy | S_phys |
|---|---|---|---|
| truth | 0.99 | 1.00 | 0.997 (the official perfect answer gets 0.96 online) |
| LightGBM | 0.99 | 0.60–0.63 | 0.73–0.75 |
| interp | 0.99 | 0.52–0.53 | 0.68 |

The noise in log v and log q is independent (residual correlation ≈ 0.00), so k = q/v carries both
(2nd-difference sd of log k is 0.059, against 0.055 for log q). There is no smoother density to
exploit, and the S_LWR floor is the same for everyone at T1 ≈ noise floor. So the remaining gap to
the top is in T2 and/or T4; the probes above decide which.
