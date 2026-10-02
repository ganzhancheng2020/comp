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

## Submission policy (from the user, 2026-09-27: every submission is expensive, do not waste any)

* No pure diagnostic probes. Zeroing T2, T4 or the onset windows is cancelled, because those
  submissions can only lower the score.
* Submit only a candidate whose local, leaderboard-aligned evaluators predict a total gain of at
  least +0.003. Merge several small improvements into one submission.
* T4: four probes (KL, sl2, plen, geo) all lost to L2, and L2 equals the split's ridge solve to
  within 0.04%. The blend probes are on hold until a new hypothesis with real upside exists.
* Before each submission, write down the expected Δ and its components. Afterwards, record the
  actual Δ in the calibration table.
* The prebuilt diagnostic zips in `out/tfb/` (`sub_p_q0v6`, `sub_p_onset0v6`, `sub_p_o0v6`,
  `sub_p_blend*`) are not to be submitted.

## Candidate v7 (built 2026-09-27 02:14 UTC; to be submitted after the 09-28 00:00 UTC reset)

`out/tfb/sub_v7.zip` = v6 with the T2 predictions replaced by `queue_models3.csv`. T1 and T4 are
unchanged.
* Onset v2 (`tfb/t2_onset2.py`): a cluster-level model predicts which bottleneck cluster activates;
  a link-level model conditioned on an active cluster predicts the links inside it; joint Monte-Carlo
  decoding keeps within-cluster correlation.
  Error analysis of v1: 82% of events were perfect; the loss split into over-prediction 0.040,
  under-prediction 0.026 and wrong cluster 0.023. Most of the over-prediction was hedging across two
  clusters, which the independent-link decoding causes.
  Big evaluator: 0.898 → 0.9175.
* Ongoing, wider candidate set: ±12 links plus all onset bottleneck links, with an `is_bneck` feature.
  Error analysis: 34% of all errors were truth cells outside the old ±6 candidate set, from queue
  extension beyond 6 links or new queues at other bottlenecks.
  Big evaluator: 0.8136 → 0.8313. The shockwave features added +0.0014 of that.
* Organizer train windows (2-fold, noisy): S_queue 0.8426 (onset 0.818, ongoing 0.867), against
  0.833 for the v3/v4 pipeline. No regression.
* **Expected Δ vs v6: about +0.004 to +0.006** (S_queue +0.019 × 0.30, rounded down for the noise
  in the organizer windows).

## T4 closed (2026-09-27): a local proxy that reproduces all five online points

l2dev = ‖f − b‖₂ / ‖b‖₂ (relative L2 deviation from the split's own prior, family-averaged).

| Method | l2dev | Online ΔS_ODME | −0.95·Δl2dev |
|---|---|---|---|
| L2 | 0.190 | 0 | 0 |
| plen | 0.303 | −0.114 | −0.107 |
| KL | 0.349 | −0.151 | −0.151 |
| geo | 0.456 | −0.250 | −0.253 |
| sl2 | 0.507 | −0.300 | −0.301 |

Over the feasible set {Af = c, f ≥ 0}, the L2 projection minimises l2dev by construction, so it is
optimal under this proxy.
Blending towards the prior trades S_link for l2dev and loses: about −0.004 at α = 0.1 and −0.009 at
α = 0.25.
**T4 is closed: keep L2, no more T4 submissions.**

Rules: Task 2 is defined as online ("at forecast origin T, participants receive the previous 60
minutes"), and recovering hidden labels is prohibited.
Observations after the post-horizon buffer (T+95 onwards) are therefore NOT used for T2, even though
they are in the release.

## Candidate v7b (supersedes v7; built 2026-09-27 07:42 UTC)

`out/tfb/sub_v7b.zip` = v7, with the ongoing model trained on 6,000 windows per corridor instead of
3,000. The training frame lives in per-panel parts (`data_tfb/cache/t2_ongoing_parts/`, 18.6M rows).
* Learning curve: halving the 3k set gives 0.8171, the 3k set gives 0.8313, and the 6k set gives
  0.8333. The 6k evaluation sample was slightly harder: its persistence baseline is 0.6381 against
  0.6442 before. So the lift over persistence rises from +0.187 to +0.195.
* **Expected Δ vs v6: about +0.005 to +0.007.** Onset +0.0195 and ongoing ≈ +0.025, so S_queue
  ≈ +0.022, times 0.30.

## Candidate v7c (supersedes v7b; 2026-09-27 07:58 UTC)

v7c = v7b with the T1 gap specialist retrained on 200k rows per panel (`t1gap200_*`, applied with
`GAP_MODEL=t1gap200 python -m tfb.patch_gaps ...`). Gap-cell RMSE: speed 7.79 → 7.58, flow/lane
79.1 → 77.4. That adds about +0.0003. **Expected Δ vs v6: +0.005 to +0.007.**

Hypotheses checked and rejected on 2026-09-27:
* Conservation-derived density. corr(ΔN, dt·(q_up − q + ramps)) is only 0.09–0.20, and the
  topology flux sd (13–22) far exceeds the ΔN sd (1.4–3.4). This matches the docs, so the physics
  floor holds.
* Clean vs noisy T2 labels. Median-3 smoothing flips ≤ 0.1% of queue cells, so it is negligible.
* Public notebooks (best 0.809) hold nothing new. The 0.809 notebook's v11→v13 step (only T4 → ridge
  λ=0.05) gained +0.0108 online, consistent with the leaderboard truth ≈ the ridge solution, so T4 is
  closed.

More rejected hypotheses (2026-09-27, local and aligned):
* Onset with earlier-day features (queue counts, time since the last queue, min ratio before T−60,
  from the masked layer; causal): 0.9153 with link-level features only and 0.9137 with
  cluster-level too, against 0.9175 without. No gain, so onset looks saturated at ≈ 0.915–0.92.
* Temporal smoothing of predicted density at target cells (normalised 3-tap, α 0.25/0.5, flow
  re-derived as k·v): net total between −0.001 and +0.0003. No gain; the model's density is already
  smooth.

## Reachable score and the gap to #1 (analysis, 2026-09-27)

Headroom left for legitimate improvements (each estimated with aligned local evaluators):
* T1: noise floor (speed σ ≈ 1.7, flow/lane ≈ 30). The rest is in the gap cells, ≤ +0.0025 total.
* Physics: floor for everyone; conservation and smoothing were both tested without gain.
* T4: ≈ optimal under the l2dev proxy that reproduces all five online points.
* T2: v7c ≈ onset 0.918 and ongoing 0.835 (big evaluators). With more work, maybe +0.005 total.
So the reachable total is ≈ 0.875–0.88, and #1 is 0.924.

The most plausible source of the remaining ≈ 0.045 is T2 read-back from observations published after
the post-horizon buffer (T+95 onwards). Queues last hours, so the far side reveals which cluster
queued (onset) and whether the queue persisted (ongoing). T2 near 1.0 would add ≈ +0.03–0.04.
We do NOT use this: Task 2 is defined as online (60-minute history), and recovering hidden labels is
prohibited. The account holder could ask the organizers in the competition forum whether
post-buffer observations are allowed for Task 2. If they rule they are allowed, it becomes a large
and legitimate lever. Until then we stay within the documented information set.

## Ongoing CNN (`tfb/t2_cnn.py`, 2026-09-27)

A 1-D fully-convolutional net over links. The input is the 13-step history × channels (speed
ratio, flow/cap, queue flag, missing flag, bottleneck flag, position, tod, dow, panel one-hot); the
output is 6 steps × links. Dilated residual blocks, BCE weighted towards eligible cells.
Big evaluator, same test windows as the LightGBM: 8 epochs 0.805, **25 epochs 0.841**, against
0.835 for LightGBM. The loss was still falling at 25 epochs.
Next: `tfb/t2_blend_eval.py` (40 epochs, LightGBM + CNN blend weights 0–1).

## Blend result and candidate v8 (2026-09-27 21:04 UTC)

`tfb/t2_blend_eval.py`, same 5,805 test windows, 2-fold:

| Ongoing | IoU |
|---|---|
| LightGBM (the v7c pipeline) | 0.8342 (0.8351 with a 0.5 threshold) |
| CNN, 40 epochs | 0.8381 |
| **0.5·LightGBM + 0.5·CNN, threshold 0.5** | **0.8483** |
| 0.3·LightGBM + 0.7·CNN, threshold 0.5 | 0.8473 |

The blend adds +0.014 on ongoing, so S_queue ≈ +0.007 and the total ≈ +0.002 on top of v7c.
(The earlier blend run at 10:55 died silently from a memory-cgroup OOM while loading the full
frame. The evaluator now reads per panel and keeps only the fold's rows as float32.)

v8 (`tfb/t2_final.py`, built 22:27 UTC, `out/tfb/sub_v8.zip`) = v7c with ongoing replaced by the blend. It is trained on all windows, with a
25-epoch CNN. **Expected Δ vs v6: about +0.007 to +0.009.** If it is ready before 00:05 UTC, it
replaces v7c as the day's single submission.

## CNN seed ensemble (2026-09-27 23:34 UTC)

`tfb/t2_blend_eval2.py`, same 5,805 windows. The second CNN is seed 2 at 25 epochs.

| Ongoing | IoU (threshold 0.5) |
|---|---|
| CNN1 | 0.8373 |
| CNN2 | 0.8382 |
| mean of the two CNNs | 0.8461 |
| blend with LightGBM, CNN1 | 0.8483 |
| blend with LightGBM, CNN2 | 0.8513 |
| **blend with LightGBM, mean of the two CNNs** | **0.8531** |

The seed ensemble adds ≈ +0.003–0.005 on ongoing over v8.
v9 production (`tfb/t2_final2.py`): LightGBM maps plus four CNN seeds (0 is saved; 1, 2 and 3 are
being trained), with the maps cached in `t2_final_maps.pkl`. Output: `queue_models6.csv`.

## v8 submitted 2026-09-28 00:06 UTC: 0.86849 (v6 0.86864). Expected +0.007–0.009, got −0.00015

Diagnosis:
* No bug. Per-window agreement between the v6 and v8 T2 predictions on validation is IoU 0.80
  (onset) and 0.88 (ongoing). A handful of onset windows switch cluster, and a few ongoing windows
  predict larger areas.
* **The public-LB T2 sample is too small to resolve these gains.** Bootstrapping the big-eval
  per-window deltas (blend − LightGBM, ongoing: mean +0.015, sd 0.097) at the public-LB size
  (5 windows per panel-condition): ongoing ΔIoU has mean +0.013 and sd 0.017, and P(Δ ≤ 0) = 0.21.
  Onset (0/1 flips) is noisier still. A ~0 online delta is therefore not evidence against the
  change.
* The private LB (April) has the same 80-window size. The expected private gain is what the
  thousand-window local evaluators measure. **Choose the final private submissions by local
  expected value, not by public-LB differences of ±0.003.** Use the public LB only to catch large
  deviations and bugs.
* A weak warning sign existed: onset v2 scored 0.818 on the organizer train windows against 0.828
  for the old model (n=40, SE ≈ 0.04). It is kept in mind, but not acted on alone.

## 2026-09-28 03:30 UTC: v9 / queue_models7 (current best local candidate, not submitted)

* v9 (`sub_v9.zip`, `queue_models6.csv`): ongoing = 0.5·LightGBM + 0.5·mean of four CNN seeds,
  threshold 0.5. Local vs v8: ongoing about +0.004–0.006, total ≈ +0.001. Below the +0.003 threshold,
  so it is held.
* Onset mixture (`predict_event_mix`: 70% cluster-level v2 joint samples + 30% old independent link
  model). Big evaluator: v2 alone 0.9147, **mix 0.3 0.9179**, mix 0.5 0.9157, old alone 0.9010.
  On validation/private it changes almost nothing (99.99% agreement). `queue_models7.csv` = v9 + mix.
* Month adaptation (mine events from the observed parts of validation/private): rejected. Cluster
  shares per train month vary within binomial noise (≈30 events per month, sd ≈ 0.09).

Status: the local T2 headroom is nearly used up. The next submission should bundle queue_models7
with any further gain; the combined local expectation vs v8 is ≈ +0.0015–0.002.

T1 congestion check (2026-09-28). Congested cells (speed < 0.8·v_free) are 4.4% of targets and 28%
of the speed MSE; on D12_I5_N they carry 36% of the |N| error that drives S_LWR. But their white-noise
upper bound is 3.3–4.7 km/h (2nd difference, against 1.45–2.1 in free flow), and our congested RMSE
is 4.0–4.4, which is at that floor. So congestion is saturated too, and there is nothing to gain in
T1 or physics there.

T4 ridge-λ sweep (2026-09-28). The l2dev proxy extrapolated to S_link < 1 favours a larger λ:
λ_rel 0.05 gives +0.0034 S_ODME. But the 0.809 public notebook contradicts it: dropping S_link only
0.0127 below the λ=0.05 ridge cost that team 0.054 S_ODME online. So the proxy does not extrapolate
off the feasible set, and the truth ≈ the λ=0.05 ridge. T4 stays at L2 (≈ that ridge); it is
closed for good.

In progress: ongoing features for episode duration (`early_features`: episode age, queued counts
earlier today, time since the last queue, 3-hour queued fraction, all from the masked layer before
T−60). Frame `t2_ongoing_parts_early`, evaluated against 0.8333 on the same windows.

## v9b submitted 2026-09-28 ~08:00 UTC: **0.87104 (new best)**

v9b = v8 T1/T4 + `queue_models7` (onset mixture + ongoing 0.5·LightGBM + 0.5·mean of four CNN
seeds). It was submitted mainly so that the best-expected-value model is eligible for the final
private selection.
Local expected Δ vs v8 was +0.0015–0.002; online Δ vs v8 is +0.0026 (vs v6: +0.0024). Consistent
within noise.
Leak check: T2 uses only the 60-minute history; T1 uses only the published masked layer; T4 uses
only the split's own prior and counts. No post-buffer observations and no hidden labels.
Final-selection candidates so far: v9b (best local EV and best public), then v8.

Episode-duration features for ongoing (2026-09-28, lean evaluator `tfb/t2_lean_eval.py`, same 5,805
windows, LightGBM with threshold 0.5): 0.8349 → **0.8367 (+0.0018)**. Inside the CNN blend that is
≈ +0.001 IoU, ≈ +0.0002 total. Kept (frame `t2_ongoing_parts_early`) for the next bundle; not a
submission on its own.
Process note: the first evaluations were OOM-killed (the wider frame did not fit the pandas-based
evaluator), and a watcher waited on its own pgrep match. Watchers now wait on PIDs, and the
evaluation is per-panel float32.

## 2026-09-29: the local/online T2 gap is a train/inference mismatch (origin row T)

* Probe `v9b_static` (v9b with onset = static per-panel set): **0.85012**, against v9b 0.87104.
  So the onset model is worth +0.021 total online (≈ +0.14 onset IoU; the local estimate was +0.2).
  The onset model transfers.
* Decomposition (v1 is the official persistence 0.3017; v1→v2 changed only T2): online S_queue
  (v9b) ≈ 0.788. With onset ≈ 0.86, **ongoing online ≈ 0.72 against 0.853 locally**. Yet the ongoing
  persistence baseline matches (0.638 both). So the ongoing MODEL does not transfer.
* Root cause: `window_history` covers T−60…T−5 (12 slots). **The origin row T is never
  published**, in all three splits. All T2 training and local evaluation used
  `z["speed"][d, T-12:T+1]`, so the origin row was present and observed. The models learned to
  rely on r0 (the ratio at T), which is only a forward-filled T−5 value at inference.
* Fix (`t2_events.visible()`): the T row is NaN in every train frame and evaluator (ongoing
  LightGBM, onset, CNN, persistence baselines). The onset frame is rebuilt; onset with visible
  history is 0.9127 (vs 0.9179 with the T row), so it was only mildly affected. The ongoing frame
  `t2_ongoing_parts_vis` is rebuilt with the episode-duration features.
* Also: validation/private ongoing windows have smaller queues than the organizer's train windows
  (links queued at the origin 23 / 21 vs 36). The big evaluator already matches this (23.4). IoU
  rises steeply with queue size (0.78 for 0–5 links, 0.96 for > 40 links); threshold 0.5 is best
  in every size bucket.

### Local confirmation of the origin-row fix (2026-09-30 01:42 UTC)

Lean evaluator, 6,396 ongoing test windows, all with the realistic input (origin row T missing):

| LightGBM ongoing | IoU (threshold 0.5) |
|---|---|
| trained on the old frame (T row present), i.e. what is online now | 0.802 |
| **trained on `t2_ongoing_parts_vis` (T row missing, plus episode-duration features)** | **0.8352** |

The old model scored 0.835 on test data *with* the T row, so the mismatch cost ≈ 0.033 ongoing IoU.
The fix recovers all of it. The CNN was also trained with the T row, so it is presumably degraded
in the same way.

### Submission log for 2026-09-29 UTC (v9b_static, P1, P2, S3, S4; earlier drafts of this section said 09-30)

**P1: probe, "does the origin-row fix transfer online?"**
* Change vs v9b: only the LightGBM inside the ongoing blend is replaced by the T-row-fixed model
  (`t2_final3.py`, maps in `t2_final_maps_vis.pkl`). The CNN (4-seed mean, old), the 0.5/0.5
  blend, onset, T1 and T4 are all unchanged.
* Local expected Δ: LightGBM +0.033 on realistic inputs → blend ≈ +0.015–0.02 ongoing IoU →
  S_queue ≈ +0.008–0.01 → **total ≈ +0.0025–0.003**. Public-LB noise sd ≈ 0.0025 on ongoing
  alone.
* Purpose: probe. If the delta is ≥ +0.002, the fix is confirmed online and P2 (CNN retrained on the
  visible history) follows. If ≤ 0, re-check the inference path for validation/private.
* Actual: **0.87037** (v9b 0.87104), so Δ = −0.0007 against an expected +0.0025–0.003. That is
  about 1.3 sd below expectation: not a confirmation and not a refutation. The ongoing gap still
  looks larger than the origin row alone explains (see the next check).
* Check: the NaN pattern of `window_history` equals that of `z["speed"]` (16.3% missing in both).
  The earlier "23%" figure included the always-missing origin row, so there is no second input
  mismatch.

**P2: probe/breakthrough, "the full origin-row fix: LightGBM and CNN both trained on the visible
history"**
* Change vs P1: the ongoing CNN is replaced by a mean of two new seeds (1, 2) trained on the visible
  history (T−60…T−5), plus the four early-day channels (`TFB_CNN_EARLY=1`, maps in
  `t2_final_maps_early.pkl`). The LightGBM is the fixed one from P1; the 0.5/0.5 blend with
  threshold 0.5 is unchanged, and onset, T1 and T4 are the same as v9b.
* Local expected Δ: not measured directly for the CNN yet (a 2-fold evaluation takes about 2 h). By
  analogy with the LightGBM (0.802 → 0.835 on realistic inputs), the CNN half gains a similar
  ≈ +0.03, so the blend ongoing ≈ +0.015 vs P1. That is **total ≈ +0.002–0.003 vs P1, ≈ +0.004–0.006
  vs v9b**, although this CNN has 2 seeds instead of 4 (−0.002 ongoing).
* Purpose: probe for criterion B (a T-row-fixed version at ≥ +0.005 over v9b online) and
  confirmation of the fix.
* Actual: **0.86921**, i.e. Δ vs v9b −0.0018 and Δ vs P1 −0.0012, against an expected +0.002–0.003
  vs P1 (≈ 2.7 sd below expectation vs v9b). P1 and P2 share the same 40 validation ongoing windows,
  so their errors are correlated; they are not independent evidence. But both point the same way.

### Conclusions of the three probe loops (2026-09-29/30)

| Loop | Hypothesis | Local evidence | Probe | Online | Conclusion |
|---|---|---|---|---|---|
| 1 | The onset model does not transfer online | the gap decomposition | `v9b_static` | 0.85012 (−0.021) | Rejected: the onset model is worth ≈ +0.14 onset IoU online |
| 2 | The origin-row mismatch costs the ongoing LightGBM | 0.802 → 0.835 (realistic inputs) | P1 | 0.87037 (−0.0007) | Not confirmed online |
| 3 | The full fix (LightGBM + CNN) | extrapolated +0.004–0.006 vs v9b | P2 | 0.86921 (−0.0018) | Not confirmed; the local/online gap on ongoing persists |

Remaining gap: the best public score is v9b at 0.87104 (#1 is 0.924). Online ongoing ≈ 0.72 against
≈ 0.80–0.85 locally (realistic inputs). There is still an unidentified systematic difference between
the local ongoing simulation and the online ongoing windows or truth. Candidates:
(a) online truth from the noise-free state gives less flicker at queue edges (our labels flicker
0.44–0.64 per cell), which favours contiguous or persistent predictions;
(b) online `is_score_eligible` for horizon cells differs from the observation-based eligibility;
(c) the selector's ongoing origins differ from uniform sampling within episodes.
Final-selection plan: hedge with **v9b** (best public) + **P2** (full fix, best local expected
value).

### Remaining submissions on 2026-09-29 UTC (S3, S4; S5 does not exist: v9b_static was the first of the day)

Observation behind S3: online v9b ≥ P1 ≥ P2, while locally the order is reversed.
Interpretation: the old models, trained with the observed origin row but fed a 5-minute-old value
at inference, behave more like persistence (conservative). The fixed models extrapolate change more
aggressively. Hypothesis (a): the online truth, from the noise-free state, is smoother and less
dynamic than our noisy labels, so persistence-like predictions score higher online.

**S3: probe, "does a more persistence-like ongoing score higher online?"**
* Change vs v9b: ongoing p = 0.7·p_v9b + 0.3·persistence (last visible state, T−5 or earlier),
  threshold 0.5. Onset, T1 and T4 are the same as v9b (`queue_models10_s3.csv`; 0.32% of cells
  differ).
* Local expected Δ (`tfb/t2_pers_mix_eval.py`): ongoing 0.8666 → 0.8402 (−0.026), i.e. **total
  ≈ −0.004**. The local truth is the noisy label.
* Decision rule. If online ≥ v9b (Δ ≥ 0), hypothesis (a) is supported; S4 then applies the same
  α=0.3 mix to P2 (the fixed models), or raises α to 0.5 on v9b. If online ≤ v9b − 0.003 (roughly
  as local predicts), hypothesis (a) is rejected. S4 then becomes the onset-refit probe
  (origin-row-fixed onset on top of v9b), local expected ≈ +0.001.
* S5 is held for a merge of whatever S3/S4 confirm; if nothing is confirmed, it stays unused.
* Actual: **0.86795** (Δ vs v9b −0.0031; local predicted −0.004). **Hypothesis (a) is rejected.**
  The online direction matches local on this axis, so the local ongoing evaluator is right about
  persistence mixing.

**S4 revised before use (the rule allowed onset refit, but its expected +0.001 is below the noise).**
New hypothesis (d): the selector drops ongoing windows whose *official-style* persistence IoU
(last visible row, eligible cells, no forward-fill) exceeds 0.9. Our local evaluators filter on the
*forward-filled* persistence, which is higher, so they also drop "static-queue" windows that stay
online. On those windows a conservative model (the old one, which treats the T−5 value as the value
at T) beats a model that extrapolates change (P1/P2). That would explain online v9b ≥ P1 ≥ P2 while
S3 still matches local.
Local test first: re-run the old-vs-fixed LightGBM comparison with the selector-style filter.
S4 is decided on that result:
* if it shows old ≥ new, S4 is not needed (keep v9b and restore the fix only for dynamic windows);
* if it shows new > old, the hypothesis is rejected, and S4 = onset refit or holding. Private uses different windows, and local evidence is statistically much larger but has a
known unresolved bias.

**Decision (account holder, 2026-09-30):** observations after the post-horizon buffer (T+95 onwards)
will NOT be used for T2 under any circumstances. It would be look-ahead leakage and break the rules.
All work stays within the documented information set.

### Hypothesis (d) result: rejected (2026-09-29 04:50 UTC)

Lean evaluator, same test windows, all with the realistic input (T row missing), three selector filters:

| Filter | windows | official persistence | old LightGBM (online) | fixed LightGBM (P1/P2) |
|---|---|---|---|---|
| ffill persistence ≤ 0.9 (old evaluators) | 6,396 | 0.516 | 0.802 | 0.835 |
| **official persistence ≤ 0.9 (selector-style)** | 10,844 | **0.608** | 0.893 | **0.914** |
| no filter | 11,771 | 0.637 | 0.900 | 0.920 |

Official-filter persistence 0.608 matches online v1 (0.603 implied), so this filter is the aligned one. The
fix still wins by +0.021 under it, so (d) does not explain online v9b ≥ P1 ≥ P2.

### Where is the local/online T2 gap: onset or ongoing? Two decompositions disagree

* Anchor A (v1→v2): v1 predicts **0 onset cells** (checked), and v1→v2 changed only onset (empty →
  static set), +0.113 online → online static onset ≈ 0.75 (local 0.713–0.725). v9b_static → v9b is
  +0.14 onset → **v9b onset online ≈ 0.89 (local 0.91): onset aligned**. Then, with the non-queue components
  reconstructed from the calibration table, S_queue(v9b) ≈ 0.78 → **ongoing online ≈ 0.68** against
  0.89 locally (official filter).
* Decomposition B (earlier today): take ongoing as local and solve for onset → onset online 0.57–0.68.
  This contradicts anchor A, and A uses fewer assumptions.

### Validation and private are different traffic scenarios from train (new, 2026-09-29)

`synthetic_release_v1.json`: "The scenario seed for validation and private is drawn at random". The
published masked layer shows whole bottleneck clusters switching on or off between splits (share of days
with a queue; masking rates are the same, ≈ 0.43, in all splits, so this is not masking):

| Panel | Links | train | validation (Mar) | private (Apr) |
|---|---|---|---|---|
| D7_I10_E | 11–23 | 0.71 | **0.03** (min speed 108) | 0.70 |
| D7_I210_W | 62–66 | 0.6–0.74 | **0.00** | **0.07** |
| D7_I405_S | 81–89 | 0.01–0.04 | 0.06 | **0.33–0.80** |
| D7_I405_S | 90–93 | 0.7 | **0.06** | 0.87 |
| D12_I5_N | 111–114, 188–195 | 0.4–0.75 | **0.00–0.03** | **0.00–0.07** |
| D12_I5_S | 43–45 (main onset cluster) | 0.89 | **0.45** | 0.93 |

The change is abrupt at the split boundary (last days of train still show the train pattern), and all
T2 windows sit in the first ~7 days of each split, so earlier days of the same scenario are almost
never available before a window origin. Opportunity size (a diagnostic on onset events mined from the
masked layer; no model uses it): a static onset set fitted on train scores 0.398 (val) and 0.442 (pri),
and an oracle set fitted 2-fold inside the split scores 0.448 / 0.439. So adapting to the scenario would
help **validation (public LB) by ≈ +0.05 onset IoU, mostly D12_I5_S (0 → 0.29) and D7_I405_S, and private not
at all**. Decisions:
* No scenario adaptation. It could only use post-origin data from the same split (forbidden), and
  private does not need it.
* Public-LB T2 is biased by the validation scenario. Final selection should keep using local expected
  value, not public deltas below the noise.

**S4: probe, "is the local/online T2 gap in ongoing (anchor A) or in onset (decomposition B)?"**
* Change vs v9b: only the 40+40 ongoing windows are replaced by the official persistence baseline
  (the v1 file, `queue_models11_s4_persong.csv`; 3,438 cells differ). Onset, T1 and T4 are the same as v9b.
* Expected: under B (ongoing online ≈ local 0.89): 0.871 − 0.15·(0.89 − 0.60) ≈ **0.828**. Under A (ongoing
  online ≈ 0.68): 0.871 − 0.15·(0.68 − 0.60) ≈ **0.859**. The gap between them is 0.03 against a
  noise sd ≈ 0.0025, so the probe is decisive.
* Purpose: probe. It measures the ongoing model's online advantage over persistence directly, and
  that decides where the remaining month goes: A → the ongoing inputs/labels (the online ongoing truth or
  windows differ from our simulation); B → onset (the validation scenario shift above already explains
  much of it, so the private expectation stays).
* Actual: **0.83590** (Δ vs v9b −0.0351) → Og(v9b) − Pers_official = 0.234 online.
* Check that broke anchor A: v2's ongoing was **our forward-filled persistence** (5,532 cells), not the official
  one used by v1 and S4 (4,374 cells; S4 = v1 on ongoing exactly). So v2 − v1 = 0.15·(onset_static + P_ffill − P_off).

**Closed decomposition (5 submissions, v1/v2/v9b/v9b_static/S4; P_off = 0.608 and P_ffill = 0.786 from the
official-filter lean evaluator):**

| Component | online | local (official filter) | gap |
|---|---|---|---|
| ongoing persistence (official) | 0.603 (v1) | 0.608 | 0.005 (aligned) |
| onset static set | 0.58 | 0.71 | 0.13 |
| **onset v9b** | **0.72** | 0.91 | **0.19** |
| **ongoing v9b** | **0.84** | 0.89–0.90 | **0.05** |

Consistency: the non-T2 part rises by +0.024 from v2 to v9b, as the local T1/physics estimates predict
(+0.02–0.03). Decomposition B was right in direction, and anchor A was wrong.
Interpretation: onset has the bigger gap. Per-window IoU sd is 0.375 (release config), so a 40-window
condition mean has SE ≈ 0.06; the onset gap is ≈ 3 SE and the ongoing gap ≈ 1 SE. On onset, part of
it is the validation-specific scenario shift above (D12_I5_S 43–45 queues on 45% of March days vs 89% in
train; D7_I405_S 90–93 is nearly off in March). Private looks like train for onset (static-set diagnostic 0.442 in both
the train-fitted and the oracle set), so the private onset expectation is closer to local than the public one.

### Loop 5 conclusion and next steps (2026-09-29 05:10 UTC; today's 5 submissions are used)

* The ongoing model transfers (+0.234 over persistence online vs +0.28–0.29 locally). The remaining ongoing gap
  (≈ 0.05) is within about 1 SE.
* The onset gap (≈ 0.19) is the largest single item: small-sample noise (SE ≈ 0.06) plus the validation
  scenario shift. Nothing compliant fixes the shift for public (it needs post-origin data from the same
  split), and private does not show it.
* Direction for the remaining month: make onset robust to scenario shift from the window history alone
  (lean less on link-identity priors, more on the state 60 minutes ahead, the organizers' "1.28× threshold"
  signal). It can only be measured on events mined from the val/private masked layer, used for evaluation only.
  The account holder must decide whether that evaluation-only use is acceptable (it is post-origin data,
  even though no prediction consumes it).
* Final selection: keep **v9b + P2**. Public deltas between them (−0.0018) are below one SE of the 80-window
  public T2 sample, and the local expected value favours the origin-row fix.

**Decisions (account holder, 2026-09-29 05:20 UTC):**
1. Events mined from the val/private masked layer may be used **for evaluation only** (no prediction reads
   post-origin data). Tool: `tfb/t2_onset_shift.py`.
2. Same-day pre-origin masked-layer features (the early/episode features in P2) are allowed: they come before the
   origin, so they are not a leak.

### Onset under scenario shift (`tfb/t2_onset_shift.py`, `tfb/t2_onset_shift_cal.py`; 2026-09-29)

Onset events mined from the masked layer of each split (945 train, 93 validation, 103 private), with inputs as at
inference (masked-layer history T−60…T−5, origin row missing). Train is scored 2-fold by day parity, and
val/private with the full-train model. Mean over panels:

| Onset | train (in scenario) | validation | private |
|---|---|---|---|
| static set | 0.695 | 0.657 | 0.719 |
| **v9b model** | **0.893** | **0.765** | **0.766** |
| no link-id feature | 0.863 | 0.697 | 0.756 |
| no link-id, no cluster-id | 0.862 | 0.704 | 0.741 |
| no tod/dow | 0.828 | 0.763 | 0.804 |

* **The onset model loses ≈ 0.13 out of scenario, in validation AND private.** The static set does not.
  This reproduces the online onset (≈ 0.72, +0.14 over static) and explains the onset local/online gap. Private is
  affected as much as public.
* Why: 35 min before onset the links that will queue still run at free flow (speed ratio ≈ 1.66 = 1/0.6 in all
  splits). The history says little about *where* the queue forms, so the model leans on link identity and time of
  day, and onset timing per bottleneck shifts with the scenario (D12_I5_N onsets at 12.4 h in train vs 10.8 h in private).
* Paired out-of-scenario deltas vs v9b: no_tod +0.016 ± 0.014 (val −0.008, pri +0.037); no_id_tod −0.008 ± 0.017;
  no_link −0.033 ± 0.010. Link identity still helps. Dropping tod is not significant; not adopted yet.
* Decoding calibration (temperature τ, shrinkage to cluster base rates): private improves (0.766 → 0.79), validation
  worsens (→ 0.70–0.75). The splits disagree, and pooled everything is ≈ 0.764. v9b decoding stays.
* Ensemble (`tfb/t2_onset_shift_ens.py`: pooled MC samples, v9b weight 1−w, no-tod weight w), paired out-of-scenario
  deltas vs v9b: w=0.3 +0.006 ± 0.007; **w=0.5 +0.022 ± 0.012** (private +0.046, validation −0.005); w=0.7 +0.022 ± 0.014.
  In scenario −0.016. The whole gain comes from private; validation is flat. Choosing it by private events would
  tune the April model on April data, so it stays a candidate, not adopted. Decision after the ongoing comparison.

### Ongoing under scenario shift (`tfb/t2_ongoing_shift.py`; 1,181 val + 1,121 private masked-layer windows)

| Ongoing IoU | private | validation | paired Δ vs v9b |
|---|---|---|---|
| v9b (old LightGBM + old 4-seed CNN) | 0.8749 | 0.8733 | – |
| **P1 (fixed LightGBM + old CNN)** | **0.8813** | **0.8772** | **+0.0046 ± 0.0012** (pri +0.0056, val +0.0037) |
| P2 (fixed LightGBM + early CNN, 2 seeds) | 0.8701 | 0.8749 | −0.0023 ± 0.0020 (pri −0.0058) |

The fixed LightGBM gains +0.005 out of scenario (vs +0.02–0.03 in scenario). The early-channel 2-seed CNN is
worse out of scenario (−0.007, private −0.014). **P2 leaves the final candidates; P1 replaces it.**

### The online v9b onset also suffers from the origin-row mismatch (found 2026-09-29 ~11:30 UTC)

The online v9b onset was trained on `t2_onset_withT.parquet` (T row present). Out of scenario, with the realistic
input (T row missing), it scores **0.721 (val) / 0.722 (pri)**, which matches the online onset estimate (≈ 0.72). The
refit on the visible history (`t2_onset.parquet`) scores 0.765 / 0.766: **paired +0.038 ± 0.012 (val +0.032,
pri +0.044), in scenario +0.055**. The earlier "+0.001" estimate compared each model on its own input and was
wrong. The pooled ensemble with the no-tod variant (w=0.5) vs online v9b: +0.060 ± 0.017 (val +0.027, pri +0.089).
Production check: `tfb/t2_v10.py` with the old frame and w=0 reproduces v9b's onset exactly (0 cells differ).

### Pre-registered submissions for 2026-09-30 UTC

**V10a: breakthrough/probe, "the origin-row fix on onset + P1 ongoing"** (`sub_v10a.zip`, `queue_models12_v10a.csv`)
* Change vs v9b: onset = visible-history refit (52 onset cells differ); ongoing = P1 (279 cells). T1 and T4 are unchanged.
* Expected Δ vs v9b, from the validation masked-layer events (public = validation): onset +0.032 → +0.0048 total,
  ongoing +0.0037 → +0.0006; **public ≈ +0.005** (noise sd ≈ 0.004–0.006 for an onset change on 40 windows). Private
  ≈ +0.0065 + 0.0008 ≈ +0.007.
* Purpose: goal criterion B (T-row-fixed version ≥ +0.005 online vs v9b), and confirmation that the out-of-scenario
  evaluator predicts online.
* Actual: **0.87446** (Δ vs v9b **+0.0034**, expected +0.005; −0.0016 deviation, within the ≈ 0.005 noise). **New best.**
  The direction confirms the onset origin-row fix. Criterion B (≥ +0.005) is not formally met.

**V10b: probe, "onset ensemble with the no-tod variant"** (`sub_v10b.zip`, `queue_models13_v10b.csv`)
* Change vs V10a: onset = pooled ensemble (w=0.5), 75 onset cells differ from v9b.
* Expected Δ vs v9b: public ≈ 0.15·0.027 + 0.0006 ≈ **+0.0046** (≈ V10a − 0.0008); private ≈ +0.014.
* Purpose: check that the ensemble costs nothing on public (validation says ≈ −0.005 onset vs V10a). The private gain
  cannot be seen on public; if V10b ≥ V10a − 0.004, the ensemble becomes a final candidate on its private evidence.
* Actual: **0.87109** (Δ vs V10a **−0.0034**, expected −0.0008). This is inside the pre-registered threshold (≥ V10a − 0.004),
  so the ensemble stays a final candidate. On validation both signals say it is slightly worse (masked-layer events −0.005,
  online −0.023 onset IoU); its case rests on private-period events (+0.046), where public cannot confirm it.
* The other three 09-30 submissions stay unassigned until V10a/V10b come back.

### 2026-09-30 status (00:10 UTC)

* Public best: **V10a 0.87446** (onset origin-row fix + P1 ongoing).
* Final-selection plan (2 picks): **V10a** (best public, and better than v9b on every out-of-scenario test) + **V10b** (highest
  private expected value, +0.007 over V10a by private-period events, −0.0034 on public). v9b and P2 drop out: V10a dominates v9b,
  and P2's CNN is worse out of scenario.
* The remaining 3 submissions today: none is ready with a positive expected Δ. Next candidate is the ongoing CNN
  retrained on the visible history **without** the early channels, 4 seeds (the old CNN is still trained with the T row).
  It is judged by `t2_ongoing_shift.py` first; a submission only follows if it beats the old CNN out of scenario. Otherwise
  the slots stay unused (reason: no candidate with a positive expected Δ).

### Onset/ongoing diagnostics (2026-09-30 02:00 UTC)

* Onset loss out of scenario is almost all **cluster choice**: when the predicted clusters equal the true ones, IoU is
  0.94–0.98 in every split; when they differ, 0.14–0.16. Wrong-cluster rate is 12% in scenario vs ≈ 20% out of scenario.
  Candidate coverage is not the issue (true links outside the train candidates: private 7%, validation 2%).
* Recent-days scenario prior (`tfb/t2_onset_recent.py`: reweight cluster probabilities by the cluster's queue-day rate on
  earlier days of the same split vs train; causal): **no effect**. The cluster probabilities are saturated (0.99 / ≈ 0), and
  in the first week, where the windows sit, the recent rates barely differ from train on the panels with events.
  Rejected.
* Ongoing candidate coverage is fine out of scenario (0.6% of true cells outside the candidates). The out-of-scenario ongoing
  level (0.877) is not comparable with the in-scenario 0.914: the masked-layer inputs miss ≈ 43% of cells vs ≈ 16% online.
* Container restart at 01:39 UTC killed the visible-history CNN training (seed 0, epoch 19). Relaunched 01:40.
* Process lesson (2026-09-30): the container is reclaimed when the session idles, which killed the CNN training twice
  (01:39 and ≈ 02:10 UTC). Long jobs now run while the session stays active (≤ 10-min waits), and each seed is saved as
  soon as it finishes. The visible-history CNN is trained with 2 seeds first and compared with 2 old seeds (fair
  seed count) before any more seeds.

### Visible-history CNN and ongoing decoding, out of scenario (2026-09-30 05:30 UTC; 1,181 val + 1,121 pri windows)

| Paired Δ | mean ± se | private | validation |
|---|---|---|---|
| CNN visible history (2 seeds) − old CNN (same 2 seeds) | −0.0027 ± 0.0021 | −0.0057 | +0.0001 |
| fixed LGB + vis CNN − P1 (both 2 CNN seeds) | −0.0003 ± 0.0016 | −0.0010 | +0.0004 |
| P1 threshold 0.45 − 0.5 | +0.0017 ± 0.0008 | +0.0015 | +0.0019 |
| P1 threshold 0.55 − 0.5 | −0.0025 ± 0.0008 | | |
| P1 blend w 0.3 / 0.7 − 0.5 | −0.0045 / −0.0039 | | |

* The origin-row fix does not help the CNN out of scenario, so **no V10c**. P1 (fixed LightGBM + old 4-seed CNN, w 0.5,
  threshold 0.5) stays.
* Threshold 0.45 is consistent in both splits but worth ≈ +0.0003 total, and the masked-layer inputs are much sparser than
  online ones (it may be an input artefact). Not adopted.
* **The remaining 3 submissions of 2026-09-30 stay unused**: no candidate with a positive expected Δ.

### Onset cluster calibration (2026-09-30 06:30 UTC): rejected

Rule set beforehand: adopt only if both splits ≥ 0 and the pooled gain > 2 se. Cluster-only temperature τ = 1.5/2/3:
out-of-scenario −0.003/−0.009/−0.017 (validation −0.008…−0.038, private +0.0016). Clipping pc to [ε, 1−ε]: ε ≤ 0.1 changes
nothing; ε = 0.2 gives −0.004. Expected-IoU decoding is not hurt by overconfident cluster probabilities: flattening only
enlarges the predicted set.

### Status and plan (2026-09-30)

* Every component is at its ceiling under the compliance rules. T1, physics and T4 were already closed. Ongoing = P1, the
  optimum out of scenario (CNN refit, threshold and blend weight all checked). Onset = origin-row-fixed refit (V10a);
  ensemble, recent-days prior and calibration were tried.
* Kaggle picks the top-2 public submissions if none are selected by hand: those are V10a (0.87446) and V10b (0.87109),
  which is exactly the planned final pair. Re-check on ≈ 2026-10-30.
* New submissions only for changes with out-of-scenario evidence (paired, both splits).

## First-principles audit against the scoring spec (`tfb_ref/docs/SCORING_SPEC.md`, 2026-09-30)

Online component estimates and headroom (weight × (1 − score)): S_state ≈ 0.94 → 0.021; **S_queue ≈ 0.80 → 0.060**;
S_physics ≈ 0.70 → 0.045; S_ODME ≈ 0.85–0.90 → 0.025.
* T1: S_state = 0.54·(1 − RMSE_v/25) + 0.46·(1 − RMSE_q,lane/600). At the noise floor (σ_v ≈ 1.7, σ_q ≈ 30/lane) that is
  0.54·0.932 + 0.46·0.95 ≈ 0.94, where we are. Closed by arithmetic.
* T3: S_physics = S_FD/3 + 2·S_LWR/3, computed on our own T1 cells (k = q/v, N = kL) against organizer boundary fluxes that
  are projected onto the noisy observations. The conservation signal is 0.75% of N, while the measurement noise in N is
  ≈ 2–3%, so at masked cells dN is mostly unpredictable noise. Occupancy is masked at the same cells (checked: 0% present),
  so there is no direct density measurement to exploit. The floor stands.
* T4: the organizer reference is `nnls` ridge (λ = 0.05) over the split's prior and counts (`build_task4_odme_artifacts.py`),
  which is what we submit. S_od (45%) compares with withheld path flows. Closed unless the truth generator differs.
* T2: the truth is taken from the **noise-free underlying state**. Tested: denoised (3-slot median) vs noisy labels differ
  on 0.1% of queued ongoing cells; IoU at thresholds 0.3–0.6 moves by ≤ 0.0002. Label noise is not a factor. Rejected.
* So the headroom is T2, and on T2 it is the scenario shift. Onset: in scenario 0.893 vs out 0.765. Ongoing: being
  measured (`tfb/t2_ongoing_idshift.py`: same masked input in and out of scenario, and with vs without the
  identity/time features).
* Ongoing, same masked-layer input in and out of scenario (`t2_ongoing_idshift.py`, 30% subsample LightGBM): persistence
  0.476 / 0.475 / 0.484 (train / val / pri, equally hard windows); model **0.898 in scenario vs 0.861 / 0.869 out**, so
  ≈ 0.033 of scenario shift. Dropping the identity/time features (link, lpos, tod, dow; also pid, is_bneck) costs 0.02 in
  scenario and −0.003 ± 0.002 out of scenario: the shift is in the dynamics (capacities), not in memorised identities.
* Onset: the models had **no ramp features**, although bottleneck activation is mainline + on-ramp demand vs capacity and
  ramp flows are published. Testing (`tfb/t2_onset_ramp.py`): on-ramp flow up to 2 links upstream, off-ramp flow up to 2
  downstream, demand ratio (mainline + on-ramp) / capacity, all from T−60…T−5.

### Onset ramp-demand features: adopted by the pre-set rule (2026-09-30)

`tfb/t2_onset_ramp.py`, onset IoU panel-mean (in scenario = train 2-fold; out = val/private masked-layer events):

| | train | validation | private |
|---|---|---|---|
| v9b structure (visible history) | 0.893 | 0.7645 | 0.766 |
| + ramp-demand features | 0.889 | **0.8248** | **0.7871** |

Panel-weighted out-of-scenario Δ **+0.041 ± 0.016** (bootstrap, P(>0) 0.997), both splits positive, in scenario −0.004.
The gain is D7_I405_S (+0.53 val, +0.22 pri), the panel whose bottlenecks switch in opposite directions in March and April.
D7_I210_W loses (−0.08 / −0.03); six panels are unchanged. Few events on D7_I405_S (5 val, 7 pri): the size is uncertain,
the direction is not.

**V11: probe/breakthrough, "ramp demand identifies the active bottleneck in a new scenario"** (`sub_v11.zip`,
`queue_models14_v11.csv`, built with `TFB_ONSET_RAMP=1 python -m tfb.t2_v10 0 …`)
* Change vs V10a: onset models get the 6 ramp features (ramp flows T−60…T−5 from the published ramp layer, last visible
  mainline flow from the window history). 51 cells differ, mostly D7_I405_S validation windows 1, 2, 4, 5. Ongoing, T1, T4
  are unchanged.
* Expected Δ vs V10a: public ≈ 0.15 × 0.06 ≈ **+0.009 (uncertain, ±0.01: only a few windows change)**; private ≈ +0.003.
* Purpose: probe of the ramp-demand hypothesis. If public ≥ V10a, V11 replaces V10b as the second final pick (or V10a).
* Actual: **0.86531** (Δ vs V10a **−0.0092**, expected +0.009; deviation −0.018 ≈ −0.12 onset IoU panel-mean). **Rejected.**
  Not a pipeline bug: ramp data in the window histories are as available as in training (≈ 20% missing everywhere).
  On the real validation windows (D7_I405_S 1, 2, 4, 5) the ramp model picks the wrong cluster, so the mined-event gain on
  that panel (5 val events) was small-sample luck.
* **Rule tightened:** a pooled out-of-scenario gain counts only if it is not carried by one panel: at least half of the
  changed panels must improve in both splits, and the gain must survive dropping the best panel.

### Ongoing kinematic-wave (LWR) tail features (`tfb/t2_ongoing_lwr.py`, 2026-09-30)

Shock speed at the nearest queue run's tail, w = (q_up − q_queue)/(k_up − k_queue) (k = q/v at T−5; median +2 km/h, 10–90%
−14…+18), and each cell's km distance to the tail extrapolated k steps ahead. Same 30% subsample protocol as the id-shift
test. Ongoing IoU: train 0.8976 → 0.8972, val 0.8610 → 0.8634, pri 0.8694 → 0.8707. Out of scenario **+0.0020 ± 0.0011
(1.8 se)**; 4 of 8 panels improve in both splits, 2 lose in both (I210 E/W). Below the 2-se bar, and worth ≈ +0.0003
total. **Not adopted, not submitted.** The model's empirical tail features (d_tail, run_growth6) already carry most of the
kinematics.

**2026-09-30: the last 2 submissions stay unused.** No candidate has a credible positive expected Δ above the public noise.
Public best stays V10a (0.87446); final picks V10a + V10b.

## Round 1 under GOAL.md (2026-10-01): first-principles re-audit, two adopted changes

Context: leaderboard on 2026-10-01: we are #24 at 0.87446 (V10a); #1 0.92406, #20 ≈ 0.877. The `out/` directory and every
trained model were lost with the old container, and Kaggle does not serve past submission files, so the pipeline is being
rebuilt (recipes now in code: `tfb/t1_prod.py`, `tfb/t2_build_parts.py`, `tfb/t2_prod.py`).

### Generator facts (measured on train, D7_I10_E / D12_I5_S / D7_I405_S)
* The fundamental diagram is two flat plateaus: free flow at a per-link speed `plat` (v/vf ≈ 1.00 at every flow level, p10–p90
  ±1.5%), queue at ≈ 0.35·vf. There is no speed precursor of breakdown, and flow/capacity at the queued cluster before onset
  is average (0.5–0.7, the top-flow links never queue). Queues switch on as whole blocks (links 4–10 at once).
  Consequence: onset location is not identifiable from the history; it is a prior (tod, panel, cluster) that the scenario
  seed changes. This explains the out-of-scenario onset loss; no physics feature can fix it.
* Free-flow speed noise = observed − plateau has sd 0.8–1.75 km/h per panel, almost white in time (lag-1 0.06), but with a
  **corridor-wide common mode**: corr ≈ 0.3 between links 1 or 30 apart; the common factor has sd 0.46–1.09 km/h. Flow
  noise has no usable common mode (second-difference residuals, corr ≈ 0.1 at distance 10).

### Adopted 1 — T1 common-mode features (`features.common_factors`, `TFB_CF=1`)
cf_sp = mean over links observed at the same slot of (speed − plateau) on free-flowing cells (also relative and ±5-link
local versions), flow analogues, and `plat`. The target cell is masked, so it never enters its own estimate.
A/B on the same frames (train days d%4≠0 → held-out d%4==0, 1M cells, 3000 rounds):

| | S_state | speed RMSE | true free-flow speed RMSE | flow/lane RMSE |
|---|---|---|---|---|
| V10a features | 0.94260 | 1.678 | 1.250 | 30.80 |
| + common mode | **0.94457** | **1.588** | **1.094** (ideal plateau+cf: 1.110) | 30.92 |

9/10 panels improve. Flow does not improve → production uses cf features in the speed model only. The free-flow error is now
at the floor; the remaining speed MSE is 54% from the 4.6% transition/queued cells. S_LWR proxy (Σ|N error| at targets):
−0.44% (N error is flow-dominated). A physics anchor (residual over plat + cf instead of temporal interpolation) gave no
further gain (1.5709 vs 1.5773). **Expected Δtotal ≈ +0.001.**

### Adopted 2 — T2 ongoing uses the published origin row T (`t2_events.visible_t`, frame `t2_ongoing_parts_vt`)
The 2026-09-29 diagnosis "the origin row T is never published" was only half right: `window_history` stops at T−5, but the
**masked layer publishes row T with ≈58% of links observed** (only T1 targets are blank; the blackout is T+5…T+90), in
validation and private alike (checked on all 160 windows). The spec gives "60 minutes of history through the forecast origin
T", so row T is in the information set. Models are now trained with row T from the train masked layer (same ≈58% pattern)
and fed the masked-layer row T at inference.
Paired A/B, ongoing LightGBM (35% of windows, 600 rounds; `tfb/t2_vt_ab.py`):

| | persistence | vis (V10a, no row T) | vt (masked row T) | Δ (paired) |
|---|---|---|---|---|
| train, 2-fold, selector filter | 0.606 | 0.8942 | **0.9059** | +0.0102 ± 0.0008 |
| validation, out of scenario | 0.484 | 0.8584 | **0.8816** | +0.0231 ± 0.0025 |
| private, out of scenario | 0.481 | 0.8628 | **0.8817** | +0.0211 ± 0.0026 |

7/8 panels improve in both splits (D7_I10_W private −0.011, n small); survives dropping the best panel. **Adopted.**
Onset with row T (`tfb/t2_onset_vt_ab.py`): train +0.004 (n.s.), validation −0.0003, private −0.015 → **rejected** (row T is
35 minutes before the queue and still free-flowing; no information).

### Rejected / closed this round
* Physics-based onset location (FD-implied capacity, demand/capacity ratios): no precursor exists in this generator (above).
* Window history carries nothing beyond the masked layer (identical values where both exist).
* Earlier days of the same split as onset evidence (`tfb/t2_onset_prior_days.py`, causal: days strictly before the
  event): 23 mined events on days 2–7 (the window period), 7 with a wrong top cluster. Both the true and the predicted
  cluster queued on almost every earlier day (day rates 0.85–1.0 in every split), so the evidence separates only 1 of the
  7. The scenario shift is in *when within the day* a cluster activates, not whether. **Closed.**

### Process lessons (round 1)
* **Never run torch next to LightGBM.** OpenMP spin-waiting made CNN training ≈ 30× slower (45–60 ms vs 1.5–6.8
  ms per window). Jobs now run strictly in sequence (`t2_prod lgb` → `cnn 0..3`).
* The memory cgroup kills at ≈ 10 GB total (two OOM kills while two big jobs overlapped). Big frames are read per panel
  and row-filtered in Arrow before pandas.
* Artifacts are mirrored to the private Kaggle dataset `baccano/tfb-steins-artifacts` (models + per-task files), so a
  container reclaim no longer loses compute. Rebuild: `TFB_CF=1 python -m tfb.t1_prod`, `python -m tfb.t4 l2`,
  `python -m tfb.t2_build_parts t2_ongoing_parts_vt 1 0.6`, `python -m tfb.t2_prod lgb|cnn <seed>|assemble <name>`.

### Adopted 3 — ramp congestion sensor for the T1 gap specialist (`features.ramp_ratio_features`, `tfb/t1_gap_prod.py`)
On-ramp inflow falls to 5–70% of its time-of-day profile while the mainline link it feeds is queued, and off-ramp flow
rises ≈ 10% (all three panels checked). The ramp layer is published at ≈ 80% coverage **inside** the 90-minute mainline
blackout after every Task 2 origin, so for Task 1 (offline reconstruction of the split; every published observation is an
input) it locates the queue inside the blackout. Task 2 never reads ramp data after T.
Gap-cell A/B (synthetic blackouts on train, d%4!=0 → d%4==0): speed RMSE 7.775 → **6.187**, flow/lane 78.80 → **73.56**,
Σ|N err| −22%, 9/10 panels. Production: 200k gap cells per panel, 3000/3500 rounds. Expected Δtotal ≈ +0.0015.

### Production ongoing, out of scenario (`tfb/t2_prod_shift.py`, same mined windows as the 09-30 P1 evaluation)

| private / validation | IoU |
|---|---|
| V10a ongoing (P1) | 0.8813 / 0.8772 |
| vt LightGBM alone (all windows) | 0.8863 / 0.8832 |
| vt CNN, mean of 2 seeds | 0.8822 / 0.8876 |
| **vt blend 0.5/0.5, threshold 0.5** | **0.8986 / 0.8980** |

Blend − LightGBM +0.013 ± 0.002. Threshold 0.45 is again +0.002 ± 0.001 in both splits; kept at 0.5 (pre-registered P1 blend).

### V12 (pre-registered 2026-10-01 22:25 UTC): breakthrough, "common mode + ramp-aware gap + origin row"
`sub_v12.zip` = T1 `state_cfr_gap.csv` (cf speed model, base flow model, ramp-aware gap specialist) + T2 `queue_v12.csv`
(onset = V10a recipe rebuilt; ongoing = vt LightGBM + 2 vt CNN seeds, 0.5/0.5, thr 0.5) + T4 `odme_l2.csv` (rebuilt).
* Expected Δ vs V10a: T1 common mode +0.0007, gap ramps +0.0009 (S_state) + physics ≈ +0.0008, ongoing +0.01–0.02 IoU →
  +0.0015–0.003. **Total ≈ +0.004–0.0055.** Public noise sd ≈ 0.004 for ongoing changes (40 windows). Rebuild noise: the
  onset and ongoing models are retrained from rebuilt frames (not bit-identical to V10a).
* Decision rule: ≥ V10a − 0.002 → V12 replaces V10b as a final pick (its private EV is the highest on every
  out-of-scenario test); below that → look for a rebuild bug before anything else.
* **Actual: 0.87324** (Δ vs V10a −0.0012; expected +0.004–0.0055; deviation ≈ −0.006). Inside the pre-registered rule
  (≥ V10a − 0.002), but the deviation is ≈ 1.5 sd, so it was investigated before anything else.
* Diagnosis 1 (found): **free-flow plateaus move per link in the new scenarios.** Median(speed − train plateau) per link has
  sd 0.8–1.7 km/h across links in validation and private (0 in train by construction). The cf speed model learned in train
  that plateau + common mode is an exact free-flow anchor; in validation/private that anchor is off by ≈ 1.3 km/h per
  link, comparable to the noise it removes. V10a's model did not see the plateau and followed each link's own
  neighbours, so it was immune. Fix: plateaus (and ramp profiles) are measured on the split's own published masked layer
  (`features.SCEN = "split"`; Task 1 is offline reconstruction, every published observation of the split is an input).
* New evaluator `tfb/t1_shift_eval.py` (out of scenario for Task 1): hide 3% of the observed eligible cells of
  validation/private (evaluation only), rebuild features, score base vs cf models under train vs split plateaus; and
  synthetic blackouts on validation/private for the gap specialist. **Lesson: every Task 1 feature that encodes a
  per-link constant measured on train must be checked out of scenario.**
* Container restarts (uptime reset at 22:03 and 22:33 UTC) killed running jobs twice; jobs are now restartable
  (models cached on disk) and waits are kept short.

### Out-of-scenario Task 1 results (`tfb/t1_shift_eval.py`, leak-free: split statistics recomputed with the hidden cells removed)
Hidden observed cells of validation/private (3%, ≈ 39k per panel-split), RMSE mean over panels, private / validation:

| main model | speed | flow/lane |
|---|---|---|
| V10a-type (no cf), train profile | 1.888 / 1.883 | 30.26 / 30.32 |
| cf, **train plateau (V12)** | 1.914 / 1.892 | 30.26 / 30.32 |
| **cf, split plateau** | **1.584 / 1.524** | 30.26 / 30.32 |
| cf, split plateau + split profile | 1.573 / 1.520 | 30.58 / 30.61 |

Synthetic blackouts on validation/private (gap specialist):

| gap model | speed | flow/lane |
|---|---|---|
| old (no ramps) | 6.32 / 6.50 | 68.3 / 68.9 |
| ramps, **train ramp profile (V12)** | **9.78 / 9.83** | **114 / 110** |
| ramps, split ramp profile | 5.58 / 5.72 | 65.7 / 65.1 |
| ramps, split ramp + split speed/flow profile | **4.92 / 5.10** | 65.7 / 65.3 |

Readings:
* V12's T1 was worse than V10a's on both counts (cf anchor off by the plateau shift; ramp ratios off by the demand shift
  → gap speed RMSE +50%). Estimated T1 loss ≈ −0.002, so V12's −0.0012 implies the T2 origin-row change gained
  ≈ +0.002–0.003 online, in line with its out-of-scenario evidence.
* The V10a model itself carries scenario error (train profiles); a split plateau removes most of it: speed −0.30/−0.36,
  ≈ +0.007 S_state ≈ **+0.0025 total**. Statistics measured on the split's own published masked layer are the general fix.
* V12b = V12 with T1 from `tfb/t1_predict.py` (cf speed with split plateau + train profiles; ramp gap specialist with the
  split's ramp and speed/flow profiles). T2/T4 unchanged from V12. Expected Δ vs V10a ≈ +0.0025 (T1 main) + 0.0005
  (gap) + T2 ≈ +0.002 + physics ≈ **+0.005**; vs V12 ≈ +0.006.
* **V12b actual: 0.87900** (2026-10-02 00:03 UTC; Δ vs V10a **+0.0045**, expected ≈ +0.005; vs V12 +0.0058, expected
  ≈ +0.006). New best; rank 18 (was 24). The out-of-scenario T1 evaluator predicted the online delta. Calibration rows:

| Change | Local predicted Δtotal | Online Δtotal |
|---|---|---|
| V10a → V12 (T1 cf/train plateau, ramp gap/train profile; T2 origin row) | +0.004–0.0055 (in-scenario evaluators) | −0.0012 |
| V12 → V12b (split-own plateau and profiles) | ≈ +0.006 (out-of-scenario T1 evaluator) | +0.0058 |

  Rule from now on: a Task 1 change is judged on `t1_shift_eval` (out of scenario), never on train-day holdouts alone.
* Final-selection plan: V12b (best public, best out-of-scenario evidence on every component) + one hedge (decided later).

## Round 2 (2026-10-02)
* Onset + same-day early evidence (masked layer before T−60: queued-slot count, slots since last queue, 3-hour queued
  fraction per link, cluster aggregates; `tfb/t2_onset_early_ab.py`), out of scenario: validation −0.0034 ± 0.0137,
  private +0.0049 ± 0.0104, train +0.0006; panels disagree (D12_I5_S −0.10 val / +0.04 pri). **Rejected.** With the
  prior-days check this closes the causal-scenario-evidence route for onset.
* Ongoing CNN seed 3 (out of scenario, same windows): blend with 3 seeds − 2 seeds = +0.0002 ± 0.0006 → the
  LightGBM + CNN blend is saturated; no more seeds (seed 4 stopped). Threshold 0.45 is +0.0022 ± 0.0008 again (third
  independent evaluation), but worth ≈ +0.0003 total and plausibly an artefact of the sparser masked-layer inputs used
  out of scenario; kept at 0.5.
* T1 main model + ramp congestion-sensor features (split ramp profile), out of scenario (`tfb/t1_ramp_main_ab.py`, 150k
  rows/panel, 3000 rounds, same hidden cells): speed private −0.015 / validation +0.009, flow ±0, 12/20 panels better.
  **Rejected** (ramps only help where the whole mainline is blank, i.e. the gap specialist).
* Side result (learning curve out of scenario): the same cf/no-ramp recipe at 150k rows/3000 rounds scores 1.652 / 1.583
  speed and 30.61 / 30.71 flow, against 1.584 / 1.524 and 30.26 / 30.32 for production (300k/4000). Out-of-scenario
  accuracy still grows with training size → round 3 tests 600k rows/panel, 5000 rounds.
* **Round 2 summary: no adoption** (CNN seeds saturated, onset early evidence n.s., T1 main ramps mixed).

## Round 3 (2026-10-02): T1 capacity, judged out of scenario (`tfb/t1_big.py`)
600k rows/panel, 5000 rounds (continuous training with a checkpoint callback; per-chunk `init_model` restarts re-predict
the whole frame and were 5× slower), same hidden cells as production:

| private / validation | speed | flow/lane |
|---|---|---|
| production (300k / 4000) | 1.584 / 1.524 | 30.26 / 30.32 |
| **big (600k / 5000)** | **1.542 / 1.490** | **30.03 / 30.03** |

Speed better on 16/20 panel-splits, flow on 19/20 → **adopted** (to be bundled: ≈ +0.0007 total alone).
Out of scenario, true free-flow cells (y and interpolation > 0.85 plateau): model 1.145 vs ideal plateau + common mode
1.152 → free flow is at the floor in the new scenarios too; the remaining speed MSE is queue-edge tails (cells queued in
time but not in space: RMSE 5–9; within-link queued speed sd ≈ 6 km/h in validation/private).
Round 4 = capacity round: T1 1.2M rows/panel (float32), gap specialist 400k rows/panel, ongoing LightGBM 2000 rounds;
bundle into V13 once the out-of-scenario evidence sums to ≥ +0.003.
* Disk allowance ≈ 39 GB: the first 1.2M-row frame write failed (ENOSPC). Deleted superseded files (V12 per-task and
  merged CSVs, finished A/B frames, the vis ongoing parts); keep ≥ 5 GB free before writing big frames.

## Round 4 (2026-10-02): capacity (`tfb/t1_cap.py`)
T1 main at 1.2M rows/panel, 5000 rounds (memory-lean: one target's columns, one float32 matrix), out of scenario:

| private / validation | speed | flow/lane |
|---|---|---|
| production V12b (300k / 4000) | 1.584 / 1.524 | 30.26 / 30.32 |
| 600k / 5000 | 1.542 / 1.490 | 30.03 / 30.03 |
| **1.2M / 5000** | **1.497 / 1.451** | **29.82 / 29.82** |

1.2M vs 600k: speed better on 16/20, flow on 19/20 → **adopted**. Cumulative vs V12b ≈ +0.0013 total (S_state
+0.0021 ≈ +0.0007; flow-driven N error −1.5% ≈ +0.0006 physics). The learning curve has not flattened (each doubling
≈ −0.04 speed, −0.2 flow out of scenario). The OOM on the first attempt came from building the flow matrix while the
speed booster still held its dataset; each target now trains in a fresh process with only its columns.
* Gap specialist at 400k cells/panel (≈ all synthetic-blackout cells; D7_I405_S has only 253k) vs 200k, out of scenario on
  synthetic blackouts: speed 4.921/5.057 vs 4.923/5.095, flow 65.63/65.45 vs 65.69/65.27, 12/20 and 10/20 panels →
  **not adopted** (no new information beyond 200k).
* Round 5: T1 main on up to 2.4M rows/panel (≈ 20M rows; `tfb/t1_all.py`, matrix filled column-wise into one float32
  array, frames per panel).
* Round 5, all data (≤ 2.4M rows/panel ≈ 20M rows), out of scenario: speed 1.486 / 1.420, flow 29.69 / 29.68 (vs 1.2M:
  1.497 / 1.451, 29.82 / 29.82); speed better on 16/20, flow on 19/20 → **adopted**. The curve is flattening (private
  speed −0.011). Cumulative vs V12b: speed −0.098 / −0.104, flow −0.56 / −0.64 → ≈ +0.0009 (S_state) + ≈ +0.0007
  (physics, flow-driven N error) ≈ **+0.0016** total.

### V13 candidate (built, not yet submitted)
V13 = V12b with the T1 main models replaced by the all-data models (`TFB_T1_MAIN=t1a python -m tfb.t1_predict v13`); T1 gap
specialist, T2 and T4 unchanged. Expected Δ vs V12b ≈ +0.0016 (out-of-scenario evaluator, which predicted V12→V12b to
within 0.0002). This is below the +0.003 submission rule, and no remaining lever adds enough to reach it (ongoing
threshold 0.45 ≈ +0.0003; more T1 data/rounds ≈ +0.0002; onset and T4 closed). Because only submitted entries can be
picked as finals, a final-candidate submission of V13 needs an explicit exception from the account holder.
* **Account holder decision (2026-10-02 ~10:00 UTC): submit V13 as a final candidate** (one-off exception to the +0.003
  rule; the rule stays for everything else). Pre-registered: Δ vs V12b ≈ +0.0016 (public noise ≈ ±0.003, so the public
  result is a bug check only: below V12b − 0.004 → look for a pipeline error).
* **V13 actual: 0.88107** (2026-10-02 09:56 UTC; Δ vs V12b **+0.0021**, expected +0.0016). New best; rank 17.

| Change | Local predicted Δtotal | Online Δtotal |
|---|---|---|
| V12b → V13 (T1 main on all train targets) | +0.0016 (out-of-scenario T1 evaluator) | +0.0021 |

Final-selection plan: **V13** (best public and best out-of-scenario evidence) + **V12b** (same pipeline with the
smaller T1 models; nearly identical, so the hedge is weak) — or V10a as a structurally different hedge. Decide by
2026-11-05; V13 dominates V10a on every out-of-scenario component, so the current pick is V13 + V12b.

## Round 6 (2026-10-02): split-own queued-speed plateau (`tfb/t1_cong_ab.py`)
Per-link queued-speed medians move between splits (sd 1–6 km/h vs train), so `plat_cong` / `cong_ratio` from the split's
own layer were tested like `plat` (paired, 300k rows/panel, out of scenario): speed 1.566 / 1.554 vs 1.558 / 1.550
without, queued-cell RMSE 5.06 / 4.70 vs 4.86 / 4.66, 6/20 panels better → **rejected** (the queue-edge tails are about
*whether* a cell is queued, not the queued level).

## Headroom assessment after round 6 (stop-condition check, GOAL.md)
| Component (weight) | Status | Remaining legit headroom (est.) |
|---|---|---|
| T1 state (0.35) | free flow at the noise floor in and out of scenario; capacity curve flattened (last doubling +0.0005, all train targets used); queued-edge tails resisted ramps and queued plateaus | ≤ +0.0005 |
| T3 physics (0.15) | follows T1 (flow-driven N error); floor otherwise | ≤ +0.0005 |
| T2 onset (0.15) | no causal signal for the cluster out of scenario: physics precursors, ramps, origin row, prior days, same-day early evidence all tested | ≈ 0 |
| T2 ongoing (0.15) | origin row adopted; LightGBM + CNN blend saturated (3 vs 2 seeds +0.0002); threshold 0.45 +0.0003 (possibly an input artefact) | ≈ +0.0003–0.0005 |
| T4 ODME (0.20) | L2 ≈ the organizers' ridge reference; 5 probes all worse | ≈ 0 |

Every remaining candidate is below the public noise (≈ ±0.003) and below the 0.002 per-component bar. Rounds without
adoption: 2 and 6; the capacity adoptions of rounds 3–5 have run out of data. **Conclusion: no obvious improvement left
under the compliance rules.** Public: V13 0.88107 (rank 17), up from V10a 0.87446 (rank 24) at the start of the goal.
The gap to #1 (0.924) is consistent with using observations after the post-horizon buffer for Task 2, which the rules and
the account holder's decision exclude.
