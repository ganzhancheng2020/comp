# TrafficFlowBench 2026 — team Steins

Code for our IEEE BigData Cup 2026 TrafficFlowBench entry. The two selected final submissions are V14a (public
leaderboard 0.88147) and V13 (0.88107); V14a = V13 + Task 1 corrections trained on each split's own published cells.
The method, analysis and results are in [`report/report.pdf`](report/report.pdf); the full experiment log is in
[`PLAN.md`](PLAN.md). License: Apache-2.0 (repository root).

## Reproduce the final submissions

```bash
# 1. data and the official toolkit, at the repository root
kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench -p data_tfb && unzip -q data_tfb/*.zip -d data_tfb
git clone https://github.com/jacky850/trafficflowbench-public tfb_ref
# 2. environment (Python 3.11, CPU only)
pip install -r trafficflow/requirements.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# 3. single entry script -> out/tfb/sub_v13.zip and out/tfb/sub_v14a.zip (upload files for Kaggle)
trafficflow/reproduce.sh
```

* Hardware: 4 CPU cores, 15 GB RAM, about 10 GB free disk besides the release; measured about 6 hours of compute for
  V13 and 1.5 more hours for V14a. No GPU, no external data, no pretrained models.
* Every step caches its result under `data_tfb/cache` (or `$TFB_CACHE`) and is skipped when the script is rerun, so an
  interrupted run resumes. Outputs go to `out/tfb` (or `$TFB_OUT`).
* Seeds are fixed (LightGBM defaults, torch seeds 0 and 1, NumPy generators seeded in every sampling step).
* Verified: a clean run reproduced V13 with 14 of 174,000 Task 2 cells different, Task 4 identical, and Task 1 within
  0.07 km/h / 2 veh/h/lane RMSE of the submitted values, a worst-case score change of ≈ 0.003 (Task 1 ≤ 0.0011 plus
  physics; the award criterion is 1%, ≈ 0.009); on held-out cells the rebuilt models score the same as the
  submitted ones (Δ ≈ 0.00001). V14a's rebuild: Task 1 within 0.12 km/h / 2.3 veh/h/lane RMSE (worst case ≈ 0.004 in
  total), Task 2 and Task 4 as for V13. `python -m tfb.repro_check` performs the comparison.

## Layout

| Path | What |
|---|---|
| `reproduce.sh` | single entry script (9 steps, listed inside) |
| `tfb/data.py` | dense (day, slot, link) caches of the release |
| `tfb/features.py` | Task 1 features: interpolation, neighbours, profiles, corridor common mode, ramp congestion sensor |
| `tfb/t1_all.py`, `tfb/t1_predict.py`, `tfb/t1_gap_prod.py` | Task 1 models (all train targets), prediction with split-own statistics, blackout specialist |
| `tfb/t1_tx_prod.py`, `tfb/t1_txgap_prod.py` | V14a: transductive corrections (main rows, blackout rows) trained on the split's own published cells |
| `tfb/t2_onset*.py`, `tfb/t2_build_parts.py`, `tfb/t2_prod.py` | Task 2 onset and ongoing models (LightGBM + CNN), assembly |
| `tfb/t4.py` | Task 4 L2 projection of the weak prior onto the link counts |
| `tfb/make_sub.py` | merge into the Kaggle upload (official `merge_submissions.py`) |
| `tfb/t1_shift_eval.py`, `tfb/t2_*shift*.py` | out-of-scenario evaluators used for every decision |
| `report/` | report source, figures, PDF build |

## Compliance

* Task 2 uses released data with a timestamp at or before the forecast origin T only (organizer ruling, forum topic
  742068): the 60-minute window history, the masked-layer origin row at T, and earlier same-day/earlier-day masked
  layers.
* Task 1 is offline and uses any released observation of the split (V14a also trains its correction on them).
