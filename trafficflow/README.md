# TrafficFlowBench 2026 — team Steins

Code for our IEEE BigData Cup 2026 TrafficFlowBench entry. The selected final submission is V13 (public leaderboard
0.88107). The method, analysis and results are in [`report/report.pdf`](report/report.pdf); the full experiment log is
in [`PLAN.md`](PLAN.md). License: Apache-2.0 (repository root).

## Reproduce the final submission

```bash
# 1. data and the official toolkit, at the repository root
kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench -p data_tfb && unzip -q data_tfb/*.zip -d data_tfb
git clone https://github.com/jacky850/trafficflowbench-public tfb_ref
# 2. environment (Python 3.11, CPU only)
pip install -r trafficflow/requirements.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# 3. single entry script -> out/tfb/sub_v13.zip (upload file for Kaggle)
trafficflow/reproduce.sh
```

* Hardware: 4 CPU cores, 15 GB RAM, about 15 GB free disk, about 8–9 hours. No GPU, no external data, no pretrained
  models.
* Every step caches its result under `data_tfb/cache` (or `$TFB_CACHE`) and is skipped when the script is rerun, so an
  interrupted run resumes. Outputs go to `out/tfb` (or `$TFB_OUT`).
* Seeds are fixed (LightGBM defaults, torch seeds 0 and 1, NumPy generators seeded in every sampling step).

## Layout

| Path | What |
|---|---|
| `reproduce.sh` | single entry script (9 steps, listed inside) |
| `tfb/data.py` | dense (day, slot, link) caches of the release |
| `tfb/features.py` | Task 1 features: interpolation, neighbours, profiles, corridor common mode, ramp congestion sensor |
| `tfb/t1_all.py`, `tfb/t1_predict.py`, `tfb/t1_gap_prod.py` | Task 1 models (all train targets), prediction with split-own statistics, blackout specialist |
| `tfb/t2_onset*.py`, `tfb/t2_build_parts.py`, `tfb/t2_prod.py` | Task 2 onset and ongoing models (LightGBM + CNN), assembly |
| `tfb/t4.py` | Task 4 L2 projection of the weak prior onto the link counts |
| `tfb/make_sub.py` | merge into the Kaggle upload (official `merge_submissions.py`) |
| `tfb/t1_shift_eval.py`, `tfb/t2_*shift*.py` | out-of-scenario evaluators used for every decision |
| `report/` | report source, figures, PDF build |

## Compliance

* Task 2 uses released data with a timestamp at or before the forecast origin T only (organizer ruling, forum topic
  742068): the 60-minute window history, the masked-layer origin row at T, and earlier same-day/earlier-day masked
  layers.
* Task 1 is offline and uses any released observation of the split.
