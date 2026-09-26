#!/usr/bin/env bash
# Reproduce every number in paper/paper.md. Roughly 1.5 h on 4 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
TR=data_arc2/data/training
EV=data_arc2/data/evaluation
export ARC_PRIOR=0  # uniform prior unless stated otherwise

# 1. Full system (uniform prior) on training and public evaluation.
[ -f out/full_train.json ] || python evaluate.py $TR --time 20 --out out/full_train.json | tail -1
[ -f out/full_eval.json ]  || python evaluate.py $EV --time 30 --out out/full_eval.json  | tail -1

# 2. Oracle expressibility vs. learnability.
python analysis_oracle.py $TR out/full_train.json > out/oracle_train.log
python analysis_oracle.py $EV out/full_eval.json  > out/oracle_eval.log
head -1 out/oracle_train.log out/oracle_eval.log

# 3. Ablations (training set).
for ab in copy keep context interp bma gate; do
  [ -f out/abl_${ab}.json ] || ARC_ABLATE=$ab python evaluate.py $TR --time 20 --out out/abl_${ab}.json | tail -1
done

# 4. Learned prior, 2-fold cross-validation on training tasks.
python learn_prior.py out/full_train_oracle.json out/foldB.txt > out/prior_fromB.json
python learn_prior.py out/full_train_oracle.json out/foldA.txt > out/prior_fromA.json
ARC_PRIOR=1 ARC_PRIOR_PATH=out/prior_fromB.json python evaluate.py $TR --ids out/foldA.txt --time 20 --out out/cv_A.json | tail -1
ARC_PRIOR=1 ARC_PRIOR_PATH=out/prior_fromA.json python evaluate.py $TR --ids out/foldB.txt --time 20 --out out/cv_B.json | tail -1

# 5. Final prior (all training tasks) for the Kaggle submission and eval set.
python learn_prior.py out/full_train_oracle.json > arcsolver/prior.json
ARC_PRIOR=1 python evaluate.py $EV --time 30 --out out/prior_eval.json | tail -1
python summarize.py
