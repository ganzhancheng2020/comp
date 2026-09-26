# Results

| System | ARC-AGI-2 training (1000) | ARC-AGI-2 public eval (120) |
|---|---|---|
| Full (uniform prior) | 135.5 (13.6%) | 0.0 (0.0%) |
| Full + learned prior | (see CV) | 0.0 (0.0%) |

## Ablations (training set)

| Removed component | Score | Δ |
|---|---|---|
| copy | 128.5 | -7.0 |
| keep | 122.0 | -13.5 |
| context | 132.0 | -3.5 |
| interp | 133.5 | -2.0 |
| bma | 136.0 | +0.5 |
| gate | 135.5 | +0.0 |

## Learned feature prior (2-fold CV)

| Fold | Uniform prior | Learned prior (other fold) |
|---|---|---|
| A (500) | 67.5 | 67.5 |
| B (500) | 68.0 | 68.0 |
| Total | 135.5 | 135.5 |

`out/oracle_train.log`: tasks=1000 shape_reachable=817 expressible=187 solved=137 solved&expressible=126

`out/oracle_eval.log`: tasks=120 shape_reachable=89 expressible=2 solved=0 solved&expressible=0
