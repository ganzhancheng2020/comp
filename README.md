# comp — AI competition entries

See [`STRATEGY.md`](STRATEGY.md) for the competition survey, the chosen targets, and the manual
steps (registration and submission) that the account holder must do.

## ARC Prize 2026: MDL local-rule induction (CPU only)

| Path | What |
|---|---|
| `arcsolver/` | Solver: transform search (`dsl.py`), feature bank (`features.py`), MDL rule induction with interpolation and model averaging (`rules.py`), pipeline (`solver.py`) |
| `evaluate.py` | Score a directory of ARC task files: `python evaluate.py data_arc2/data/evaluation` |
| `analysis_oracle.py` | Expressibility vs. learnability analysis (the paper's main table) |
| `learn_prior.py` | Meta-learn the feature prior from oracle rules → `arcsolver/prior.json` |
| `kaggle/arc_mdl_rules.ipynb` | Self-contained offline Kaggle notebook (rebuild with `python kaggle/build_notebook.py`) |
| `paper/` | Paper Track write-up |

Setup:

```bash
pip install numpy scipy
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 data_arc2
python evaluate.py data_arc2/data/training --time 20
```

Ablations are switched with `ARC_ABLATE=copy,keep,context,interp,bma,gate`. `ARC_PRIOR=0`
disables the learned prior, and `ARC_PRIOR_PATH=...` selects a different prior file.

## Kaggle MCP

`.mcp.json` registers Kaggle's remote MCP server (`https://www.kaggle.com/mcp`) for Claude Code, and
`.claude/settings.json` pre-approves it. It authenticates with `Authorization: Bearer ${KAGGLE_KEY}`,
so set `KAGGLE_KEY` (and `KAGGLE_USERNAME`) in the environment. The server exposes competition
search, data download, `submit_to_competition`, `create_code_competition_submission` and
`search_competition_submissions`. You must still accept each competition's rules on kaggle.com
before submitting.

License: Apache-2.0.
