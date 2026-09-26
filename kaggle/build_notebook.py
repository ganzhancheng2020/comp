"""Build a self-contained Kaggle notebook embedding the arcsolver package.

Usage: python kaggle/build_notebook.py  ->  kaggle/arc_mdl_rules.ipynb
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "arcsolver")

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})


md("# MDL local-rule induction for ARC-AGI-2 (CPU only)\n\n"
   "Transform search + minimum-description-length rule induction with Bayesian model averaging. "
   "No internet, no GPU, no pretrained weights. Source: see the linked paper / GitHub repository.")

files = {}
for fn in sorted(os.listdir(PKG)):
    if fn.endswith(".py"):
        files[fn] = open(os.path.join(PKG, fn)).read()
code("import os, json\nos.makedirs('arcsolver', exist_ok=True)\nFILES = " + json.dumps(files) +
     "\nfor k, v in FILES.items():\n    open(os.path.join('arcsolver', k), 'w').write(v)\nprint(sorted(FILES))")

code('''import glob, json
from arcsolver.runner import solve_file

cands = glob.glob('/kaggle/input/**/arc-agi_test_challenges.json', recursive=True)
TEST = cands[0] if cands else 'arc-agi_test_challenges.json'
print('input:', TEST)
submission, info = solve_file(TEST, 'submission.json', per_task=60)
''')

code('''# Local sanity check when the public evaluation solutions are available.
sol = glob.glob('/kaggle/input/**/arc-agi_evaluation_solutions.json', recursive=True)
if sol and 'evaluation' in TEST:
    sols = json.load(open(sol[0]))
    score = 0
    for tid, outs in submission.items():
        s = sum(o['attempt_1'] == g or o['attempt_2'] == g for o, g in zip(outs, sols[tid]))
        score += s / len(outs)
    print('score', score, '/', len(submission))
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "kaggle", "arc_mdl_rules.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
