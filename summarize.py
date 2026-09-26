"""Summarize experiment outputs in out/ into out/results.md (tables for the paper)."""
import json
import os


def score(path, ids=None):
    r = json.load(open(path))
    if ids is not None:
        r = {k: v for k, v in r.items() if k in ids}
    return sum(v["score"] for v in r.values()), len(r)


def ids(path):
    return set(open(path).read().split())


def main():
    L = ["# Results", ""]
    s, n = score("out/full_train.json")
    e, m = score("out/full_eval.json")
    L += ["| System | ARC-AGI-2 training (1000) | ARC-AGI-2 public eval (120) |", "|---|---|---|",
          f"| Full (uniform prior) | {s:.1f} ({100 * s / n:.1f}%) | {e:.1f} ({100 * e / m:.1f}%) |"]
    if os.path.exists("out/prior_eval.json"):
        pe, pm = score("out/prior_eval.json")
        L.append(f"| Full + learned prior | (see CV) | {pe:.1f} ({100 * pe / pm:.1f}%) |")
    L += ["", "## Ablations (training set)", "", "| Removed component | Score | Δ |", "|---|---|---|"]
    for ab in ("copy", "keep", "context", "interp", "bma", "gate"):
        p = f"out/abl_{ab}.json"
        if os.path.exists(p):
            a, _ = score(p)
            L.append(f"| {ab} | {a:.1f} | {a - s:+.1f} |")
    if os.path.exists("out/cv_A.json") and os.path.exists("out/cv_B.json"):
        A, B = ids("out/foldA.txt"), ids("out/foldB.txt")
        ua, _ = score("out/full_train.json", A)
        ub, _ = score("out/full_train.json", B)
        ca, _ = score("out/cv_A.json")
        cb, _ = score("out/cv_B.json")
        L += ["", "## Learned feature prior (2-fold CV)", "", "| Fold | Uniform prior | Learned prior (other fold) |",
              "|---|---|---|", f"| A (500) | {ua:.1f} | {ca:.1f} |", f"| B (500) | {ub:.1f} | {cb:.1f} |",
              f"| Total | {ua + ub:.1f} | {ca + cb:.1f} |"]
    for f in ("out/oracle_train.log", "out/oracle_eval.log"):
        if os.path.exists(f):
            L += ["", f"`{f}`: " + open(f).readline().strip()]
    open("out/results.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
