"""Evaluate the solver on an ARC-AGI directory of task JSON files.

Usage: python evaluate.py data_arc2/data/evaluation [--limit N] [--time 30] [--jobs 4]
"""
import argparse
import json
import os
import signal
import time
from multiprocessing import Pool

from arcsolver.solver import solve_task


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def run_one(args):
    path, tl = args
    task = json.load(open(path))
    tid = os.path.basename(path)[:-5]
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(tl * 2) + 5)
    t0 = time.time()
    try:
        out, info = solve_task(task, time_limit=tl)
    except Timeout:
        return tid, 0.0, ["timeout"], time.time() - t0
    except Exception as e:
        return tid, 0.0, [f"error:{e!r}"], time.time() - t0
    finally:
        signal.alarm(0)
    score = 0.0
    for o, t in zip(out, task["test"]):
        if "output" in t and t["output"] in (o["attempt_1"], o["attempt_2"]):
            score += 1
    return tid, score / len(task["test"]), info, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time", type=float, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--ids", default="", help="file with task ids to restrict to")
    a = ap.parse_args()
    files = sorted(os.path.join(a.dir, f) for f in os.listdir(a.dir) if f.endswith(".json"))
    if a.ids:
        keep = set(open(a.ids).read().split())
        files = [f for f in files if os.path.basename(f)[:-5] in keep]
    if a.limit:
        files = files[: a.limit]
    t0 = time.time()
    with Pool(a.jobs) as p:
        res = p.map(run_one, [(f, a.time) for f in files], chunksize=1)
    total = sum(r[1] for r in res)
    for tid, s, info, dt in res:
        if s > 0:
            print(f"SOLVED {tid} {s:.2f} {dt:5.1f}s {info[0]}")
    print(f"score {total:.2f}/{len(res)} = {100 * total / len(res):.2f}%  wall {time.time() - t0:.0f}s")
    if a.out:
        json.dump({r[0]: {"score": r[1], "info": r[2], "time": r[3]} for r in res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
