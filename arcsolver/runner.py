"""Parallel batch runner used by the Kaggle notebook."""
import json
import os
import signal
import time
from multiprocessing import Pool

from .solver import solve_task


def _alarm(*_):
    raise TimeoutError()


def _run(args):
    tid, task, per_task = args
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(per_task * 2) + 5)
    try:
        out, info = solve_task(task, time_limit=per_task)
    except BaseException as e:  # never lose a task: fall back to identity
        out, info = None, [repr(e)]
    finally:
        signal.alarm(0)
    if out is None:
        out = [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]]
    return tid, out, info


def solve_file(path, out_path="submission.json", per_task=60, jobs=None):
    tasks = json.load(open(path))
    t0 = time.time()
    with Pool(jobs or os.cpu_count()) as p:
        results = p.map(_run, [(tid, t, per_task) for tid, t in tasks.items()], chunksize=1)
    submission = {tid: out for tid, out, _ in results}
    json.dump(submission, open(out_path, "w"))
    print(f"wrote {out_path}: {len(submission)} tasks in {time.time() - t0:.0f}s")
    return submission, {tid: info for tid, _, info in results}
