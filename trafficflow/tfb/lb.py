"""Kaggle 提交与读分：python -m tfb.lb <zip> "<message>"  （提交后轮询直到出分并打印）。"""
from __future__ import annotations

import os
import subprocess
import sys
import time

COMP = "2026-ieee-big-data-traffic-flow-bench"
ENV = {**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_API_TOKEN", os.environ.get("KAGGLE_KEY", ""))}


def kaggle(*args) -> str:
    return subprocess.run(["kaggle", "competitions", *args], capture_output=True, text=True, env=ENV).stdout


def latest() -> str:
    lines = kaggle("submissions", COMP).splitlines()
    return lines[2] if len(lines) > 2 else ""


def submit(path: str, msg: str) -> str:
    print(kaggle("submit", COMP, "-f", path, "-m", msg).strip().splitlines()[-1], flush=True)
    for _ in range(120):
        row = latest()
        if "COMPLETE" in row or "ERROR" in row:
            return row
        time.sleep(15)
    return latest()


if __name__ == "__main__":
    print(submit(sys.argv[1], sys.argv[2]))
