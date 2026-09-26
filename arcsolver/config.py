"""Ablation switches, set via ARC_ABLATE="copy,keep,context,interp,bma,gate"."""
import os

ABLATE = set(filter(None, os.environ.get("ARC_ABLATE", "").split(",")))


def on(component):
    return component not in ABLATE
