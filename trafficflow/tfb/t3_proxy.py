"""Local proxy for S_LWR on train: organizer fluxes reproduce the observations, so rhs ~ dN_true.

S_LWR ~ 1 - sum|dN_pred - dN_true| / sum|dN_true| over consecutive eligible cells of the same link.
(Ramp validity and topology-observability filters of the official scorer are ignored.)
"""
import numpy as np

from .data import network


def lwr_proxy(panel, z, days, S_pred, F_pred):
    """S_pred/F_pred: full (D, T, L) arrays equal to the truth except at target cells."""
    Lk = network(panel).length_km.to_numpy()
    el = z["elig"][days] == 1
    Nt = z["flow"][days] / np.maximum(z["speed"][days], 1) * Lk
    Np = F_pred[days] / np.maximum(S_pred[days], 1) * Lk
    ok = el[:, 1:] & el[:, :-1]
    dt, dp = np.diff(Nt, axis=1)[ok], np.diff(Np, axis=1)[ok]
    return 1 - np.abs(dp - dt).sum() / np.abs(dt).sum()
