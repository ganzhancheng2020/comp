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


def fd_score(panel, z, days, S_pred, F_pred, target_mask):
    """Official S_FD (per-lane triangular FD, normalised by submitted flow) over eligible cells of `days`,
    using predictions at target cells and truth elsewhere."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "tfb_ref" / "src"))
    from task3.score_task3 import network_parameters
    from .data import REL
    par = network_parameters(REL / "corridors" / panel).set_index("link_id").reindex(z["links"])
    el = z["elig"][days] == 1
    q = np.where(target_mask[days], F_pred[days], z["flow"][days])[el]
    v = np.where(target_mask[days], S_pred[days], z["speed"][days])[el]
    L = el.shape[-1]
    lane = np.broadcast_to(par.lanes.clip(lower=1).to_numpy(), el.shape)[el]
    vf = np.broadcast_to(par.free_speed_kmh.to_numpy(), el.shape)[el]
    cap = np.broadcast_to(par.capacity_vph.to_numpy(), el.shape)[el] / lane
    kc = np.broadcast_to(par.critical_density.to_numpy(), el.shape)[el] / lane
    kj = np.broadcast_to(par.k_jam.to_numpy(), el.shape)[el] / lane
    ql = q / lane
    kl = q / np.maximum(v, 1) / lane
    qfd = np.where(kl <= kc, vf * kl, cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - kl, 0))
    return max(0.0, 1 - np.abs(ql - qfd).sum() / (np.abs(ql).sum() + 1e-9))
