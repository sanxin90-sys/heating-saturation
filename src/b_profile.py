"""
b_profile.py
============

How precisely is the saturation scale b identified at each tower?

Added while drafting the manuscript (2026-10-29). Descriptive, not a
pre-registered test: the experiments EXP-009..016 chose b from the coarse grid
B_GRID = {0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02}, so "the same
b at five towers" could partly reflect the grid. Here b is profiled on a fine
log grid with exactly the data, rows and weak form of those experiments:

    dT/dt = c - lam*T + A*(1 - exp(-b*SW)) + harmonics (K = 3)   (sat+harm)

For each site and each b, the other coefficients are refitted by least squares
on the weak-form rows of the training years; b_hat minimises the weak residual.
Uncertainty: 500 daily-block bootstrap resamples of the training rows (b_hat
recomputed on each). The same profile is computed on the test years alone, as
an independent estimate. Also reported: the half-saturation irradiance
SW_1/2 = ln 2 / b and the ratio of the marginal response at 800 W m-2 to the
response at SW -> 0, exp(-800 b).

Usage: python src/b_profile.py      (writes results/b_profile.csv and b_profile_curves.csv)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_weak_loyo as W  # noqa: E402
import sindy_wind_forms as F  # noqa: E402
import sindy_h_and_npw  # noqa: E402,F401  (registers Npw in F.SITES)

B_FINE = np.round(np.geomspace(1e-4, 3e-2, 49), 6)
N_BOOT = 500
SITES = ["Jaru", "K34", "Sa1", "Ma3", "Npw"]


def site_rows(name):
    d = F.load_site(name)
    if name == "Npw":  # as in EXP-016: sat+harm and relax+harm do not use WS
        d = d.assign(WS=d.WS.fillna(1.0))
    rows = W.weak_rows(d, F.M)
    ry = np.array([r[0] for r in rows]); rd = np.array([r[2] for r in rows])
    return d, rows, ry, rd


def profile(Gs, y, sel, rd, rng):
    """SSE(b) on rows sel, b_hat and a daily-block bootstrap of b_hat."""
    def sse_curve(idx):
        out = []
        for G in Gs:
            xi = np.linalg.lstsq(G[idx], y[idx], rcond=None)[0]
            out.append(np.sum((y[idx] - G[idx] @ xi) ** 2))
        return np.array(out)

    base = np.flatnonzero(sel)
    sse = sse_curve(base)
    days = rd[base]
    ud = np.unique(days)
    pos = {d: base[days == d] for d in ud}
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([pos[d] for d in rng.choice(ud, len(ud))])
        boots.append(B_FINE[np.argmin(sse_curve(idx))])
    return sse, B_FINE[np.argmin(sse)], np.percentile(boots, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    res, curves = [], []
    for name in SITES:
        cfg = F.SITES[name]
        d, rows, ry, rd = site_rows(name)
        y = F.weak_G(d, rows, F.cols(d, "relax+harm"))[1]
        Gs = [F.weak_G(d, rows, F.cols(d, "sat+harm", b))[0] for b in B_FINE]
        Gr = F.weak_G(d, rows, F.cols(d, "relax+harm"))[0]
        for part, years in (("train", cfg["train"]), ("test", cfg["test"])):
            sel = np.isin(ry, years)
            sse, bh, (lo, hi) = profile(Gs, y, sel, rd, rng)
            xr = np.linalg.lstsq(Gr[sel], y[sel], rcond=None)[0]
            sse_lin = np.sum((y[sel] - Gr[sel] @ xr) ** 2)
            xi = np.linalg.lstsq(Gs[int(np.argmin(sse))][sel], y[sel], rcond=None)[0]
            res.append({"site": name, "part": part, "years": " ".join(map(str, years)), "n_rows": int(sel.sum()),
                        "b_hat": bh, "b_lo": lo, "b_hi": hi, "SW_half": np.log(2) / bh,
                        "slope_ratio_800": np.exp(-800 * bh), "A": xi[2], "lam": -xi[1],
                        "sse_min_over_sse_linear": sse.min() / sse_lin})
            curves += [{"site": name, "part": part, "b": b, "sse_rel": s / sse.min()} for b, s in zip(B_FINE, sse)]
        print(pd.DataFrame(res)[lambda x: x.site == name].round(5).to_string(index=False), flush=True)
    res = pd.DataFrame(res)
    res.to_csv(W.OUT / "b_profile.csv", index=False)
    pd.DataFrame(curves).to_csv(W.OUT / "b_profile_curves.csv", index=False)
    print("\n", res.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
