"""
h_shape.py
==========

Is measured midday sensible heat flux H concave in SW at the five towers?

Added while drafting the manuscript (2026-10-29); descriptive, not pre-registered.
EXP-014 C fitted a(1 - exp(-b SW)) + c to midday H at four towers and found
b ~ 0. Here, for all five towers (same QC = 0 data, 10-14 h local time, all
years used in the experiments): an OLS quadratic H = c0 + c1 SW + c2 SW^2 with a
daily-block bootstrap CI (500) of c2 (c2 < 0 with CI below zero = concave), and
the OLS slopes dH/dSW below and above 600 W m-2.

Usage: python src/h_shape.py RAW    (RAW = folder with ma2/, Sa1/, Ma3/, Npw/ FLUXMET_HH csvs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_wind_forms as F  # noqa: E402
import sindy_h_and_npw as HN  # noqa: E402

SITES = ["Jaru", "K34", "Sa1", "Ma3", "Npw"]


def main(raw):
    rng = np.random.default_rng(0)
    out = []
    for s in SITES:
        d = HN.with_H(s, F.load_site(s), raw)
        m = d[d.hour.between(10, 14) & d.SW.notna() & d.H.notna()]
        x, y, days = m.SW.to_numpy(float), m.H.to_numpy(float), m.day.to_numpy()
        X = np.column_stack([np.ones_like(x), x, x ** 2])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        ud = np.unique(days); pos = {u: np.flatnonzero(days == u) for u in ud}
        bs = []
        for _ in range(500):
            idx = np.concatenate([pos[u] for u in rng.choice(ud, len(ud))])
            bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][2])
        lo, hi = (x > 200) & (x < 600), x >= 600
        out.append({"site": s, "n": len(x), "c2": c[2], "c2_lo": np.percentile(bs, 2.5), "c2_hi": np.percentile(bs, 97.5),
                    "slope_200_600": np.polyfit(x[lo], y[lo], 1)[0], "slope_ge600": np.polyfit(x[hi], y[hi], 1)[0]})
    out = pd.DataFrame(out)
    out.to_csv(F.OUT / "h_shape.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
