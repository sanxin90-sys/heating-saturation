"""
sindy_blh_mechanism.py — EXP-017 (design and rule in EXPERIMENT_LOG, registered before running).
Does dividing SW by the daily afternoon ERA5 boundary-layer height absorb the
saturating SW term in dT/dt at the five towers?

Usage: python src/sindy_blh_mechanism.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_weak_loyo as W  # noqa: E402
import sindy_wind_forms as F  # noqa: E402
import sindy_h_and_npw  # noqa: E402,F401  (registers the Npw site config in F.SITES)

BLH = F.ROOT / "data/era5_blh/blh_sites.csv"
SITES = ["Jaru", "K34", "Sa1", "Ma3", "Npw"]


def cols(d, model, b=None):
    T, SW, H, h = (np.asarray(d[c], float) for c in ("T", "SW", "hour", "hday"))
    w = 2 * np.pi * H / 24
    hh = [f(k * w) for k in range(1, 4) for f in (np.cos, np.sin)]
    one = np.ones_like(T)
    c = {"relax+harm": [one, T, SW], "sat+harm": [one, T, 1 - np.exp(-b * SW) if b else SW],
         "BL+harm": [one, T, 1000 * SW / h], "BL+sat+harm": [one, T, 1000 * SW / h, 1 - np.exp(-b * SW) if b else SW]}[model]
    return np.stack(c + hh, axis=-1)


def fit_pred(d, rows, tr, model):
    best = None
    for b in (F.B_GRID if "sat" in model else [None]):
        G, y = F.weak_G(d, rows, cols(d, model, b))
        xi = np.linalg.lstsq(G[tr], y[tr], rcond=None)[0]
        sse = np.sum((y[tr] - G[tr] @ xi) ** 2)
        if best is None or sse < best[0]:
            best = (sse, G @ xi, b, xi)
    return best


def main():
    blh = pd.read_csv(BLH, parse_dates=["date"])
    hday = blh.groupby(["site", "date"]).blh.max().rename("hday").reset_index()
    rng = np.random.default_rng(0)
    res = []
    for n in SITES:
        cfg = F.SITES[n]
        d = F.load_site(n)
        d["date"] = pd.to_datetime(d.timestamp.dt.date)
        d = d.merge(hday[hday.site == n][["date", "hday"]], on="date", how="left")
        dd = d.assign(Rn=np.where(d.hday.notna(), d.SW, np.nan), WS=1.0)
        rows = W.weak_rows(dd, F.M)
        ry = np.array([r[0] for r in rows]); rd = np.array([r[2] for r in rows])
        tr, te = np.isin(ry, cfg["train"]), np.isin(ry, cfg["test"])
        y = F.weak_G(dd, rows, cols(dd, "relax+harm"))[1]
        P = {m: fit_pred(dd, rows, tr, m) for m in ["relax+harm", "sat+harm", "BL+harm", "BL+sat+harm"]}
        r2 = {m: W.r2(y[te], P[m][1][te]) for m in P}
        lo1, hi1 = W.boot_ci(y[te], P["BL+harm"][1][te], P["relax+harm"][1][te], rd[te], rng)
        g_sat = r2["sat+harm"] - r2["relax+harm"]
        g_add = r2["BL+sat+harm"] - r2["BL+harm"]
        lo2, hi2 = W.boot_ci(y[te], P["BL+sat+harm"][1][te], P["BL+harm"][1][te], rd[te], rng)
        day = d[d.hour.between(10, 14)].groupby("date").agg(sw=("SW", "mean"), h=("hday", "first")).dropna()
        rho = spearmanr(day.sw, day.h)
        res.append({"site": n, "n_test": int(te.sum()), "median_hday_m": float(d.hday.median()),
                    "spearman_SW_h": rho.statistic, "spearman_p": rho.pvalue,
                    "b_sat": P["sat+harm"][2], **{f"r2_{m}": v for m, v in r2.items()},
                    "P1_dR2_BL_vs_relax": r2["BL+harm"] - r2["relax+harm"], "P1_ci_lo": lo1, "P1_ci_hi": hi1,
                    "gain_sat_over_relax": g_sat, "gain_sat_over_BL": g_add, "gain_sat_over_BL_ci": f"[{lo2:.4f}; {hi2:.4f}]",
                    "frac_sat_left": g_add / g_sat if g_sat > 0 else np.nan})
    res = pd.DataFrame(res)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    P1 = res.P1_ci_lo > 0
    P2 = res.frac_sat_left < 0.5
    n_ok = int((P1 & P2).sum()); n_p2fail = int((~P2).sum())
    verdict = "supported" if n_ok >= 4 else ("rejected" if n_p2fail >= 4 else "inconclusive")
    print(f"\nP1 per site: {dict(zip(res.site, P1))}\nP2 per site: {dict(zip(res.site, P2))}\nVERDICT: {verdict}")
    print("between sites: spearman(b, median h) =", spearmanr(res.b_sat, res.median_hday_m))
    res.to_csv(F.OUT / "sindy_blh_mechanism.csv", index=False)


if __name__ == "__main__":
    main()
