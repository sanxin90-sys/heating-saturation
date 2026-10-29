"""
sindy_confirm_newyears.py
=========================

EXP-009: confirmatory test, on years no model has seen, of the structure that
survived EXP-007/008 (methods/sindy.md §8):

    dT/dt = c - lam*T + A*(1 - exp(-b*SW)) + kappa*SW*WS + diurnal harmonics (K = 3)

Data: FLUXNET BR-Ji3 only (TA_F_MDS, SW_IN_F_MDS, WS, each with QC = 0), so
no spreadsheet Rn (DQ-001/DQ-003) or T22 (DQ-002) enters. Checks made before
fixing the design (2026-10-29, logged in EXPERIMENT_LOG):
  * WS in 2004 has a median of 0.37 m/s against 1.5-1.7 in every other year
    -> 2004 is not used for wind models; it gets a no-wind secondary test only.
  * FLUXNET WS 2007 correlates r = 0.31 with the spreadsheet WS (0.82 in 2006
    and 2008) -> 2007 is not used for training.
  * 2009-12 has no diurnal wind cycle (noon - night median -0.08 m/s) -> dropped.
Training: 2006 + 2008. Test: 2005, 2009, 2010 (each scored with the model
trained on 2006 + 2008; no refit on test data). Weak form exactly as EXP-007
(phi = (t-a)^4 (b-t)^4, m = 9 half-hours, stride 4, gap-free runs per year).

Models: relax+harm [1, T, SW, harm]; sat+harm [1, T, 1-exp(-b SW), harm];
M1 = sat+wind+harm [sat+harm, SW*WS]; sindy+harm (STLSQ poly2 in SW, T, WS
+ harm). b is chosen on the training years from a grid by weak residual.

Decision rules (fixed before running):
 R1 pooled test dR2_weak(M1 - relax+harm) > 0 with 95 % daily-block CI
    excluding zero in season all, wet and dry;
 R2 dR2_weak > 0 in each test year (season all);
 R3 24-h free-run mean RMSE of T lower for M1 than relax+harm, CI excluding 0;
 R4 kappa < 0 in the training fit and in a refit on each test year alone.
The structure is "confirmed" only if R1-R3 pass; R4 is reported as a sign check.
Secondary (not part of the rule): 2004, sat+harm vs relax+harm.

Usage: python src/sindy_confirm_newyears.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_weak_loyo as W  # noqa: E402

FLX = W.FLX
OUT = W.OUT
TRAIN, TEST = [2006, 2008], [2005, 2009, 2010]
B_GRID = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02]
M = 9
MODELS = ["harm", "relax+harm", "sat+harm", "sat+wind+harm", "sindy+harm"]


def load():
    f = pd.read_csv(FLX, na_values=[-9999], usecols=["TIMESTAMP_END", "TA_F_MDS", "TA_F_MDS_QC", "SW_IN_F_MDS",
                                                     "SW_IN_F_MDS_QC", "WS", "WS_F_QC"])
    d = pd.DataFrame({"timestamp": pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")})
    d["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0)
    d["SW"] = f.SW_IN_F_MDS.where(f.SW_IN_F_MDS_QC == 0)
    d["WS"] = f.WS.where(f.WS_F_QC == 0)
    d["year"], d["month"] = d.timestamp.dt.year, d.timestamp.dt.month
    d = d[d.year.between(2004, 2010)].reset_index(drop=True)
    d.loc[(d.year == 2009) & (d.month == 12), "WS"] = np.nan
    d["hour"] = d.timestamp.dt.hour + d.timestamp.dt.minute / 60
    d["season"] = np.select([d.month.isin([12, 1, 2, 3, 4]), d.month.isin([6, 7, 8, 9])], ["wet", "dry"], "transition")
    d["day"] = d.timestamp.dt.date
    d["Rn"] = d["SW"]  # W.weak_rows checks T, Rn, WS
    return d


def cols(d, model, b=None):
    T, SW, WS, H = (np.asarray(d[c], float) for c in ("T", "SW", "WS", "hour"))
    one = np.ones_like(T)
    w = 2 * np.pi * H / 24
    hh = [f(k * w) for k in range(1, 4) for f in (np.cos, np.sin)]
    if model == "harm":
        c = [one, *hh]
    elif model == "relax+harm":
        c = [one, T, SW, *hh]
    elif model == "sat+harm":
        c = [one, T, 1 - np.exp(-b * SW), *hh]
    elif model == "sat+wind+harm":
        c = [one, T, 1 - np.exp(-b * SW), SW * WS, *hh]
    else:
        X = [SW, T, WS]
        c = [one, *X] + [X[i] * X[j] for i in range(3) for j in range(i, 3)] + hh
    return np.stack(c, axis=-1)


def weak_G(d, rows, Th):
    v, dv = W.phi(M)
    T = d["T"].to_numpy(float)
    return np.array([v @ Th[idx] for *_, idx in rows]), np.array([-(dv @ T[idx]) for *_, idx in rows])


def fit(d, rows, tr, model):
    if model in ("sat+harm", "sat+wind+harm"):
        best = None
        for b in B_GRID:
            G, y = weak_G(d, rows, cols(d, model, b))
            xi = np.linalg.lstsq(G[tr], y[tr], rcond=None)[0]
            sse = np.sum((y[tr] - G[tr] @ xi) ** 2)
            if best is None or sse < best[0]:
                best = (sse, xi, b)
        return best[1], best[2]
    G, y = weak_G(d, rows, cols(d, model))
    return (W.stlsq(G[tr], y[tr]) if model == "sindy+harm" else np.linalg.lstsq(G[tr], y[tr], rcond=None)[0]), None


def simulate(d, model, xi, b, year):
    dy = d[d.year == year].reset_index(drop=True)
    need = dy[["T", "SW", "WS"]].notna().all(axis=1).to_numpy()
    starts = [s for s in np.flatnonzero(np.isclose(dy.hour.to_numpy(), 0.0)) if s + 48 < len(dy) and need[s:s + 49].all()]
    if not starts:
        return np.array([]), np.array([])
    idx = np.array(starts)[:, None] + np.arange(49)[None, :]
    SWa, WSa, Ta = (dy[c].to_numpy(float)[idx] for c in ("SW", "WS", "T"))
    h0 = dy.hour.to_numpy(float)[idx[:, 0]]

    def rhs(T, j, fr):
        jj = min(j + 1, 48)
        dd = {"T": T, "SW": SWa[:, j] + fr * (SWa[:, jj] - SWa[:, j]), "WS": WSa[:, j] + fr * (WSa[:, jj] - WSa[:, j]),
              "hour": (h0 + 0.5 * (j + fr)) % 24}
        return cols(dd, model, b) @ xi

    T = Ta[:, 0].copy(); sim = []; h = 0.05
    with np.errstate(all="ignore"):
        for j in range(48):
            for k in range(10):
                fr = k / 10
                k1 = rhs(T, j, fr); k2 = rhs(T + h / 2 * k1, j, fr + 0.05)
                k3 = rhs(T + h / 2 * k2, j, fr + 0.05); k4 = rhs(T + h * k3, j, fr + 0.1)
                T = T + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            sim.append(T.copy())
    sim = np.array(sim).T
    rmse = np.sqrt(np.mean((sim - Ta[:, 1:]) ** 2, axis=1))
    rmse[~np.all((sim > 0) & (sim < 60), axis=1)] = np.nan
    return dy.day.to_numpy()[idx[:, 0]], rmse


def main():
    d = load()
    rng = np.random.default_rng(0)
    rows = W.weak_rows(d, M)
    ry = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows]); rd = np.array([r[2] for r in rows])
    print("weak rows per year:", pd.Series(ry).value_counts().sort_index().to_dict())
    tr_all = np.isin(ry, TRAIN)

    # wind models: trained once on 2006 + 2008 (season-specific fits for the season rows)
    res, coefs = [], []
    for season in ["all", "wet", "dry"]:
        smask = np.ones(len(rows), bool) if season == "all" else rs == season
        tr, te = tr_all & smask, np.isin(ry, TEST) & smask
        P = {}
        for m in MODELS:
            xi, b = fit(d, rows, tr, m)
            G, y = weak_G(d, rows, cols(d, m, b))
            P[m] = (G @ xi, xi, b)
            if season == "all" and m in ("sat+harm", "sat+wind+harm"):
                coefs.append({"fit": "train 2006+2008", "model": m, "b": b,
                              **({"kappa": xi[3]} if m == "sat+wind+harm" else {}), "lam": -xi[1], "A": xi[2]})
        y = weak_G(d, rows, cols(d, "harm"))[1]
        for m in MODELS:
            p, pr = P[m][0], P["relax+harm"][0]
            lo, hi = W.boot_ci(y[te], p[te], pr[te], rd[te], rng) if m != "relax+harm" else (0, 0)
            row = {"season": season, "model": m, "n_test": int(te.sum()), "r2_test": W.r2(y[te], p[te]),
                   "dR2": W.r2(y[te], p[te]) - W.r2(y[te], pr[te]), "ci_lo": lo, "ci_hi": hi}
            for ty in TEST:
                t1 = te & (ry == ty)
                row[f"dR2_{ty}"] = W.r2(y[t1], p[t1]) - W.r2(y[t1], pr[t1]) if t1.sum() > 20 else np.nan
                row[f"n_{ty}"] = int(t1.sum())
            res.append(row)
    res = pd.DataFrame(res)
    print("\nR1/R2. weak form, trained 2006+2008, tested 2005/2009/2010\n", res.round(4).to_string(index=False))

    # R4: refit M1 on each test year alone
    for ty in TEST:
        xi, b = fit(d, rows, ry == ty, "sat+wind+harm")
        coefs.append({"fit": f"refit {ty} alone", "model": "sat+wind+harm", "b": b, "kappa": xi[3], "lam": -xi[1], "A": xi[2]})
    coefs = pd.DataFrame(coefs)
    print("\nR4. coefficients\n", coefs.to_string(index=False))

    # R3: simulation, season all models trained on 2006+2008
    sims = []
    for m in MODELS:
        xi, b = fit(d, rows, tr_all, m)
        for ty in TEST:
            days, rm = simulate(d, m, xi, b, ty)
            sims += [{"model": m, "test_year": ty, "day": x, "rmse": r} for x, r in zip(days, rm)]
    sims = pd.DataFrame(sims)
    piv = sims.pivot_table(index=["test_year", "day"], columns="model", values="rmse")
    sim_out = []
    for m in MODELS:
        ok = piv[m].notna() & piv["relax+harm"].notna()
        diff = (piv[m] - piv["relax+harm"])[ok].to_numpy()
        bs = [np.mean(rng.choice(diff, len(diff))) for _ in range(1000)]
        sim_out.append({"model": m, "n_days": int(ok.sum()), "blowups": int(piv[m].isna().sum()),
                        "mean_rmse": piv[m][ok].mean(), "dRMSE": diff.mean(),
                        "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                        **{f"mean_{ty}": piv[m][ok].xs(ty, level=0).mean() for ty in TEST}})
    sim_out = pd.DataFrame(sim_out)
    print("\nR3. 24-h free runs, test years pooled\n", sim_out.round(4).to_string(index=False))

    # secondary: 2004 without wind
    sec = []
    te4 = ry == 2004
    for m in ["harm", "relax+harm", "sat+harm"]:
        xi, b = fit(d, rows, tr_all, m)
        G, y = weak_G(d, rows, cols(d, m, b))
        sec.append({"model": m, "n": int(te4.sum()), "r2_2004": W.r2(y[te4], G[te4] @ xi)})
    sec = pd.DataFrame(sec)
    print("\nSecondary, 2004 (no wind)\n", sec.round(4).to_string(index=False))

    verdict = {
        "R1": bool(all(res[(res.model == "sat+wind+harm") & (res.season == s)].ci_lo.iloc[0] > 0 for s in ["all", "wet", "dry"])),
        "R2": bool(all(res[(res.model == "sat+wind+harm") & (res.season == "all")][f"dR2_{ty}"].iloc[0] > 0 for ty in TEST)),
        "R3": bool(sim_out[sim_out.model == "sat+wind+harm"].ci_hi.iloc[0] < 0),
        "R4": bool((coefs[coefs.model == "sat+wind+harm"].kappa < 0).all()),
    }
    print("\nVERDICT", verdict)
    res.to_csv(OUT / "sindy_confirm_weak.csv", index=False)
    coefs.to_csv(OUT / "sindy_confirm_coefs.csv", index=False)
    sims.to_csv(OUT / "sindy_confirm_sim.csv", index=False)
    sim_out.to_csv(OUT / "sindy_confirm_sim_summary.csv", index=False)
    sec.to_csv(OUT / "sindy_confirm_2004.csv", index=False)


if __name__ == "__main__":
    main()
