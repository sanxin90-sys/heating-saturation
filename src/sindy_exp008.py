"""
sindy_exp008.py
===============

EXP-008: next actions after EXP-007 (methods/sindy.md §8). Same data, state
(T62 = FLUXNET TA_F_MDS, QC = 0), DQ-001 exclusions, weak form (m = 9,
phi = (t-a)^4 (b-t)^4) and leave-one-year-out (2006/2007/2008) as EXP-007.

A. Radiation scale between years. Diagnostic: monthly median of Rn/PAR,
   SW_IN/PAR and Rn/SW_IN (daytime, PAR > 50). Test: refit relax+harm and
   sindy+harm with (i) raw Rn, (ii) Rn rescaled per year to a common median
   Rn/SW_IN (uses forcing only, never the target), (iii) SW_IN (FLUXNET,
   QC = 0) as the forcing instead of Rn.
B. Physical library: sat+harm = OLS on [1, T, 1-exp(-b*F), harmonics K=3]
   (same 9 coefficients as relax+harm, plus b); b chosen on the training years
   from a grid by weak residual. phys+harm adds the linear F term.
C. Simulation: from each held-out midnight with 48 gap-free half-hours of
   forcing and observed T, integrate the fitted ODE for 24 h (RK4, 10 substeps,
   forcing linearly interpolated) from the observed T0; daily RMSE of T.

Decision rules (fixed before running):
 A  the 2007 hold-out failure is attributed to radiation scale if the
    dR2_weak(sindy+harm - relax+harm) for 2007 held out becomes > 0 under
    (ii) or (iii) while 2006 and 2008 stay > 0.
 B  saturation alone carries the gain if sat+harm > relax+harm with pooled
    CI > 0 in all three seasons, dR2 > 0 in every held-out year, and the
    pooled all-season gain is >= 80 % of the sindy+harm gain.
 C  a model is dynamically better if its pooled mean daily RMSE is lower than
    relax+harm with a daily-bootstrap CI excluding zero.

Usage: python src/sindy_exp008.py "Dados Rebio Jaru 2006 a 2008.xlsx" [--cache df.pkl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_weak_loyo as W  # noqa: E402

M = 9
B_GRID = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02]
OUT = W.OUT


def load(xlsx, cache):
    df = W.load(xlsx, cache)
    f = pd.read_csv(W.FLX, na_values=[-9999], usecols=["TIMESTAMP_END", "SW_IN_F_MDS", "SW_IN_F_MDS_QC"])
    f["timestamp"] = pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")
    df = df.merge(f[["timestamp", "SW_IN_F_MDS", "SW_IN_F_MDS_QC"]], on="timestamp", how="left")
    df["SW"] = df.SW_IN_F_MDS.where(df.SW_IN_F_MDS_QC == 0)
    day = df.PAR > 50
    ratio = (df.Rn / df.SW).where(day & (df.SW > 100))
    ref = ratio.median()
    k = ratio.groupby(df.year).median()
    df["Rn_scaled"] = df.Rn * df.year.map(ref / k)
    return df, k, ref


def cols(d, model, F="Rn", b=None):
    """d: DataFrame or dict of equal-shape arrays with T, F, WS, hour."""
    T, Fv, H = (np.asarray(d[c], float) for c in ("T", F, "hour"))
    one = np.ones_like(T)
    w = 2 * np.pi * H / 24
    hh = [f(k * w) for k in range(1, 4) for f in (np.cos, np.sin)]
    if model == "relax+harm":
        c = [one, T, Fv, *hh]
    elif model == "sat+harm":
        c = [one, T, 1 - np.exp(-b * Fv), *hh]
    elif model == "phys+harm":
        c = [one, T, Fv, 1 - np.exp(-b * Fv), *hh]
    elif model == "harm":
        c = [one, *hh]
    else:  # sindy+harm
        X = [Fv, T, np.asarray(d["WS"], float)]
        c = [one, *X] + [X[i] * X[j] for i in range(3) for j in range(i, 3)] + hh
    return np.stack(c, axis=-1)


def rows_for(df, F):
    d = df.copy()
    d.loc[d[F].isna(), "Rn"] = np.nan  # weak_rows checks T, Rn, WS
    return W.weak_rows(d, M)


def weak_G(df, rows, Theta):
    v, dv = W.phi(M)
    T = df["T"].to_numpy(float)
    return np.array([v @ Theta[idx] for *_, idx in rows]), np.array([-(dv @ T[idx]) for *_, idx in rows])


def fit_model(df, rows, tr, model, F):
    """Returns (xi, b) fitted on weak rows tr."""
    if model in ("sat+harm", "phys+harm"):
        best = None
        for b in B_GRID:
            G, y = weak_G(df, rows, cols(df, model, F, b))
            xi = np.linalg.lstsq(G[tr], y[tr], rcond=None)[0]
            sse = np.sum((y[tr] - G[tr] @ xi) ** 2)
            if best is None or sse < best[0]:
                best = (sse, xi, b)
        return best[1], best[2]
    G, y = weak_G(df, rows, cols(df, model, F))
    xi = W.stlsq(G[tr], y[tr]) if model == "sindy+harm" else np.linalg.lstsq(G[tr], y[tr], rcond=None)[0]
    return xi, None


def simulate(df, model, F, xi, b, year):
    """24 h free runs from each midnight of `year`, all days at once (RK4, 10 substeps)."""
    d = df[df.year == year].reset_index(drop=True)
    need = d[["T", F, "WS"]].notna().all(axis=1).to_numpy()
    starts = [s for s in np.flatnonzero(np.isclose(d.hour.to_numpy(), 0.0))
              if s + 48 < len(d) and need[s:s + 49].all()]
    if not starts:
        return np.array([]), np.array([])
    idx = np.array(starts)[:, None] + np.arange(49)[None, :]
    Fa, WSa, Ta = (d[c].to_numpy(float)[idx] for c in (F, "WS", "T"))
    h0 = d.hour.to_numpy(float)[idx[:, 0]]

    def rhs(T, j, fr):
        jj = min(j + 1, 48)
        dd = {"T": T, F: Fa[:, j] + fr * (Fa[:, jj] - Fa[:, j]), "WS": WSa[:, j] + fr * (WSa[:, jj] - WSa[:, j]),
              "hour": (h0 + 0.5 * (j + fr)) % 24}
        return cols(dd, model, F, b) @ xi

    T = Ta[:, 0].copy(); sim = []
    n_sub = 10; h = 0.5 / n_sub
    with np.errstate(all="ignore"):
        for j in range(48):
            for k in range(n_sub):
                fr = k / n_sub
                k1 = rhs(T, j, fr); k2 = rhs(T + h / 2 * k1, j, fr + 0.05)
                k3 = rhs(T + h / 2 * k2, j, fr + 0.05); k4 = rhs(T + h * k3, j, fr + 0.1)
                T = T + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            sim.append(T.copy())
    sim = np.array(sim).T
    rmse = np.sqrt(np.mean((sim - Ta[:, 1:]) ** 2, axis=1))
    bad = ~np.all((sim > 0) & (sim < 60), axis=1)
    rmse[bad] = np.nan
    return d.day.to_numpy()[idx[:, 0]], rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    df, k, ref = load(a.xlsx, a.cache)
    rng = np.random.default_rng(0)

    # ---- A. diagnostic
    day = (df.PAR > 50) & (df.SW > 100)
    diag = df[day].assign(Rn_PAR=df.Rn / df.PAR, SW_PAR=df.SW / df.PAR, Rn_SW=df.Rn / df.SW) \
        .groupby(["year", "month"])[["Rn_PAR", "SW_PAR", "Rn_SW"]].median()
    print("A. monthly medians (daytime)\n", diag.round(3).to_string())
    print("\nA. per-year median Rn/SW:", k.round(3).to_dict(), "reference", round(ref, 3))
    diag.to_csv(OUT / "sindy_exp008_radiation_diag.csv")

    res, sims = [], []
    for F in ["Rn", "Rn_scaled", "SW"]:
        rows = rows_for(df, F)
        ry = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows]); rd = np.array([r[2] for r in rows])
        models = ["harm", "relax+harm", "sindy+harm", "sat+harm", "phys+harm"]
        Gs = {}
        for season in ["all", "wet", "dry"]:
            mask = np.ones(len(rows), bool) if season == "all" else rs == season
            P = {m: [] for m in models}; Y, D, YR = [], [], []
            per_year = {m: {} for m in models}
            for ty in W.YEARS:
                tr, te = mask & (ry != ty), mask & (ry == ty)
                yy = None
                for m in models:
                    xi, b = fit_model(df, rows, tr, m, F)
                    G, y = weak_G(df, rows, cols(df, m, F, b))
                    P[m].append(G[te] @ xi); yy = y[te]
                    per_year[m][ty] = (xi, b, W.r2(yy, G[te] @ xi))
                    if season == "all":
                        dd, rm = simulate(df, m, F, xi, b, ty)
                        sims += [{"forcing": F, "model": m, "test_year": ty, "day": x, "rmse": r, "b": b}
                                 for x, r in zip(dd, rm)]
                Y.append(yy); D.append(rd[te])
            Y, D = np.concatenate(Y), np.concatenate(D)
            for m in models:
                p = np.concatenate(P[m]); pr = np.concatenate(P["relax+harm"])
                lo, hi = W.boot_ci(Y, p, pr, D, rng) if m != "relax+harm" else (0, 0)
                row = {"forcing": F, "season": season, "model": m, "r2_pooled": W.r2(Y, p),
                       "dR2": W.r2(Y, p) - W.r2(Y, pr), "ci_lo": lo, "ci_hi": hi}
                for ty in W.YEARS:
                    row[f"dR2_{ty}"] = per_year[m][ty][2] - per_year["relax+harm"][ty][2]
                    row[f"b_{ty}"] = per_year[m][ty][1]
                res.append(row)
        print(f"\n=== forcing {F}: {len(rows)} weak rows")
        r = pd.DataFrame([x for x in res if x["forcing"] == F])
        print(r.drop(columns="forcing").round(4).to_string(index=False))
    res = pd.DataFrame(res); res.to_csv(OUT / "sindy_exp008_weak.csv", index=False)

    # ---- C. simulation summary
    sims = pd.DataFrame(sims); sims.to_csv(OUT / "sindy_exp008_sim.csv", index=False)
    print("\nC. 24-h free-run RMSE of T (K), pooled held-out years")
    out = []
    for F, g in sims.groupby("forcing"):
        piv = g.pivot_table(index=["test_year", "day"], columns="model", values="rmse")
        ref_ = piv["relax+harm"]
        for m in piv.columns:
            ok = piv[m].notna() & ref_.notna()
            diff = (piv[m] - ref_)[ok].to_numpy()
            bs = [np.mean(rng.choice(diff, len(diff))) for _ in range(1000)] if m != "relax+harm" else [0]
            out.append({"forcing": F, "model": m, "n_days": int(ok.sum()), "blowups": int(piv[m].isna().sum() - ref_.isna().sum()),
                        "mean_rmse": piv[m][ok].mean(), "median_rmse": piv[m][ok].median(),
                        "dRMSE_vs_relax": diff.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                        **{f"mean_{ty}": piv[m][ok].xs(ty, level=0).mean() for ty in W.YEARS}})
    out = pd.DataFrame(out); out.to_csv(OUT / "sindy_exp008_sim_summary.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()


def wind_check(xlsx=None, cache=None):
    """Follow-up (EXP-008 D, decided after seeing the SW sindy+harm terms, so exploratory):
    does one wind term SW*WS on top of sat+harm recover the rest of the sindy+harm gain?"""
    df, *_ = load(xlsx, cache)
    rng = np.random.default_rng(1)
    rows = rows_for(df, "SW")
    ry = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows]); rd = np.array([r[2] for r in rows])
    T, SW, WS = (df[c].to_numpy(float) for c in ("T", "SW", "WS"))
    out = []
    for season in ["all", "wet", "dry"]:
        mask = np.ones(len(rows), bool) if season == "all" else rs == season
        P = {"sat+harm": [], "sat+wind+harm": []}; Y = []; D = []; per = {}
        for ty in W.YEARS:
            tr, te = mask & (ry != ty), mask & (ry == ty)
            for m in P:
                best = None
                for b in B_GRID:
                    base = cols(df, "sat+harm", "SW", b)
                    Th = base if m == "sat+harm" else np.column_stack([base, SW * WS])
                    G, y = weak_G(df, rows, Th)
                    xi = np.linalg.lstsq(G[tr], y[tr], rcond=None)[0]
                    sse = np.sum((y[tr] - G[tr] @ xi) ** 2)
                    if best is None or sse < best[0]:
                        best = (sse, G[te] @ xi, xi[-1])
                P[m].append(best[1]); per[(m, ty)] = (W.r2(y[te], best[1]), best[2])
            Y.append(y[te]); D.append(rd[te])
        Y, D = np.concatenate(Y), np.concatenate(D)
        pa, pb = np.concatenate(P["sat+wind+harm"]), np.concatenate(P["sat+harm"])
        lo, hi = W.boot_ci(Y, pa, pb, D, rng)
        out.append({"season": season, "r2_sat": W.r2(Y, pb), "r2_sat_wind": W.r2(Y, pa), "dR2": W.r2(Y, pa) - W.r2(Y, pb),
                    "ci_lo": lo, "ci_hi": hi,
                    **{f"dR2_{ty}": per[("sat+wind+harm", ty)][0] - per[("sat+harm", ty)][0] for ty in W.YEARS},
                    **{f"coef_SWxWS_{ty}": per[("sat+wind+harm", ty)][1] for ty in W.YEARS}})
    out = pd.DataFrame(out)
    out.to_csv(OUT / "sindy_exp008_wind.csv", index=False)
    print("\nD. sat+harm + SW*WS (forcing SW, weak, LOYO)\n", out.round(5).to_string(index=False))
