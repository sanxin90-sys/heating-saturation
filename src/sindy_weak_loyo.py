"""
sindy_weak_loyo.py
==================

EXP-007: weak-form SINDy + diurnal harmonics, leave-one-year-out (methods/sindy.md ideas 1, 3, 6).

Question: does the quadratic SINDy library (Rn, T, WS) add structure beyond
the linear relaxation + diurnal cycle when (a) no pointwise derivative is
estimated (weak form) and (b) every year is held out in turn?

State variable. Because of DQ-002 (T22 of 2007 has no diurnal cycle; T22 of
2006 equals the 62 m sensor) the state is T62 = FLUXNET BR-Ji3 TA_F_MDS with
QC = 0, the same tower and clock (EXP-005). Rn and WS come from the
spreadsheet; the DQ-001 months of suspect Rn are dropped.

Weak form (Messenger & Bortz 2021 style). On every gap-free run of
(T62, Rn, WS) inside one year, test functions phi_k(t) = (t-a)^p (b-t)^p,
p = 4, support of m half-hours, stride m//2. For dT/dt = Theta(x) xi:
    b_k = -int phi_k' T dt        G_kj = int phi_k Theta_j dt   (trapezoid)
and xi solves G xi = b by STLSQ (same column scaling and 0.1*||b|| threshold
as methods.fit_sindy_poly). The strong form uses the Savitzky-Golay window-7
derivative of T62, as in the earlier experiments.

Models (all with intercept): relax+harm = OLS on [T, Rn, harmonics K=3];
harm = harmonics only; sindy = STLSQ on poly2(Rn, T, WS); sindy+harm = poly2
plus harmonics. Each is fitted in strong and in weak form.

Scores on the held-out year: weak R2 (rows b_k, derivative-free, primary)
and strong R2 against the SG-7 derivative. 95% CI of the difference by a
daily block bootstrap (1000) of the pooled out-of-year predictions.

Decision rule (fixed before running): weak-form sindy+harm is credited with
structure beyond relax+harm only if dR2_weak > 0 with CI excluding zero in all
three season settings (pooled LOYO) AND the point dR2_weak > 0 in each of the
three held-out years (season all). Main support m = 9 (4 h); m = 5, 13, 17 are
a sensitivity check, not part of the rule.

Usage:  python src/sindy_weak_loyo.py "Dados Rebio Jaru 2006 a 2008.xlsx" [--cache df.pkl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_discovery import _sg_derivative, load_jaru  # noqa: E402
from dtdt_robustness import SUSPECT_RN  # noqa: E402

FLX = Path(__file__).resolve().parents[1] / "data/fluxnet_BR-Ji3/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv"
OUT = Path(__file__).resolve().parent / "sr_benchmark" / "results"
YEARS = [2006, 2007, 2008]
DT = 0.5  # h
P = 4
N_BOOT = 1000
MODELS = ["harm", "relax+harm", "sindy", "sindy+harm"]


def harm(h, K=3):
    w = 2 * np.pi * h / 24
    return np.column_stack([f(k * w) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def library(d: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str]]:
    Rn, T, WS, H = (d[c].to_numpy(float) for c in ["Rn", "T", "WS", "hour"])
    one = np.ones(len(d))
    if model == "harm":
        return np.column_stack([one, harm(H)]), ["1"] + [f"h{i}" for i in range(6)]
    if model == "relax+harm":
        return np.column_stack([one, T, Rn, harm(H)]), ["1", "T", "Rn"] + [f"h{i}" for i in range(6)]
    X, nm = [Rn, T, WS], ["Rn", "T", "WS"]
    cols, names = [one, *X], ["1", *nm]
    for i in range(3):
        for j in range(i, 3):
            cols.append(X[i] * X[j]); names.append(f"{nm[i]}*{nm[j]}")
    if model == "sindy+harm":
        cols += list(harm(H).T); names += [f"h{i}" for i in range(6)]
    return np.column_stack(cols), names


def load(xlsx, cache):
    df = pd.read_pickle(cache) if cache and Path(cache).exists() else load_jaru(xlsx)
    f = pd.read_csv(FLX, na_values=[-9999], usecols=["TIMESTAMP_END", "TA_F_MDS", "TA_F_MDS_QC"])
    f["timestamp"] = pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")
    df = df.merge(f[["timestamp", "TA_F_MDS", "TA_F_MDS_QC"]], on="timestamp", how="left")
    df["T"] = df.TA_F_MDS.where(df.TA_F_MDS_QC == 0)
    bad = np.array([(y, m) in SUSPECT_RN for y, m in zip(df.year, df.month)])
    df.loc[bad, "Rn"] = np.nan
    df["dTdt"] = np.nan
    for y in YEARS:  # SG derivative never crosses a year boundary
        s = df.year == y
        df.loc[s, "dTdt"] = _sg_derivative(df.loc[s, "T"], dt_h=DT, window=7, polyorder=2).to_numpy()
    df["day"] = df.timestamp.dt.date
    return df


def weak_rows(df: pd.DataFrame, m: int):
    """Test-function rows: list of (year, season, day, index array) on gap-free runs."""
    ok = df[["T", "Rn", "WS"]].notna().all(axis=1).to_numpy()
    yr = df.year.to_numpy()
    rows = []
    run_start = None
    for i in range(len(df) + 1):
        good = i < len(df) and ok[i] and (run_start is None or yr[i] == yr[run_start])
        if good and run_start is None:
            run_start = i
        if not good and run_start is not None:
            for a in range(run_start, i - m + 1, max(m // 2, 1)):
                idx = np.arange(a, a + m)
                c = idx[m // 2]
                rows.append((yr[c], df.season.iat[c], df.day.iat[c], idx))
            run_start = i if (i < len(df) and ok[i]) else None
    return rows


def phi(m):
    s = np.linspace(0.0, 1.0, m)  # scaled time on the support
    L = (m - 1) * DT
    v = (s * (1 - s)) ** P
    dv = P * (s * (1 - s)) ** (P - 1) * (1 - 2 * s) / L
    w = np.full(m, DT); w[0] = w[-1] = DT / 2  # trapezoid weights
    return v * w, dv * w


def weak_system(df, rows, model, m):
    v, dv = phi(m)
    Theta, names = library(df, model)
    T = df["T"].to_numpy(float)
    G = np.array([v @ Theta[idx] for *_, idx in rows])
    b = np.array([-(dv @ T[idx]) for *_, idx in rows])
    return G, b, names


def stlsq(A, y, threshold=0.1):
    import pysindy as ps
    sc = np.linalg.norm(A, axis=0); sc[sc == 0] = 1
    opt = ps.STLSQ(threshold=threshold * np.linalg.norm(y), alpha=1e-8, max_iter=50)
    opt.fit(A / sc, y.reshape(-1, 1))
    return np.ravel(opt.coef_) / sc


def fit(A, y, model):
    if model in ("harm", "relax+harm"):
        return np.linalg.lstsq(A, y, rcond=None)[0]
    return stlsq(A, y)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def boot_ci(y, pa, pb, days, rng):
    ud = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in ud}
    out = []
    for _ in range(N_BOOT):
        s = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
        out.append(r2(y[s], pa[s]) - r2(y[s], pb[s]))
    return np.percentile(out, [2.5, 97.5])


def run(df, m, rng, verbose=True):
    rows = weak_rows(df, m)
    ry = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows]); rd = np.array([r[2] for r in rows])
    sys_ = {mod: weak_system(df, rows, mod, m) for mod in MODELS}
    strong_ok = df[["T", "Rn", "WS", "dTdt"]].notna().all(axis=1).to_numpy()
    lib = {mod: library(df, mod)[0] for mod in MODELS}
    y_s = df.dTdt.to_numpy(float)
    res, terms = [], []
    for season in ["all", "wet", "dry"]:
        wmask = np.ones(len(rows), bool) if season == "all" else rs == season
        smask = strong_ok & ((df.season == season).to_numpy() if season != "all" else True)
        pred = {(form, mod): {} for form in ("weak", "strong") for mod in MODELS}
        for ty in YEARS:
            wtr, wte = wmask & (ry != ty), wmask & (ry == ty)
            str_, ste = smask & (df.year != ty).to_numpy(), smask & (df.year == ty).to_numpy()
            for mod in MODELS:
                G, b, names = sys_[mod]
                xi_w = fit(G[wtr], b[wtr], mod)
                xi_s = fit(lib[mod][str_], y_s[str_], mod)
                # weak-row predictions and strong (pointwise) predictions, for both fits
                pred[("weak", mod)][ty] = {"weak": G[wte] @ xi_w, "strong": lib[mod][ste] @ xi_w}
                pred[("strong", mod)][ty] = {"weak": G[wte] @ xi_s, "strong": lib[mod][ste] @ xi_s}
                if mod.startswith("sindy"):
                    for form, xi in (("weak", xi_w), ("strong", xi_s)):
                        terms.append({"m": m, "season": season, "test_year": ty, "form": form, "model": mod,
                                      "n_terms": int(np.sum(xi != 0)),
                                      "terms": " + ".join(f"{c:.3g}*{n}" for c, n in zip(xi, names) if c != 0)})
        # pooled out-of-year scores
        for form in ("weak", "strong"):
            for score in ("weak", "strong"):
                if score == "weak":
                    yy = np.concatenate([sys_["harm"][1][wmask & (ry == ty)] for ty in YEARS])
                    dd = np.concatenate([rd[wmask & (ry == ty)] for ty in YEARS])
                else:
                    yy = np.concatenate([y_s[smask & (df.year == ty).to_numpy()] for ty in YEARS])
                    dd = np.concatenate([df.day.to_numpy()[smask & (df.year == ty).to_numpy()] for ty in YEARS])
                P_ = {mod: np.concatenate([pred[(form, mod)][ty][score] for ty in YEARS]) for mod in MODELS}
                ref = P_["relax+harm"]
                for mod in MODELS:
                    lo, hi = boot_ci(yy, P_[mod], ref, dd, rng) if mod != "relax+harm" else (0.0, 0.0)
                    per_year = {}
                    for ty in YEARS:
                        if score == "weak":
                            yt = sys_["harm"][1][wmask & (ry == ty)]
                        else:
                            yt = y_s[smask & (df.year == ty).to_numpy()]
                        per_year[f"dR2_{ty}"] = (r2(yt, pred[(form, mod)][ty][score])
                                                 - r2(yt, pred[(form, "relax+harm")][ty][score])) if len(yt) > 10 else np.nan
                        per_year[f"n_{ty}"] = len(yt)
                    res.append({"m": m, "season": season, "fit": form, "score": score, "model": mod,
                                "r2_pooled": r2(yy, P_[mod]), "dR2_vs_relaxharm": r2(yy, P_[mod]) - r2(yy, ref),
                                "ci_lo": lo, "ci_hi": hi, **per_year})
    return pd.DataFrame(res), pd.DataFrame(terms), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--outdir", type=Path, default=OUT)
    a = ap.parse_args()
    df = load(a.xlsx, a.cache)
    print("half-hours with T62(QC=0), Rn, WS by year:",
          df[df[["T", "Rn", "WS"]].notna().all(axis=1)].groupby("year").size().to_dict())
    rng = np.random.default_rng(0)
    allres, allterms = [], []
    for m in [9, 5, 13, 17]:
        res, terms, n = run(df, m, rng)
        print(f"\n=== support m = {m} half-hours ({(m - 1) * DT:.1f} h), {n} test functions")
        show = res[(res.fit == res.score)]
        print(show[["season", "fit", "model", "r2_pooled", "dR2_vs_relaxharm", "ci_lo", "ci_hi",
                    "dR2_2006", "dR2_2007", "dR2_2008", "n_2006", "n_2007", "n_2008"]].round(4).to_string(index=False))
        allres.append(res); allterms.append(terms)
    res, terms = pd.concat(allres), pd.concat(allterms)
    a.outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.outdir / "sindy_weak_loyo.csv", index=False)
    terms.to_csv(a.outdir / "sindy_weak_loyo_terms.csv", index=False)
    t9 = terms[(terms.m == 9) & (terms.season == "all")]
    print("\nterms, m = 9, season all\n", t9[["test_year", "form", "model", "n_terms", "terms"]].to_string(index=False))


if __name__ == "__main__":
    main()
