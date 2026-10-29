"""
pysr_sites.py
=============

PySR on the five FLUXNET towers used by the SINDy thread (methods/sindy.md
§9-13), to test whether an independent SR engine finds, without being told,
the radiation saturation that SINDy found (b ~ 0.002 m2 W-1 at 5/5 sites).

Design (fixed before running, 2026-10-29):

* Data exactly as the SINDy thread: FLUXNET TA_F_MDS, SW_IN_F_MDS (QC = 0),
  same train/test years per site. No variable from the Jaru spreadsheet.
* Target: dT/dt (K h-1) from a Savitzky-Golay filter (window 7, order 2) on
  gap-free runs of T. Strong form, since PySR fits pointwise.
* Grey-box: the reference model relax+harm (OLS on 1, T, SW and diurnal
  harmonics K = 3) is fitted first; PySR searches only the residual
  r = dT/dt - relax+harm, with features SW and T. Without the harmonics
  baseline the search rediscovers the daily cycle (sindy.md §14, lesson i).
* Selection: the train years are split into fit (all but the last) and
  validation (the last). relax+harm and PySR are fitted on the fit years; the
  Pareto candidate with the lowest validation RMSE is chosen; the test years
  are used only for scoring.
* Seeds 0, 1, 2 (each with its own subsample of the fit rows).
* Scores on the test years: R2 of relax+harm, sat+harm (A(1 - exp(-b SW)),
  b on the SINDy grid, OLS) and relax+harm + PySR; dR2 against relax+harm
  with a 95 % daily-block bootstrap CI.
* Structure: the chosen f(SW, T) is "saturating in SW" when its mean second
  difference over SW = 0-900 W m-2 (T at the fit-year median) is negative.

Rule (written before running): PySR independently reproduces the SINDy result
at a site when, in >= 2 of 3 seeds, the chosen f is saturating in SW and its
dR2 CI on the test years excludes zero.

Usage
-----
    python src/pysr_sites.py                       # all sites, PySR (needs Julia)
    python src/pysr_sites.py --no-pysr             # baselines only
    python src/pysr_sites.py --sites Jaru K34 --seeds 0 --budget 60
    python pysr_sites.py --flux-dir <folder with AMF_*_FLUXMET_HH csvs> --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "sr_benchmark" / "results"
B_GRID = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02]

SITES = {
    "Jaru": dict(fid="BR-Ji3", csv="data/fluxnet_BR-Ji3/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv",
                 train=[2006, 2008], test=[2005, 2009, 2010]),
    "K34": dict(fid="BR-Ma2", csv="data/fluxnet_BR-Ma2/BR-Ma2_HH_TA_SW_WS_1999-2016.csv",
                train=[2002, 2003, 2004, 2005], test=[2006, 2007, 2008, 2009]),
    "Sa1": dict(fid="BR-Sa1", csv="data/fluxnet_BR-Sa1/BR-Sa1_HH_TA_SW_WS.csv",
                train=[2002, 2003, 2004], test=[2009, 2010, 2011]),
    "Ma3": dict(fid="BR-Ma3", csv="data/fluxnet_BR-Ma3/BR-Ma3_HH_TA_SW_WS.csv",
                train=[2009, 2010, 2011], test=[2013, 2018]),
    "Npw": dict(fid="BR-Npw", csv="data/fluxnet_BR-Npw/BR-Npw_HH_TA_SW_WS.csv",
                train=[2014, 2015], test=[2016, 2017]),
}


FLUX_DIR: Path | None = None  # --flux-dir: search the original AMF_*_FLUXMET_HH_*.csv there instead


def load(site: str) -> pd.DataFrame:
    cfg = SITES[site]
    path = ROOT / cfg["csv"]
    if FLUX_DIR is not None:
        path = sorted(FLUX_DIR.rglob(f"AMF_{cfg['fid']}_FLUXNET_FLUXMET_HH_*.csv"))[0]
    f = pd.read_csv(path, na_values=[-9999],
                    usecols=["TIMESTAMP_END", "TA_F_MDS", "TA_F_MDS_QC", "SW_IN_F_MDS", "SW_IN_F_MDS_QC"])
    d = pd.DataFrame({"timestamp": pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")})
    d["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0)
    d["SW"] = f.SW_IN_F_MDS.where(f.SW_IN_F_MDS_QC == 0)
    d["year"] = d.timestamp.dt.year
    d["hour"] = d.timestamp.dt.hour + d.timestamp.dt.minute / 60
    d["day"] = d.timestamp.dt.date
    d = d[d.year.isin(cfg["train"] + cfg["test"])].reset_index(drop=True)
    d["dTdt"] = np.nan
    run = d["T"].isna().cumsum()
    for _, seg in d["T"].dropna().groupby(run[d["T"].notna()]):
        if len(seg) >= 7:
            d.loc[seg.index, "dTdt"] = savgol_filter(seg.to_numpy(), 7, 2, deriv=1, delta=0.5)
    return d.dropna(subset=["T", "SW", "dTdt"]).reset_index(drop=True)


def design(d: pd.DataFrame, model: str, b: float | None = None) -> np.ndarray:
    w = 2 * np.pi * d.hour.to_numpy() / 24
    harm = [f(k * w) for k in range(1, 4) for f in (np.cos, np.sin)]
    T, SW = d["T"].to_numpy(), d["SW"].to_numpy()
    rad = SW if model == "relax+harm" else 1 - np.exp(-b * SW)
    return np.column_stack([np.ones(len(d)), T, rad, *harm])


def ols(d_fit, model, b=None):
    coef, *_ = np.linalg.lstsq(design(d_fit, model, b), d_fit.dTdt.to_numpy(), rcond=None)
    return coef


def fit_sat(d_fit, d_val):
    """b chosen on the validation year, like the SINDy grid search."""
    best = None
    for b in B_GRID:
        c = ols(d_fit, "sat+harm", b)
        e = np.mean((d_val.dTdt - design(d_val, "sat+harm", b) @ c) ** 2)
        if best is None or e < best[0]:
            best = (e, b, c)
    return best[1], best[2]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def boot_dr2(y, p, p0, days, rng, n=1000):
    days = np.asarray(days)
    uniq = np.unique(days)
    idx = {u: np.flatnonzero(days == u) for u in uniq}
    out = []
    for _ in range(n):
        sel = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        out.append(r2(y[sel], p[sel]) - r2(y[sel], p0[sel]))
    return np.percentile(out, [2.5, 97.5])


def saturating(predict_f, T_med: float) -> tuple[bool, float]:
    sw = np.linspace(0, 900, 91)
    f = predict_f(np.column_stack([sw, np.full_like(sw, T_med)]))
    d2 = np.diff(f, 2).mean()
    return bool(d2 < 0), float(d2)


def run_site(site, seeds, budget, n_max, use_pysr, rng):
    cfg = SITES[site]
    d = load(site)
    fit_y, val_y = cfg["train"][:-1], cfg["train"][-1]
    dfit, dval, dte = d[d.year.isin(fit_y)], d[d.year == val_y], d[d.year.isin(cfg["test"])]
    y = dte.dTdt.to_numpy()
    c0 = ols(dfit, "relax+harm")
    p0 = design(dte, "relax+harm") @ c0
    b, cs = fit_sat(dfit, dval)
    ps = design(dte, "sat+harm", b) @ cs
    lo, hi = boot_dr2(y, ps, p0, dte.day, rng)
    print(f"\n== {site}: fit {fit_y}, val {val_y}, test {cfg['test']}; rows fit {len(dfit)}, val {len(dval)}, test {len(dte)}")
    rows = [{"site": site, "model": "relax+harm", "seed": -1, "r2_test": r2(y, p0), "dR2": 0.0},
            {"site": site, "model": f"sat+harm (b={b})", "seed": -1, "r2_test": r2(y, ps),
             "dR2": r2(y, ps) - r2(y, p0), "ci_lo": lo, "ci_hi": hi}]
    fronts = []
    if use_pysr:
        from pysr import PySRRegressor

        res = lambda dd: dd.dTdt.to_numpy() - design(dd, "relax+harm") @ c0
        X = lambda dd: dd[["SW", "T"]].to_numpy(float)
        for seed in seeds:
            sub = dfit.sample(min(n_max, len(dfit)), random_state=seed)
            reg = PySRRegressor(
                niterations=100000, populations=30, maxsize=15,
                binary_operators=["+", "-", "*", "/"], unary_operators=["exp", "square"],
                nested_constraints={"exp": {"exp": 0}}, timeout_in_seconds=budget,
                model_selection="accuracy", random_state=seed, deterministic=True,
                parallelism="serial", progress=False, verbosity=0,
            )
            reg.fit(X(sub), res(sub), variable_names=["SW", "T"])
            eqs = reg.equations_[["complexity", "loss", "equation"]].copy()
            eqs["rmse_val"] = [np.sqrt(np.mean((res(dval) - reg.predict(X(dval), index=i)) ** 2)) for i in range(len(eqs))]
            k = int(eqs.rmse_val.idxmin())
            pt = p0 + reg.predict(X(dte), index=k)
            lo, hi = boot_dr2(y, pt, p0, dte.day, rng)
            sat, d2 = saturating(lambda Z: reg.predict(Z, index=k), float(dfit["T"].median()))
            rows.append({"site": site, "model": "relax+harm + PySR", "seed": seed, "r2_test": r2(y, pt),
                         "dR2": r2(y, pt) - r2(y, p0), "ci_lo": lo, "ci_hi": hi, "complexity": eqs.complexity[k],
                         "equation": eqs.equation[k], "saturating_SW": sat, "d2f_dSW2": d2})
            eqs["r2_test_total"] = [r2(y, p0 + reg.predict(X(dte), index=i)) for i in range(len(eqs))]
            eqs.insert(0, "seed", seed)
            eqs.insert(0, "site", site)
            fronts.append(eqs)
    out = pd.DataFrame(rows)
    print(out.drop(columns=[c for c in ["equation"] if c in out]).round(4).to_string(index=False))
    if "equation" in out:
        for _, r in out.dropna(subset=["equation"]).iterrows():
            print(f"  seed {r.seed}: {r.equation}")
    return out, (pd.concat(fronts) if fronts else pd.DataFrame())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=180)
    ap.add_argument("--n-max", type=int, default=5000)
    ap.add_argument("--no-pysr", action="store_true")
    ap.add_argument("--flux-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    global FLUX_DIR, OUT
    FLUX_DIR, OUT = args.flux_dir, (args.out or OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summ, fronts = zip(*[run_site(s, args.seeds, args.budget, args.n_max, not args.no_pysr, rng) for s in args.sites])
    summ = pd.concat(summ, ignore_index=True)
    tag = "baselines" if args.no_pysr else "pysr"
    summ.to_csv(OUT / f"pysr_sites_{tag}.csv", index=False)
    if not args.no_pysr:
        pd.concat(fronts, ignore_index=True).to_csv(OUT / "pysr_sites_fronts.csv", index=False)
        p = summ[summ.model == "relax+harm + PySR"]
        verdict = p.assign(ok=p.saturating_SW & (p.ci_lo > 0)).groupby("site").ok.sum()
        print("\nRULE (>= 2 of 3 seeds saturating with CI > 0):", (verdict >= 2).to_dict())


if __name__ == "__main__":
    main()
