"""
real_discovery.py
=================

Equation discovery on the real Rebio Jaru tower record and comparison with
established (textbook) relations re-fitted to the same data.

Protocol
--------
* Data: sheet "DADOS OK", years 2006-2008 (side-by-side column blocks are
  reshaped into one half-hourly table and screened for physical plausibility).
* Split: train on 2006-2007, test on 2008 (out-of-time validation); optional
  restriction to the wet (Dec-Apr) or dry (Jun-Sep) season.
* Baselines: each classical relation keeps its functional form but its
  constants are re-estimated on the training years (a fair benchmark).
* Discovery: any engine of `sr_benchmark.methods` (PySR, Operon, SINDy, ...)
  sees exactly the same inputs; PySR additionally returns its Pareto front.
* Scores on the test year: RMSE, R2, number of fitted constants, node count,
  AIC/BIC (Gaussian errors) and target-specific physical checks.

Usage
-----
    python src/real_discovery.py "Dados Rebio Jaru 2006 a 2008.xlsx" --methods operon sindy_poly2
    python src/real_discovery.py "Dados Rebio Jaru 2006 a 2008.xlsx" --methods pysr --season dry
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent / "sr_benchmark"))
from methods import METHODS, Unavailable, complexity  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Loading and quality control
# ---------------------------------------------------------------------------
# Column blocks of the "DADOS OK" sheet (pandas suffixes) for each year
BLOCKS = {
    2006: {"PAR": "RFA_in.1", "Rn": "R_liq.2", "RH": "UR_62m.2", "T22": "T_ar_22m.1",
           "WS": "VV_ane_62m.2", "P": "P_atm.1", "PRP": "PRP.1"},
    2007: {"PAR": "RFA_in.2", "Rn": "R_liq.3", "RH": "UR_62m.3", "T22": "T_ar_22m.2",
           "WS": "VV_ane_62m.3", "P": "P_atm.2", "PRP": "PRP.2"},
    2008: {"PAR": "RFA_in.3", "Rn": "R_liq.4", "RH": "UR_62m.4", "T22": "T_ar_22m.3",
           "T62": "T_ar_62m.1", "WS": "VV_ane_62m.4", "P": "P_atm.3", "PRP": "PRP.3"},
}
VALID = {"PAR": (0, 700), "Rn": (-150, 1100), "RH": (5, 100.5), "T22": (5, 45), "T62": (5, 45),
         "WS": (0, 25), "P": (950, 1020), "PRP": (0, 150)}


def load_jaru(xlsx: str | Path, years=(2006, 2007, 2008)) -> pd.DataFrame:
    """Reshape the yearly column blocks into one tidy half-hourly table."""
    raw = pd.read_excel(xlsx, sheet_name="DADOS OK")
    frames = []
    for year in years:
        cols = BLOCKS[year]
        n = 366 * 48 if year % 4 == 0 else 365 * 48
        block = raw[list(cols.values())].iloc[:n].copy()
        block.columns = list(cols.keys())
        block.insert(0, "year", year)
        # row i = i-th half hour of the year, first stamp 00:30 (interval end)
        block.insert(0, "timestamp", pd.Timestamp(f"{year}-01-01 00:30") + pd.to_timedelta(np.arange(n) * 30, "min"))
        frames.append(block)
    df = pd.concat(frames, ignore_index=True)
    for var, (lo, hi) in VALID.items():
        if var in df:
            df.loc[(df[var] < lo) | (df[var] > hi), var] = np.nan
    df["RH"] = df["RH"].clip(upper=100)
    df["month"] = df.timestamp.dt.month
    df["hour"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60
    df["season"] = np.select([df.month.isin([12, 1, 2, 3, 4]), df.month.isin([6, 7, 8, 9])], ["wet", "dry"], "transition")
    # daily minimum temperature: a standard proxy for the dew point in humid tropics
    df["Tmin_day"] = df.groupby(df.timestamp.dt.date)["T22"].transform("min")
    # temperature tendency (K h-1) from a Savitzky-Golay filter on gap-free runs only
    df["dTdt"] = _sg_derivative(df["T22"], dt_h=0.5, window=7, polyorder=2)
    return df


def _sg_derivative(s: pd.Series, dt_h: float, window: int, polyorder: int) -> pd.Series:
    out = pd.Series(np.nan, index=s.index)
    run_id = s.isna().cumsum()
    for _, seg in s.dropna().groupby(run_id[s.notna()]):
        if len(seg) >= window:
            out.loc[seg.index] = savgol_filter(seg.to_numpy(), window, polyorder, deriv=1, delta=dt_h)
    return out


# ---------------------------------------------------------------------------
# 2. Targets and their classical (existing) equations
# ---------------------------------------------------------------------------
def _es(t):
    """Saturation vapour pressure (hPa), Tetens/Bolton."""
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


@dataclass
class Target:
    name: str
    y: str
    features: list[str]
    baseline_name: str
    baseline: Callable                 # f(X, *params), X = array (n, len(features))
    baseline_sympy: str                # same law, with p0, p1, ... as constants
    p0: list[float]
    check: Callable[[np.ndarray], str] = lambda y: "-"
    description: str = ""


TARGETS = {
    "net_radiation": Target(
        "net_radiation", "Rn", ["PAR"],
        "linear radiation balance  Rn = a*PAR + b",
        lambda X, a, b: a * X[:, 0] + b,
        "p0*PAR + p1", [2.0, -40.0],
        check=lambda y: f"night Rn<0: {np.mean(y[:10] < 0):.0%}" if len(y) else "-",
        description="Net radiation vs incident photosynthetically active radiation",
    ),
    "relative_humidity": Target(
        "relative_humidity", "RH", ["T22", "Tmin_day"],
        "Tetens, dew point ~ a*Tmin_day + b  RH = 100*es(Td)/es(T)",
        lambda X, a, b: 100 * _es(a * X[:, 1] + b) / _es(X[:, 0]),
        "100*exp(17.67*(p0*Tmin_day + p1)/(p0*Tmin_day + p1 + 243.5) - 17.67*T22/(T22 + 243.5))",
        [1.0, -1.0],
        check=lambda y: f"RH>100: {np.mean(y > 100.5):.1%}, RH<0: {np.mean(y < 0):.1%}",
        description="Relative humidity from air temperature and a dew-point proxy",
    ),
    "temperature_tendency": Target(
        "temperature_tendency", "dTdt", ["Rn", "T22", "WS"],
        "radiative relaxation  dT/dt = a*Rn - lam*T + c",
        lambda X, a, lam, c: a * X[:, 0] - lam * X[:, 1] + c,
        "p0*Rn - p1*T22 + p2", [0.005, 0.3, 7.0],
        description="Half-hourly tendency of air temperature (22 m)",
    ),
}


# ---------------------------------------------------------------------------
# 3. Comparison
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame, target: Target, season: str | None = None, n_max: int = 3000, seed: int = 0):
    d = df.dropna(subset=[target.y, *target.features])
    if season:
        d = d[d.season == season]
    train, test = d[d.year <= 2007], d[d.year == 2008]
    if len(train) > n_max:  # SR cost grows with rows; a random subsample keeps runs short
        train = train.sample(n_max, random_state=seed)
    arr = lambda a: np.array(a.to_numpy(dtype=float), dtype=float, order="C", copy=True)  # writeable copy (pandas 3 returns read-only views)
    return arr(train[target.features]), arr(train[target.y]), arr(test[target.features]), arr(test[target.y])


def n_constants(expr: str) -> int:
    try:
        e = sp.sympify(expr)
        return sum(1 for a in sp.preorder_traversal(e) if a.is_Float or (a.is_Number and not a.is_Integer))
    except Exception:
        return -1


def scores(y, p, k) -> dict:
    p = np.asarray(p, float)
    rss = float(np.sum((y - p) ** 2))
    n = len(y)
    return {
        "rmse_test": np.sqrt(rss / n),
        "r2_test": 1 - rss / np.sum((y - y.mean()) ** 2),
        "aic_test": n * np.log(rss / n) + 2 * k,
        "bic_test": n * np.log(rss / n) + k * np.log(n),
    }


def compare(df: pd.DataFrame, target_name: str, methods: list[str], budget: int = 60,
            season: str | None = None, seed: int = 0) -> pd.DataFrame:
    """Re-fit the classical law, run each SR engine, score everything on 2008."""
    t = TARGETS[target_name]
    X, y, Xt, yt = split(df, t, season, seed=seed)
    rows = []

    popt, _ = curve_fit(lambda Xf, *p: t.baseline(Xf.reshape(-1, len(t.features)), *p),
                        X.ravel(), y, p0=t.p0, maxfev=20000)
    expr = t.baseline_sympy
    for i, v in enumerate(popt):
        expr = expr.replace(f"p{i}", f"{v:.6g}")
    pred = t.baseline(Xt, *popt)
    rows.append({"model": "existing: " + t.baseline_name, "expression": expr, "constants": len(popt),
                 "complexity": complexity(expr), "seconds": 0.0, "check": t.check(pred),
                 **scores(yt, pred, len(popt))})

    for m in methods:
        try:
            fit = METHODS[m](X, y, t.features, budget, seed)
            pred = fit.predict(Xt)
            k = max(n_constants(fit.expression), 1)
            rows.append({"model": m, "expression": fit.expression, "constants": k,
                         "complexity": complexity(fit.expression), "seconds": round(fit.seconds, 1),
                         "check": t.check(pred), **scores(yt, pred, k)})
        except Unavailable as exc:
            rows.append({"model": m, "expression": f"unavailable: {exc}"})
        except Exception as exc:
            rows.append({"model": m, "expression": f"error: {type(exc).__name__}: {str(exc)[:100]}"})
    out = pd.DataFrame(rows)
    out.insert(0, "target", target_name)
    out.insert(1, "season", season or "all")
    out.attrs["n_train"], out.attrs["n_test"] = len(y), len(yt)
    return out


def pysr_pareto(df: pd.DataFrame, target_name: str, budget: int = 120, season: str | None = None,
                seed: int = 0) -> pd.DataFrame:
    """Full PySR Pareto front with the test-year error of every candidate."""
    from pysr import PySRRegressor

    t = TARGETS[target_name]
    X, y, Xt, yt = split(df, t, season, seed=seed)
    reg = PySRRegressor(
        niterations=1000, populations=15, maxsize=25,
        binary_operators=["+", "-", "*", "/"], unary_operators=["exp", "log", "sqrt", "square"],
        timeout_in_seconds=budget, model_selection="best", random_state=seed,
        deterministic=True, parallelism="serial", progress=False, verbosity=0,
    )
    reg.fit(X, y, variable_names=t.features)
    eqs = reg.equations_.copy()
    eqs["rmse_test"] = [np.sqrt(np.mean((yt - reg.predict(Xt, index=i)) ** 2)) for i in range(len(eqs))]
    return eqs[["complexity", "loss", "rmse_test", "equation"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--methods", nargs="+", default=["pysr", "operon", "sindy_poly2"])
    ap.add_argument("--season", choices=["wet", "dry"], default=None)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("results/real_discovery.csv"))
    args = ap.parse_args()

    df = load_jaru(args.xlsx)
    print(f"Loaded {len(df)} half-hours, years {sorted(df.year.unique())}")
    results = []
    for name in args.targets:
        res = compare(df, name, args.methods, args.budget, args.season)
        print(f"\n=== {name} (train n={res.attrs['n_train']}, test 2008 n={res.attrs['n_test']}) ===")
        cols = ["model", "rmse_test", "r2_test", "constants", "complexity", "bic_test", "check", "expression"]
        print(res[[c for c in cols if c in res]].to_string(index=False, max_colwidth=90))
        results.append(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(results).to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
