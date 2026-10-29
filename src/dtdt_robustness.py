"""
dtdt_robustness.py
==================

Robustness test for the candidate dT/dt relation (saturating response to net
radiation) found by Operon in real_discovery.py.

Question: does the saturating Rn term improve out-of-year prediction of the
22 m temperature tendency beyond (a) the linear relaxation law and (b) a
purely diurnal baseline (hour-of-day harmonics)?

Design: train 2006-2007, test 2008 (as in real_discovery.py); seasons all/wet/dry;
5 training-subsample seeds for the SR engines; 95% CIs from a daily block
bootstrap of the 2008 test set.

Decision rule (fixed before running): the saturation is supported only if
sat+harmonics beats relaxation+harmonics with a bootstrap CI on dR2 that
excludes zero, in all three season settings.

Usage:  python src/dtdt_robustness.py "Dados Rebio Jaru 2006 a 2008.xlsx" [all|exclude|train2006]

Training-set variants (data-quality audit, EXPERIMENT_LOG EXP-002):
  all        - every valid half-hour of 2006-2007 (original EXP-001)
  exclude    - drop months whose median Rn/PAR (PAR > 50) is inconsistent with
               the other years: 2006-10, 2007-05, 2007-06, 2007-09..2007-12.
               Rows are excluded, never corrected.
  train2006  - train on 2006 only, without 2006-10
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_discovery import load_jaru, METHODS, complexity  # noqa: E402

FEATS = ["Rn", "T22", "WS", "hour"]
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = 3000
N_BOOT = 1000


def harm(h, K=3):
    w = 2 * np.pi * h / 24
    return np.column_stack([f(k * w) for k in range(1, K + 1) for f in (np.cos, np.sin)])


# parametric models: f(X, *p) with X columns = FEATS
def relax(X, a, lam, c):
    return a * X[:, 0] - lam * X[:, 1] + c


def sat(X, A, b, lam, c):
    return A * (1 - np.exp(-b * X[:, 0])) - lam * X[:, 1] + c


PARAMETRIC = {"relaxation": (relax, [0.002, 0.05, 1.2]), "saturating": (sat, [3.0, 0.005, 0.05, 1.2])}


def fit_parametric(name, X, y, Xt, with_harm):
    f, p0 = PARAMETRIC[name]
    if not with_harm:
        p, _ = curve_fit(lambda Xf, *q: f(Xf.reshape(-1, X.shape[1]), *q), X.ravel(), y, p0=p0, maxfev=40000)
        return f(Xt, *p), len(p)
    H, Ht = harm(X[:, 3]), harm(Xt[:, 3])
    nh = H.shape[1]

    def g(Z, *q):
        Z = Z.reshape(-1, X.shape[1] + nh)
        return f(Z[:, :4], *q[: len(p0)]) + Z[:, 4:] @ np.array(q[len(p0):])
    Z, Zt = np.hstack([X, H]), np.hstack([Xt, Ht])
    p, _ = curve_fit(g, Z.ravel(), y, p0=list(p0) + [0.0] * nh, maxfev=40000)
    return g(Zt.ravel(), *p), len(p)


def fit_harmonic_only(X, y, Xt):
    A = np.column_stack([np.ones(len(X)), harm(X[:, 3])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.column_stack([np.ones(len(Xt)), harm(Xt[:, 3])]) @ beta, A.shape[1]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def block_boot(y, preds: dict, days, ref: str, rng):
    """Daily block bootstrap of R2 and of R2 differences against `ref`."""
    ud = np.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in ud}
    out = {k: [] for k in preds}
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_day[d] for d in rng.choice(ud, len(ud))])
        r_ref = r2(y[idx], preds[ref][idx])
        for k, p in preds.items():
            out[k].append((r2(y[idx], p[idx]), r2(y[idx], p[idx]) - r_ref))
    return {k: np.array(v) for k, v in out.items()}


SUSPECT_RN = {(2006, 10), (2007, 5), (2007, 6), (2007, 9), (2007, 10), (2007, 11), (2007, 12)}


def main(xlsx, variant="all"):
    df = load_jaru(xlsx)
    d = df.dropna(subset=["dTdt", "Rn", "T22", "WS"])
    if variant in ("exclude", "train2006"):
        bad = [(y, m) in SUSPECT_RN for y, m in zip(d.year, d.month)]
        d = d[~np.array(bad)]
    if variant == "train2006":
        d = d[d.year != 2007]
    rng = np.random.default_rng(0)
    rows = []
    for season in ["all", "wet", "dry"]:
        ds = d if season == "all" else d[d.season == season]
        tr_full, te = ds[ds.year <= 2007], ds[ds.year == 2008]
        Xt, yt = te[FEATS].to_numpy(float), te["dTdt"].to_numpy(float)
        days = te.timestamp.dt.dayofyear.to_numpy()
        preds, meta = {}, {}
        for seed in SEEDS:
            tr = tr_full.sample(min(N_TRAIN, len(tr_full)), random_state=seed)
            X, y = tr[FEATS].to_numpy(float), tr["dTdt"].to_numpy(float)
            cands = {
                "harmonic_only": lambda: fit_harmonic_only(X, y, Xt),
                "relaxation": lambda: fit_parametric("relaxation", X, y, Xt, False),
                "saturating": lambda: fit_parametric("saturating", X, y, Xt, False),
                "relaxation+harm": lambda: fit_parametric("relaxation", X, y, Xt, True),
                "saturating+harm": lambda: fit_parametric("saturating", X, y, Xt, True),
            }
            for k, fn in cands.items():
                p, npar = fn()
                preds[f"{k}|s{seed}"], meta[f"{k}|s{seed}"] = p, (npar, "")
            # SR engines on the original three physical inputs (no hour), as in real_discovery.py
            for m in ["operon", "sindy_poly2"]:
                try:
                    fit = METHODS[m](np.ascontiguousarray(X[:, :3]), y.copy(), FEATS[:3], 30, seed)
                    preds[f"{m}|s{seed}"] = fit.predict(np.ascontiguousarray(Xt[:, :3]))
                    meta[f"{m}|s{seed}"] = (complexity(fit.expression), fit.expression)
                except Exception as exc:  # report, do not hide
                    print(f"{season} {m} seed {seed}: {type(exc).__name__}: {exc}")
            print(f"{season}: seed {seed} done", flush=True)
        boot = block_boot(yt, preds, days, "relaxation+harm|s0", rng)
        for k, p in preds.items():
            b = boot[k]
            model, seed = k.split("|")
            rows.append({"season": season, "model": model, "seed": int(seed[1:]), "n_train": min(N_TRAIN, len(tr_full)),
                         "n_test": len(yt), "r2_test": r2(yt, p), "rmse_test": float(np.sqrt(np.mean((yt - p) ** 2))),
                         "r2_ci_lo": np.percentile(b[:, 0], 2.5), "r2_ci_hi": np.percentile(b[:, 0], 97.5),
                         "dr2_vs_relax_harm_s0": float(np.mean(b[:, 1])),
                         "dr2_ci_lo": np.percentile(b[:, 1], 2.5), "dr2_ci_hi": np.percentile(b[:, 1], 97.5),
                         "n_params_or_complexity": meta[k][0], "expression": meta[k][1]})
    out = pd.DataFrame(rows)
    path = Path(__file__).resolve().parent / "sr_benchmark" / "results" / (f"dtdt_robustness_{variant}.csv" if variant != "all" else "dtdt_robustness.csv")
    out.to_csv(path, index=False)
    summ = out.groupby(["season", "model"]).agg(r2_mean=("r2_test", "mean"), r2_sd=("r2_test", "std"),
                                                 dr2_lo=("dr2_ci_lo", "mean"), dr2_hi=("dr2_ci_hi", "mean")).round(3)
    print(summ.to_string())
    print(f"Saved {path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "all")
