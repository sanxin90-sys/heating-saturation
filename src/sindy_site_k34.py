"""
sindy_site_k34.py
=================

EXP-010: does the dT/dt structure confirmed at Rebio Jaru (EXP-009) hold at a
second Amazon forest site, Manaus ZF2 K34 (AmeriFlux/FLUXNET BR-Ma2,
-2.609, -60.209, canopy 35 m, UTC-4)?

    M1:  dT/dt = c - lam*T + A*(1 - exp(-b*SW)) + kappa*SW*WS + diurnal harmonics (K = 3)

Data: data/fluxnet_BR-Ma2/BR-Ma2_HH_TA_SW_WS_1999-2016.csv (columns copied
from AMF_BR-Ma2_FLUXNET_FLUXMET_HH_1999-2016_v1.3_r1.csv, CC-BY-4.0, from the
user's Downloads). TA, SW_IN and WS with QC = 0. Months whose median wind has
no diurnal cycle (noon minus night median <= 0) are dropped, the same rule as
2009-12 at Jaru. Seasons for Manaus: wet = Dec-May, dry = Jul-Oct.

Part 1 (primary, rules R1-R4 of EXP-009, unchanged): structure refitted at
K34. Train 2002-2005, test 2006-2009 (four full years), no refit on test.
Part 2 (secondary, descriptive, no rule): transfer of the Jaru coefficients
(trained on Jaru 2006 + 2008, EXP-009) to K34 2006-2009 without any refit.

Usage: python src/sindy_site_k34.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_weak_loyo as W  # noqa: E402
import sindy_confirm_newyears as C  # noqa: E402

CSV = Path(__file__).resolve().parents[1] / "data/fluxnet_BR-Ma2/BR-Ma2_HH_TA_SW_WS_1999-2016.csv"
TRAIN, TEST = [2002, 2003, 2004, 2005], [2006, 2007, 2008, 2009]
MODELS = C.MODELS
OUT = W.OUT


def load_k34():
    f = pd.read_csv(CSV)
    d = pd.DataFrame({"timestamp": pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")})
    d["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0)
    d["SW"] = f.SW_IN_F_MDS.where(f.SW_IN_F_MDS_QC == 0)
    d["WS"] = f.WS.where(f.WS_F_QC == 0)
    d["year"], d["month"] = d.timestamp.dt.year, d.timestamp.dt.month
    d["hour"] = d.timestamp.dt.hour + d.timestamp.dt.minute / 60
    noon = d[d.hour.between(11, 14.5)].groupby(["year", "month"]).WS.median()
    night = d[d.hour.between(0, 3.5)].groupby(["year", "month"]).WS.median()
    flat = set((noon - night)[(noon - night) <= 0].index)
    d.loc[[(y, m) in flat for y, m in zip(d.year, d.month)], "WS"] = np.nan
    d["season"] = np.select([d.month.isin([12, 1, 2, 3, 4, 5]), d.month.isin([7, 8, 9, 10])], ["wet", "dry"], "transition")
    d["day"] = d.timestamp.dt.date
    d["Rn"] = d["SW"]
    return d[d.year.between(2002, 2009)].reset_index(drop=True), sorted(flat)


def main():
    d, flat = load_k34()
    print("months dropped for flat wind cycle:", [x for x in flat if 2002 <= x[0] <= 2009])
    rng = np.random.default_rng(0)
    rows = W.weak_rows(d, C.M)
    ry = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows]); rd = np.array([r[2] for r in rows])
    print("weak rows per year:", pd.Series(ry).value_counts().sort_index().to_dict())
    tr_all, te_all = np.isin(ry, TRAIN), np.isin(ry, TEST)

    res, coefs = [], []
    for season in ["all", "wet", "dry"]:
        sm = np.ones(len(rows), bool) if season == "all" else rs == season
        tr, te = tr_all & sm, te_all & sm
        P = {}
        for m in MODELS:
            xi, b = C.fit(d, rows, tr, m)
            G, y = C.weak_G(d, rows, C.cols(d, m, b))
            P[m] = G @ xi
            if season == "all" and m == "sat+wind+harm":
                coefs.append({"fit": "K34 train 2002-05", "b": b, "lam": -xi[1], "A": xi[2], "kappa": xi[3]})
        y = C.weak_G(d, rows, C.cols(d, "harm"))[1]
        for m in MODELS:
            lo, hi = W.boot_ci(y[te], P[m][te], P["relax+harm"][te], rd[te], rng) if m != "relax+harm" else (0, 0)
            row = {"season": season, "model": m, "n_test": int(te.sum()), "r2_test": W.r2(y[te], P[m][te]),
                   "dR2": W.r2(y[te], P[m][te]) - W.r2(y[te], P["relax+harm"][te]), "ci_lo": lo, "ci_hi": hi}
            for ty in TEST:
                t1 = te & (ry == ty)
                row[f"dR2_{ty}"] = W.r2(y[t1], P[m][t1]) - W.r2(y[t1], P["relax+harm"][t1]) if t1.sum() > 20 else np.nan
            res.append(row)
    res = pd.DataFrame(res)
    print("\nPart 1, R1/R2: K34, train 2002-05, test 2006-09\n", res.round(4).to_string(index=False))

    for ty in TEST:
        xi, b = C.fit(d, rows, ry == ty, "sat+wind+harm")
        coefs.append({"fit": f"K34 refit {ty} alone", "b": b, "lam": -xi[1], "A": xi[2], "kappa": xi[3]})

    sims = []
    fits = {m: C.fit(d, rows, tr_all, m) for m in MODELS}
    for m in MODELS:
        for ty in TEST:
            days, rm = C.simulate(d, m, *fits[m], ty)
            sims += [{"model": m, "test_year": ty, "day": x, "rmse": r} for x, r in zip(days, rm)]
    sims = pd.DataFrame(sims)
    piv = sims.pivot_table(index=["test_year", "day"], columns="model", values="rmse")
    sim_out = []
    for m in MODELS:
        ok = piv[m].notna() & piv["relax+harm"].notna()
        diff = (piv[m] - piv["relax+harm"])[ok].to_numpy()
        bs = [np.mean(rng.choice(diff, len(diff))) for _ in range(1000)]
        sim_out.append({"model": m, "n_days": int(ok.sum()), "blowups": int(piv[m].isna().sum()), "mean_rmse": piv[m][ok].mean(),
                        "dRMSE": diff.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
    sim_out = pd.DataFrame(sim_out)
    print("\nPart 1, R3: 24-h free runs, K34 test years pooled\n", sim_out.round(4).to_string(index=False))

    # Part 2: Jaru coefficients transferred without refit
    dj = C.load()
    rj = W.weak_rows(dj, C.M)
    rjy = np.array([r[0] for r in rj])
    tj = np.isin(rjy, C.TRAIN)
    y = C.weak_G(d, rows, C.cols(d, "harm"))[1]
    tr2 = []
    for m in ["relax+harm", "sat+harm", "sat+wind+harm"]:
        xi, b = C.fit(dj, rj, tj, m)
        G, _ = C.weak_G(d, rows, C.cols(d, m, b))
        p = G @ xi
        tr2.append({"model": m, "r2_K34_test_jaru_coefs": W.r2(y[te_all], p[te_all]),
                    "r2_K34_test_local_fit": W.r2(y[te_all], (C.weak_G(d, rows, C.cols(d, m, fits[m][1]))[0] @ fits[m][0])[te_all])})
        if m == "sat+wind+harm":
            coefs.append({"fit": "Jaru train 2006+2008", "b": b, "lam": -xi[1], "A": xi[2], "kappa": xi[3]})
    tr2 = pd.DataFrame(tr2)
    coefs = pd.DataFrame(coefs)
    print("\nR4 / coefficients\n", coefs.to_string(index=False))
    print("\nPart 2, transfer Jaru -> K34 (no refit), weak R2 on K34 2006-09\n", tr2.round(4).to_string(index=False))

    m1 = res[res.model == "sat+wind+harm"]
    verdict = {
        "R1": bool((m1.ci_lo > 0).all()),
        "R2": bool((m1[m1.season == "all"][[f"dR2_{t}" for t in TEST]] > 0).all(axis=None)),
        "R3": bool(sim_out[sim_out.model == "sat+wind+harm"].ci_hi.iloc[0] < 0),
        "R4": bool((coefs[coefs.fit.str.startswith("K34")].kappa < 0).all()),
    }
    print("\nVERDICT", verdict)
    res.to_csv(OUT / "sindy_k34_weak.csv", index=False)
    coefs.to_csv(OUT / "sindy_k34_coefs.csv", index=False)
    sim_out.to_csv(OUT / "sindy_k34_sim_summary.csv", index=False)
    tr2.to_csv(OUT / "sindy_k34_transfer.csv", index=False)


if __name__ == "__main__":
    main()
