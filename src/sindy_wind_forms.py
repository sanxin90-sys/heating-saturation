"""
sindy_wind_forms.py
===================

EXP-014: functional form of the wind effect on dT/dt, and two follow-ups
(methods/sindy.md §12). Registered in EXPERIMENT_LOG before running.

A. Wind form. On top of sat+harm = [1, T, 1-exp(-b SW), harmonics K=3]:
     W1  + SW*WS                     (M1 of EXP-009/010/013)
     W2  + SW*WS/(1 + WS/w0)         wind effect saturating in WS, w0 in {0.5, 1, 2, 4} m/s
     W3  + (1-exp(-b SW))*WS         wind modulates the saturated radiative term
     W4  + WS + WS*T                 wind-dependent relaxation rate, lam(WS) = lam0 + lam1*WS
   Selection (development sites Jaru and K34, their EXP-009/010 test years):
   the form (and w0) with the lowest pooled mean 24-h free-run RMSE.
   Confirmation sites: Santarem km67 (BR-Sa1) and Colosso (BR-Ma3), with the
   EXP-013 train/test years; coefficients refitted locally on the training
   years, form and w0 fixed.
   Rules for the selected form W*, at each confirmation site, against sat+harm:
     R1 weak dR2 > 0 with 95 % daily-block CI excluding 0 in all, wet and dry;
     R2 weak dR2 > 0 in each test year (season all);
     R3 pooled 24-h RMSE lower, CI excluding 0.
B. Santarem simulation without the wind-continuity requirement for the
   models that do not use WS (relax+harm, sat+harm): R3 for sat+harm vs
   relax+harm with the larger day sample. Same rule R3 as EXP-013.
C. Physical reading of b (descriptive, no rule): at each site, midday
   (10-14 h) half-hours, measured H and LE (QC = 0) fitted as
   a*(1-exp(-b_X*SW)) + c; b_H, b_LE and b_(H+LE) compared with the dT/dt b.

Usage: python src/sindy_wind_forms.py <folder with ma2/, Sa1/, Ma3/ extracted FLUXMET_HH csvs>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_weak_loyo as W  # noqa: E402
import sindy_confirm_newyears as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = W.OUT
B_GRID = C.B_GRID
W0_GRID = [0.5, 1.0, 2.0, 4.0]
M = 9

SITES = {
    "Jaru": dict(role="dev", train=[2006, 2008], test=[2005, 2009, 2010]),
    "K34": dict(role="dev", csv="data/fluxnet_BR-Ma2/BR-Ma2_HH_TA_SW_WS_1999-2016.csv", train=[2002, 2003, 2004, 2005],
                test=[2006, 2007, 2008, 2009], wet=[12, 1, 2, 3, 4, 5], dry=[7, 8, 9, 10], wind_filter=True),
    "Sa1": dict(role="confirm", csv="data/fluxnet_BR-Sa1/BR-Sa1_HH_TA_SW_WS.csv", train=[2002, 2003, 2004],
                test=[2009, 2010, 2011], wet=[12, 1, 2, 3, 4, 5], dry=[8, 9, 10, 11], wind_filter=False),
    "Ma3": dict(role="confirm", csv="data/fluxnet_BR-Ma3/BR-Ma3_HH_TA_SW_WS.csv", train=[2009, 2010, 2011],
                test=[2013, 2018], wet=[12, 1, 2, 3, 4, 5], dry=[7, 8, 9, 10], wind_filter=True),
}
FLUX = {"Jaru": "data/fluxnet_BR-Ji3/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv"}


def load_site(name):
    cfg = SITES[name]
    if name == "Jaru":
        return C.load()
    f = pd.read_csv(ROOT / cfg["csv"])
    d = pd.DataFrame({"timestamp": pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")})
    d["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0)
    d["SW"] = f.SW_IN_F_MDS.where(f.SW_IN_F_MDS_QC == 0)
    d["WS"] = f.WS.where(f.WS_F_QC == 0)
    d["year"], d["month"] = d.timestamp.dt.year, d.timestamp.dt.month
    d["hour"] = d.timestamp.dt.hour + d.timestamp.dt.minute / 60
    if cfg["wind_filter"]:
        noon = d[d.hour.between(11, 14.5)].groupby(["year", "month"]).WS.median()
        night = d[d.hour.between(0, 3.5)].groupby(["year", "month"]).WS.median()
        flat = set((noon - night)[(noon - night) <= 0].index)
        d.loc[[(y, m) in flat for y, m in zip(d.year, d.month)], "WS"] = np.nan
    d["season"] = np.select([d.month.isin(cfg["wet"]), d.month.isin(cfg["dry"])], ["wet", "dry"], "transition")
    d["day"] = d.timestamp.dt.date
    d["Rn"] = d["SW"]
    return d[d.year.isin(cfg["train"] + cfg["test"])].reset_index(drop=True)


def cols(d, model, b=None, w0=None):
    T, SW, H = (np.asarray(d[c], float) for c in ("T", "SW", "hour"))
    WS = np.asarray(d["WS"], float) if "WS" in d else np.zeros_like(T)
    one = np.ones_like(T)
    w = 2 * np.pi * H / 24
    hh = [f(k * w) for k in range(1, 4) for f in (np.cos, np.sin)]
    sat = 1 - np.exp(-b * SW) if b is not None else None
    base = {"relax+harm": [one, T, SW, *hh]}
    if model == "relax+harm":
        c = base["relax+harm"]
    else:
        c = [one, T, sat, *hh]
        if model == "W1":
            c.append(SW * WS)
        elif model == "W2":
            c.append(SW * WS / (1 + WS / w0))
        elif model == "W3":
            c.append(sat * WS)
        elif model == "W4":
            c += [WS, WS * T]
    return np.stack(c, axis=-1)


def weak_G(d, rows, Th):
    v, dv = W.phi(M)
    T = d["T"].to_numpy(float)
    return np.array([v @ Th[idx] for *_, idx in rows]), np.array([-(dv @ T[idx]) for *_, idx in rows])


def fit(d, rows, tr, model, w0=None):
    if model == "relax+harm":
        G, y = weak_G(d, rows, cols(d, model))
        return np.linalg.lstsq(G[tr], y[tr], rcond=None)[0], None
    best = None
    for b in B_GRID:
        G, y = weak_G(d, rows, cols(d, model, b, w0))
        xi = np.linalg.lstsq(G[tr], y[tr], rcond=None)[0]
        sse = np.sum((y[tr] - G[tr] @ xi) ** 2)
        if best is None or sse < best[0]:
            best = (sse, xi, b)
    return best[1], best[2]


def simulate(d, model, xi, b, w0, year, need_ws=True):
    dy = d[d.year == year].reset_index(drop=True)
    req = ["T", "SW"] + (["WS"] if need_ws else [])
    need = dy[req].notna().all(axis=1).to_numpy()
    starts = [s for s in np.flatnonzero(np.isclose(dy.hour.to_numpy(), 0.0)) if s + 48 < len(dy) and need[s:s + 49].all()]
    if not starts:
        return np.array([]), np.array([])
    idx = np.array(starts)[:, None] + np.arange(49)[None, :]
    SWa, Ta = dy.SW.to_numpy(float)[idx], dy["T"].to_numpy(float)[idx]
    WSa = dy.WS.to_numpy(float)[idx] if need_ws else np.zeros_like(SWa)
    h0 = dy.hour.to_numpy(float)[idx[:, 0]]

    def rhs(T, j, fr):
        jj = min(j + 1, 48)
        dd = {"T": T, "SW": SWa[:, j] + fr * (SWa[:, jj] - SWa[:, j]), "WS": WSa[:, j] + fr * (WSa[:, jj] - WSa[:, j]),
              "hour": (h0 + 0.5 * (j + fr)) % 24}
        return cols(dd, model, b, w0) @ xi

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


def candidates():
    out = [("relax+harm", None), ("sat+harm", None), ("W1", None), ("W3", None), ("W4", None)]
    return out + [("W2", w0) for w0 in W0_GRID]


def site_eval(name, d, cands, rng, do_weak=True):
    cfg = SITES[name]
    rows = W.weak_rows(d, M)
    ry = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows]); rd = np.array([r[2] for r in rows])
    tr_all, te_all = np.isin(ry, cfg["train"]), np.isin(ry, cfg["test"])
    weak, sims = [], []
    fits = {c: fit(d, rows, tr_all, c[0], c[1]) for c in cands}
    for c in cands:
        xi, b = fits[c]
        for ty in cfg["test"]:
            days, rm = simulate(d, c[0], xi, b, c[1], ty)
            sims += [{"site": name, "model": c[0], "w0": c[1], "test_year": ty, "day": x, "rmse": r} for x, r in zip(days, rm)]
    if do_weak:
        y = weak_G(d, rows, cols(d, "relax+harm"))[1]
        for season in ["all", "wet", "dry"]:
            sm = np.ones(len(rows), bool) if season == "all" else rs == season
            tr, te = tr_all & sm, te_all & sm
            P = {}
            for c in cands:
                xi, b = fit(d, rows, tr, c[0], c[1])
                P[c] = weak_G(d, rows, cols(d, c[0], b, c[1]))[0] @ xi
            ref = P[("sat+harm", None)]
            for c in cands:
                lo, hi = W.boot_ci(y[te], P[c][te], ref[te], rd[te], rng) if c != ("sat+harm", None) else (0, 0)
                row = {"site": name, "season": season, "model": c[0], "w0": c[1], "r2": W.r2(y[te], P[c][te]),
                       "dR2_vs_sat": W.r2(y[te], P[c][te]) - W.r2(y[te], ref[te]), "ci_lo": lo, "ci_hi": hi}
                for ty in cfg["test"]:
                    t1 = te & (ry == ty)
                    row[f"dR2_{ty}"] = W.r2(y[t1], P[c][t1]) - W.r2(y[t1], ref[t1])
                weak.append(row)
    return pd.DataFrame(weak), pd.DataFrame(sims), fits


def sim_summary(sims, rng, ref=("sat+harm", None)):
    sims = sims.copy(); sims["key"] = [(m, None if pd.isna(w) else float(w)) for m, w in zip(sims.model, sims.w0)]
    out = []
    for site, g in sims.groupby("site"):
        piv = g.pivot_table(index=["test_year", "day"], columns="key", values="rmse")
        for k in piv.columns:
            ok = piv[k].notna() & piv[ref].notna()
            diff = (piv[k] - piv[ref])[ok].to_numpy()
            bs = [np.mean(rng.choice(diff, len(diff))) for _ in range(1000)] if len(diff) else [np.nan]
            out.append({"site": site, "model": k[0], "w0": k[1], "n_days": int(ok.sum()), "mean_rmse": piv[k][ok].mean(),
                        f"dRMSE_vs_{ref[0]}": diff.mean() if len(diff) else np.nan,
                        "ci_lo": np.nanpercentile(bs, 2.5), "ci_hi": np.nanpercentile(bs, 97.5)})
    return pd.DataFrame(out)


def part_c(name):
    """b of H, LE at midday from the raw FLUXNET files (scratch copies or data/)."""
    raw = {"Jaru": ROOT / FLUX["Jaru"]}
    base = Path(sys.argv[1] if len(sys.argv) > 1 else ".")  # folder with the extracted raw FLUXMET_HH files
    p = raw.get(name) or base / {"K34": "ma2/AMF_BR-Ma2_FLUXNET_FLUXMET_HH_1999-2016_v1.3_r1.csv",
                                 "Sa1": "Sa1/AMF_BR-Sa1_FLUXNET_FLUXMET_HH_2002-2011_v1.3_r1.csv",
                                 "Ma3": "Ma3/AMF_BR-Ma3_FLUXNET_FLUXMET_HH_2008-2018_v1.3_r1.csv"}[name]
    if not p.exists():
        return {"site": name, "note": f"missing {p}"}
    from scipy.optimize import curve_fit
    f = pd.read_csv(p, na_values=[-9999], usecols=["TIMESTAMP_END", "SW_IN_F_MDS", "SW_IN_F_MDS_QC", "H_F_MDS", "H_F_MDS_QC",
                                                   "LE_F_MDS", "LE_F_MDS_QC"])
    hr = (f.TIMESTAMP_END % 10000) // 100
    sel = hr.between(10, 14) & (f.SW_IN_F_MDS_QC == 0)
    out = {"site": name}
    for lab, y, q in [("H", f.H_F_MDS, f.H_F_MDS_QC == 0), ("LE", f.LE_F_MDS, f.LE_F_MDS_QC == 0),
                      ("H+LE", f.H_F_MDS + f.LE_F_MDS, (f.H_F_MDS_QC == 0) & (f.LE_F_MDS_QC == 0))]:
        s = sel & q & y.notna()
        x, yy = f.SW_IN_F_MDS[s].to_numpy(float), y[s].to_numpy(float)
        try:
            p_, _ = curve_fit(lambda x, a, b, c: a * (1 - np.exp(-b * x)) + c, x, yy, p0=[500, 0.002, 0], maxfev=20000)
            out[f"b_{lab}"] = p_[1]; out[f"n_{lab}"] = int(s.sum())
            lin = np.polyfit(x, yy, 1); r_sat = yy - (p_[0] * (1 - np.exp(-p_[1] * x)) + p_[2]); r_lin = yy - np.polyval(lin, x)
            out[f"dR2_sat_vs_lin_{lab}"] = (np.sum(r_lin ** 2) - np.sum(r_sat ** 2)) / np.sum((yy - yy.mean()) ** 2)
        except Exception as e:
            out[f"b_{lab}"] = np.nan
    return out


def main():
    rng = np.random.default_rng(0)
    data = {n: load_site(n) for n in SITES}
    cands = candidates()

    # A1. selection on development sites (simulation only)
    dev_sims = []
    for n in ["Jaru", "K34"]:
        _, s, _ = site_eval(n, data[n], cands, rng, do_weak=False)
        dev_sims.append(s)
    dev = sim_summary(pd.concat(dev_sims), rng)
    print("A1. development sites, 24-h RMSE vs sat+harm\n", dev.round(4).to_string(index=False))
    pooled = pd.concat(dev_sims).groupby(["model", "w0"], dropna=False).rmse.mean().sort_values()
    print("\npooled mean RMSE (Jaru + K34)\n", pooled.round(4).to_string())
    wind = pooled[[m.startswith("W") for m, _ in pooled.index]]
    sel = wind.index[0]
    sel = (sel[0], None if pd.isna(sel[1]) else float(sel[1]))
    print("\nSELECTED wind form:", sel)

    # A2. confirmation sites
    conf_weak, conf_sims = [], []
    keep = [("relax+harm", None), ("sat+harm", None), ("W1", None), sel]
    for n in ["Sa1", "Ma3"]:
        wk, s, _ = site_eval(n, data[n], keep, rng)
        conf_weak.append(wk); conf_sims.append(s)
    cw = pd.concat(conf_weak); cs = sim_summary(pd.concat(conf_sims), rng)
    print("\nA2. confirmation sites, weak dR2 vs sat+harm\n", cw.round(4).to_string(index=False))
    print("\nA2. confirmation sites, 24-h RMSE vs sat+harm\n", cs.round(4).to_string(index=False))
    verdict = {}
    for n in ["Sa1", "Ma3"]:
        g = cw[(cw.site == n) & (cw.model == sel[0]) & ((cw.w0 == sel[1]) if sel[1] is not None else cw.w0.isna())]
        cfg = SITES[n]
        r3 = cs[(cs.site == n) & (cs.model == sel[0]) & ((cs.w0 == sel[1]) if sel[1] is not None else cs.w0.isna())]
        verdict[n] = {"R1": bool((g.ci_lo > 0).all()),
                      "R2": bool((g[g.season == "all"][[f"dR2_{t}" for t in cfg["test"]]] > 0).all(axis=None)),
                      "R3": bool(r3.ci_hi.iloc[0] < 0)}
    print("\nVERDICT (selected form vs sat+harm):", verdict)

    # B. Santarem simulation, no WS requirement
    d = data["Sa1"]; rows = W.weak_rows(d.assign(WS=d.WS.fillna(1.0)), M)  # rows only need T, SW for these models
    ry = np.array([r[0] for r in rows]); tr = np.isin(ry, SITES["Sa1"]["train"])
    dB = d.assign(WS=d.WS.fillna(1.0))
    simsB = []
    for m in ["relax+harm", "sat+harm"]:
        xi, b = fit(dB, rows, tr, m)
        for ty in SITES["Sa1"]["test"]:
            days, rm = simulate(dB, m, xi, b, None, ty, need_ws=False)
            simsB += [{"site": "Sa1_noWSreq", "model": m, "w0": None, "test_year": ty, "day": x, "rmse": r} for x, r in zip(days, rm)]
    sB = sim_summary(pd.DataFrame(simsB), rng, ref=("relax+harm", None))
    print("\nB. Santarem 24-h RMSE, no WS requirement (vs relax+harm)\n", sB.round(4).to_string(index=False))

    # C. b of H and LE
    cC = pd.DataFrame([part_c(n) for n in SITES])
    bdt = {}
    for n in SITES:
        rows = W.weak_rows(data[n], M); ry = np.array([r[0] for r in rows])
        bdt[n] = fit(data[n], rows, np.isin(ry, SITES[n]["train"]), "sat+harm")[1]
    cC["b_dTdt"] = cC.site.map(bdt)
    print("\nC. midday saturation of measured fluxes vs dT/dt\n", cC.round(4).to_string(index=False))

    dev.to_csv(OUT / "sindy_wind_dev_sim.csv", index=False)
    cw.to_csv(OUT / "sindy_wind_confirm_weak.csv", index=False)
    cs.to_csv(OUT / "sindy_wind_confirm_sim.csv", index=False)
    sB.to_csv(OUT / "sindy_wind_Sa1_sim_noWS.csv", index=False)
    cC.to_csv(OUT / "sindy_b_fluxes.csv", index=False)


if __name__ == "__main__":
    main()
