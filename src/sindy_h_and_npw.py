"""
sindy_h_and_npw.py
==================

EXP-015: boundary-layer reading of the saturation (methods/sindy.md §12 C).
If dT/dt saturates in SW because the heated layer deepens on sunnier days,
it should also saturate in the measured sensible heat flux H, since H is
linear in SW. If the saturation sat in the energy partition instead, it
would disappear with H as the forcing.
    relaxH+harm = [1, T, H, harmonics];  satH+harm = [1, T, 1-exp(-bH*H), harmonics]
    bH grid 0.001 ... 0.05 (W m-2)^-1 (H spans ~ -50 ... 400 W m-2).
Rows: weak-form windows with T and H (QC = 0) gap-free; train/test years as
in EXP-009/010/013 (Npw as in EXP-016). No simulation (H has too many gaps).
Rule (fixed before running): the boundary-layer reading is supported if
satH+harm > relaxH+harm with weak dR2 CI > 0 (season all) at all three forest
sites (Jaru, K34, Sa1); contradicted if it fails at all three.

EXP-016: fifth site, Northern Pantanal Wetland (BR-Npw; -16.50, -56.41; IGBP
WSA, Aw; non-forest, outside the Amazon). Train 2014-2015, test 2016-2017;
wet Nov-Mar, dry May-Sep. sat+harm vs relax+harm with SW forcing, rules R1-R3
of EXP-013; the simulation needs only T and SW (as EXP-014 B). Prediction from
the forest / non-forest contrast (Hypothesis): R1-R2 pass, R3 fails.

Usage: python src/sindy_h_and_npw.py <folder with ma2/, Sa1/, Ma3/, Npw/ extracted FLUXMET_HH csvs>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_weak_loyo as W  # noqa: E402
import sindy_wind_forms as F  # noqa: E402

F.SITES["Npw"] = dict(role="confirm", csv="data/fluxnet_BR-Npw/BR-Npw_HH_TA_SW_WS.csv", train=[2014, 2015],
                      test=[2016, 2017], wet=[11, 12, 1, 2, 3], dry=[5, 6, 7, 8, 9], wind_filter=False)
RAW = {"Jaru": F.ROOT / F.FLUX["Jaru"], "K34": "ma2/AMF_BR-Ma2_FLUXNET_FLUXMET_HH_1999-2016_v1.3_r1.csv",
       "Sa1": "Sa1/AMF_BR-Sa1_FLUXNET_FLUXMET_HH_2002-2011_v1.3_r1.csv",
       "Ma3": "Ma3/AMF_BR-Ma3_FLUXNET_FLUXMET_HH_2008-2018_v1.3_r1.csv",
       "Npw": "Npw/AMF_BR-Npw_FLUXNET_FLUXMET_HH_2013-2017_v1.3_r1.csv"}
B_H = [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05]
FOREST = ["Jaru", "K34", "Sa1"]


def with_H(name, d, base):
    p = RAW[name] if name == "Jaru" else Path(base) / RAW[name]
    f = pd.read_csv(p, na_values=[-9999], usecols=["TIMESTAMP_END", "H_F_MDS", "H_F_MDS_QC"])
    f["timestamp"] = pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")
    f["H"] = f.H_F_MDS.where(f.H_F_MDS_QC == 0)
    return d.merge(f[["timestamp", "H"]], on="timestamp", how="left")


def exp015(data, rng):
    out = []
    for n, d in data.items():
        cfg = F.SITES[n]
        dh = d.assign(SW_orig=d.SW, SW=d.H, Rn=d.H, WS=1.0)  # weak_rows needs T, Rn, WS
        rows = W.weak_rows(dh, F.M)
        ry = np.array([r[0] for r in rows]); rd = np.array([r[2] for r in rows])
        tr, te = np.isin(ry, cfg["train"]), np.isin(ry, cfg["test"])
        if tr.sum() < 50 or te.sum() < 50:
            out.append({"site": n, "note": f"too few rows (train {tr.sum()}, test {te.sum()})"}); continue
        G0, y = F.weak_G(dh, rows, F.cols(dh, "relax+harm"))
        xr = np.linalg.lstsq(G0[tr], y[tr], rcond=None)[0]
        best = None
        for b in B_H:
            G, _ = F.weak_G(dh, rows, F.cols(dh, "sat+harm", b))
            xi = np.linalg.lstsq(G[tr], y[tr], rcond=None)[0]
            sse = np.sum((y[tr] - G[tr] @ xi) ** 2)
            if best is None or sse < best[0]:
                best = (sse, G @ xi, b, xi)
        pr, ps = G0 @ xr, best[1]
        lo, hi = W.boot_ci(y[te], ps[te], pr[te], rd[te], rng)
        # same rows with SW forcing, for reference
        ds = dh.assign(SW=dh.SW_orig)
        okSW = np.array([np.isfinite(ds.SW.to_numpy()[idx]).all() for *_, idx in rows])
        G1, _ = F.weak_G(ds, rows, F.cols(ds, "relax+harm"))
        tr1, te1 = tr & okSW, te & okSW
        x1 = np.linalg.lstsq(G1[tr1], y[tr1], rcond=None)[0]
        bs = None
        for b in F.B_GRID:
            G2, _ = F.weak_G(ds, rows, F.cols(ds, "sat+harm", b))
            xi = np.linalg.lstsq(G2[tr1], y[tr1], rcond=None)[0]
            sse = np.sum((y[tr1] - G2[tr1] @ xi) ** 2)
            if bs is None or sse < bs[0]:
                bs = (sse, G2 @ xi, b)
        out.append({"site": n, "n_train": int(tr.sum()), "n_test": int(te.sum()), "bH": best[2],
                    "A_H": best[3][2], "r2_relaxH": W.r2(y[te], pr[te]), "r2_satH": W.r2(y[te], ps[te]),
                    "dR2_H": W.r2(y[te], ps[te]) - W.r2(y[te], pr[te]), "ci_lo": lo, "ci_hi": hi,
                    "dR2_SW_same_rows": W.r2(y[te1], bs[1][te1]) - W.r2(y[te1], (G1 @ x1)[te1]), "bSW": bs[2],
                    "satH_over_A_H": best[3][2] * best[2]})
    return pd.DataFrame(out)


def exp016(d, rng):
    cfg = F.SITES["Npw"]
    dB = d.assign(WS=d.WS.fillna(1.0))
    rows = W.weak_rows(dB, F.M)
    ry = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows]); rd = np.array([r[2] for r in rows])
    y = F.weak_G(dB, rows, F.cols(dB, "relax+harm"))[1]
    res = []
    for season in ["all", "wet", "dry"]:
        sm = np.ones(len(rows), bool) if season == "all" else rs == season
        tr, te = np.isin(ry, cfg["train"]) & sm, np.isin(ry, cfg["test"]) & sm
        P = {}
        for m in ["relax+harm", "sat+harm"]:
            xi, b = F.fit(dB, rows, tr, m)
            P[m] = (F.weak_G(dB, rows, F.cols(dB, m, b))[0] @ xi, b)
        lo, hi = W.boot_ci(y[te], P["sat+harm"][0][te], P["relax+harm"][0][te], rd[te], rng)
        row = {"season": season, "n_test": int(te.sum()), "b": P["sat+harm"][1],
               "r2_relax": W.r2(y[te], P["relax+harm"][0][te]), "dR2": W.r2(y[te], P["sat+harm"][0][te]) - W.r2(y[te], P["relax+harm"][0][te]),
               "ci_lo": lo, "ci_hi": hi}
        for ty in cfg["test"]:
            t1 = te & (ry == ty)
            row[f"dR2_{ty}"] = (W.r2(y[t1], P["sat+harm"][0][t1]) - W.r2(y[t1], P["relax+harm"][0][t1])) if t1.sum() > 20 else np.nan
        res.append(row)
    res = pd.DataFrame(res)
    tr = np.isin(ry, cfg["train"])
    sims = []
    for m in ["relax+harm", "sat+harm"]:
        xi, b = F.fit(dB, rows, tr, m)
        for ty in cfg["test"]:
            days, rm = F.simulate(dB, m, xi, b, None, ty, need_ws=False)
            sims += [{"site": "Npw", "model": m, "w0": None, "test_year": ty, "day": x, "rmse": r} for x, r in zip(days, rm)]
    sim = F.sim_summary(pd.DataFrame(sims), rng, ref=("relax+harm", None))
    r = res.set_index("season")
    verdict = {"R1": bool((res.ci_lo > 0).all()),
               "R2": bool(all(r.loc["all", f"dR2_{t}"] > 0 for t in cfg["test"])),
               "R3": bool(sim[sim.model == "sat+harm"].ci_hi.iloc[0] < 0)}
    return res, sim, verdict


def main():
    base = sys.argv[1]
    rng = np.random.default_rng(0)
    data = {n: with_H(n, F.load_site(n), base) for n in ["Jaru", "K34", "Sa1", "Ma3", "Npw"]}
    r15 = exp015(data, rng)
    print("EXP-015: H as forcing (weak R2 on test years, season all)\n", r15.round(4).to_string(index=False))
    f = r15[r15.site.isin(FOREST)]
    sup = bool((f.ci_lo > 0).all()); con = bool((f.ci_hi <= 0).all() | (f.ci_lo <= 0).all())
    print("EXP-015 VERDICT: supported =", sup, "| all forest CIs touch/below zero =", bool((f.ci_lo <= 0).all()))
    res, sim, v = exp016(data["Npw"], rng)
    print("\nEXP-016: BR-Npw sat+harm vs relax+harm (weak)\n", res.round(4).to_string(index=False))
    print("\nEXP-016: 24-h simulation\n", sim.round(4).to_string(index=False))
    print("EXP-016 VERDICT:", v)
    r15.to_csv(F.OUT / "sindy_h_forcing.csv", index=False)
    res.to_csv(F.OUT / "sindy_Npw_weak.csv", index=False)
    sim.to_csv(F.OUT / "sindy_Npw_sim.csv", index=False)


if __name__ == "__main__":
    main()
