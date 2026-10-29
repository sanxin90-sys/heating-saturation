"""
make_figures.py
===============

Figures and tables of the manuscript, built only from the CSV outputs of the
experiment scripts (src/sr_benchmark/results/) plus, for Figures 2 and 6a,
the same FLUXNET files and loaders those scripts use.

Usage: python src/make_figures.py [RAW]
  RAW = folder with ma2/, Sa1/, Ma3/, Npw/ holding the extracted FLUXMET_HH csvs
        (needed only for Figure 6a, midday H against SW). Without it, 6a is skipped.
Outputs: figures/fig{2..6}.{pdf,png}, figures/table{1..3}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_weak_loyo as W  # noqa: E402
import sindy_wind_forms as F  # noqa: E402
import sindy_h_and_npw as HN  # noqa: E402  (registers Npw in F.SITES)
from real_discovery import _sg_derivative  # noqa: E402

R = W.OUT
FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
SITES = ["Jaru", "K34", "Sa1", "Ma3", "Npw"]
LABEL = {"Jaru": "Jaru (BR-Ji3)", "K34": "K34 (BR-Ma2)", "Sa1": "km67 (BR-Sa1)",
         "Ma3": "Colosso (BR-Ma3)", "Npw": "Pantanal (BR-Npw)"}
FOREST = {"Jaru": True, "K34": True, "Sa1": True, "Ma3": False, "Npw": False}
# Okabe-Ito colours + distinct markers (colour is never the only cue)
COL = {"Jaru": "#0072B2", "K34": "#009E73", "Sa1": "#56B4E9", "Ma3": "#E69F00", "Npw": "#D55E00"}
MK = {"Jaru": "o", "K34": "s", "Sa1": "^", "Ma3": "D", "Npw": "v"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "savefig.dpi": 600})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600)
    plt.close(fig)


def load(name):
    d = F.load_site(name)
    return d.assign(WS=d.WS.fillna(1.0)) if name == "Npw" else d


# ---------------------------------------------------------------- tables
def tables():
    weak = {"Jaru": "sindy_confirm_weak", "K34": "sindy_k34_weak", "Sa1": "sindy_Sa1_weak", "Ma3": "sindy_Ma3_weak"}
    rows = []
    for s in SITES:
        if s == "Npw":
            d = pd.read_csv(R / "sindy_Npw_weak.csv")
            for _, r in d.iterrows():
                rows.append({"site": s, "season": r.season, "n_test": r.n_test, "r2_relax": r.r2_relax,
                             "dR2": r.dR2, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                             "per_year": " / ".join(f"{r[c]:+.3f}" for c in d.columns if c.startswith("dR2_2"))})
            continue
        d = pd.read_csv(R / f"{weak[s]}.csv")
        for season in ["all", "wet", "dry"]:
            rr = d[(d.season == season) & (d.model == "relax+harm")].iloc[0]
            r = d[(d.season == season) & (d.model == "sat+harm")].iloc[0]
            rows.append({"site": s, "season": season, "n_test": r.n_test, "r2_relax": rr.r2_test, "dR2": r.dR2,
                         "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                         "per_year": " / ".join(f"{r[c]:+.3f}" for c in d.columns if c.startswith("dR2_2"))})
    t2 = pd.DataFrame(rows)
    sims = {"Jaru": ("sindy_confirm_sim_summary", "dRMSE"), "K34": ("sindy_k34_sim_summary", "dRMSE"),
            "Sa1": ("sindy_wind_Sa1_sim_noWS", "dRMSE_vs_relax+harm"), "Ma3": ("sindy_Ma3_sim_summary", "dRMSE"),
            "Npw": ("sindy_Npw_sim", "dRMSE_vs_relax+harm")}
    s3 = []
    for s, (f, c) in sims.items():
        d = pd.read_csv(R / f"{f}.csv")
        ref = d[d.model == "relax+harm"].iloc[0]; r = d[d.model == "sat+harm"].iloc[0]
        s3.append({"site": s, "n_days": int(r.n_days), "rmse_relax": ref.mean_rmse, "rmse_sat": r.mean_rmse,
                   "dRMSE": r[c], "ci_lo": r.ci_lo, "ci_hi": r.ci_hi})
    t3 = pd.DataFrame(s3)
    bp = pd.read_csv(R / "b_profile.csv")
    t2.to_csv(FIG / "table2_weak_gain.csv", index=False)
    t3.to_csv(FIG / "table3_simulation.csv", index=False)
    bp.to_csv(FIG / "table4_b_profile.csv", index=False)
    return t2, t3, bp


# ---------------------------------------------------------------- figure 2: shape of the response
def fig2():
    fig, axs = plt.subplots(1, 5, figsize=(180 / 25.4, 50 / 25.4), layout="constrained", sharey=True)
    out = []
    for ax, s in zip(axs, SITES):
        cfg = F.SITES[s]
        d = load(s)
        rows = W.weak_rows(d, F.M)
        ry = np.array([r[0] for r in rows])
        xi, b = F.fit(d, rows, np.isin(ry, cfg["train"]), "sat+harm")
        # strong-form tendency (SG window 7, order 2) on gap-free runs of T within each year
        d["dTdt"] = pd.concat([_sg_derivative(g["T"], 0.5, 7, 2) for _, g in d.groupby("year")])
        te = d[d.year.isin(cfg["test"]) & (d.SW > 5) & d.dTdt.notna()].copy()
        X = F.cols(te, "sat+harm", b)
        nonsw = X @ xi - X[:, 2] * xi[2]
        te["pres"] = te.dTdt - nonsw  # partial residual: tendency minus the fitted non-radiative terms
        te["bin"] = (te.SW // 50) * 50 + 25
        daily = te.groupby(["bin", "day"]).pres.mean().reset_index()  # day = sampling unit
        g = daily.groupby("bin").pres.agg(["mean", "std", "count"])
        g = g[g["count"] >= 20]
        se = g["std"] / np.sqrt(g["count"])
        ax.errorbar(g.index, g["mean"], yerr=1.96 * se, fmt=MK[s], ms=2.5, color=COL[s], lw=0.8, capsize=0,
                    label="binned partial residual")
        sw = np.linspace(0, g.index.max() + 25, 200)
        ax.plot(sw, xi[2] * (1 - np.exp(-b * sw)), color="k", lw=1.2, label=r"$A(1-e^{-b\,SW})$")
        wls = np.polyfit(g.index, g["mean"], 1, w=np.sqrt(g["count"]))
        ax.plot(sw, np.polyval(wls, sw), color="0.4", ls="--", lw=0.9, label="weighted linear fit to bins")
        ax.set_title(LABEL[s])
        ax.text(0.97, 0.04, f"b = {b:g} m$^2$ W$^{{-1}}$\nA = {xi[2]:.2f} K h$^{{-1}}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6)
        ax.set_xlabel(r"SW$_{in}$ (W m$^{-2}$)")
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        out.append(g.assign(site=s, se=se).reset_index())
    axs[0].set_ylabel("partial residual of dT/dt (K h$^{-1}$)")
    axs[0].legend(loc="upper left", frameon=False, fontsize=6)
    save(fig, "fig1_response_shape")
    pd.concat(out).to_csv(FIG / "fig1_data.csv", index=False)


# ---------------------------------------------------------------- figure 3: gain in the tendency fit
def fig3(t2):
    ps = pd.read_csv(R / "pysr_sites_pysr.csv")
    fig, axs = plt.subplots(1, 2, figsize=(180 / 25.4, 62 / 25.4), layout="constrained",
                            gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    off = {"all": -0.22, "wet": 0.0, "dry": 0.22}
    fill = {"all": "full", "wet": "none", "dry": "bottom"}
    for i, s in enumerate(SITES):
        for season in ["all", "wet", "dry"]:
            r = t2[(t2.site == s) & (t2.season == season)].iloc[0]
            ax.errorbar(i + off[season], r.dR2, yerr=[[r.dR2 - r.ci_lo], [r.ci_hi - r.dR2]], fmt=MK[s],
                        color=COL[s], fillstyle=fill[season], ms=4, lw=0.9, capsize=1.5)
    for season, f in fill.items():
        ax.plot([], [], "o", color="0.3", fillstyle=f, label=season)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(range(5), [LABEL[s].replace(" (", "\n(") for s in SITES])
    ax.set_ylabel(r"$\Delta R^2_{weak}$, sat+harm $-$ relax+harm")
    ax.set_title("(a) weak-form SINDy, independent test years (95% daily-block CI)")
    ax.legend(title="season", frameon=False, ncol=3, loc="upper left")
    ax = axs[1]
    for i, s in enumerate(SITES):
        sat = ps[(ps.site == s) & ps.model.str.startswith("sat")].iloc[0]
        ax.errorbar(i - 0.2, sat.dR2, yerr=[[sat.dR2 - sat.ci_lo], [sat.ci_hi - sat.dR2]], fmt=MK[s], color=COL[s],
                    ms=4, lw=0.9, capsize=1.5)
        p = ps[(ps.site == s) & (ps.model == "relax+harm + PySR")]
        for k, (_, r) in enumerate(p.iterrows()):
            ax.errorbar(i + 0.08 + 0.1 * k, r.dR2, yerr=[[r.dR2 - r.ci_lo], [r.ci_hi - r.dR2]], fmt=MK[s],
                        color=COL[s], fillstyle="none" if r.saturating_SW else "full", mfc="w" if r.saturating_SW else "k",
                        ms=3, lw=0.6, capsize=1)
    ax.plot([], [], "o", color="0.3", label="sat+harm (2 parameters)")
    ax.plot([], [], "o", color="0.3", mfc="w", ms=3, label="PySR residual, seeds 0-2 (concave)")
    ax.plot([], [], "o", color="k", ms=3, label="PySR, not concave")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(range(5), [s for s in SITES])
    ax.set_ylabel(r"$\Delta R^2$ (strong form, SG derivative)")
    ax.set_title("(b) free-form PySR search on the residual")
    ax.set_ylim(-0.04, 0.07)
    ax.legend(frameon=False, loc="upper left", fontsize=6, ncol=1)
    save(fig, "fig2_tendency_gain")


# ---------------------------------------------------------------- figure 4: 24-h free runs
def fig4(t3):
    fig, ax = plt.subplots(figsize=(89 / 25.4, 62 / 25.4), layout="constrained")
    for i, r in t3.reset_index(drop=True).iterrows():
        s = r.site
        ax.errorbar(i, r.dRMSE, yerr=[[r.dRMSE - r.ci_lo], [r.ci_hi - r.dRMSE]], fmt=MK[s], color=COL[s], ms=4,
                    lw=0.9, capsize=2, fillstyle="full" if FOREST[s] else "none")
        ax.text(i, 0.235, f"n = {r.n_days} d", ha="center", va="top", fontsize=6)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(range(5), [LABEL[s].replace(" (", "\n(") for s in t3.site], fontsize=6)
    ax.set_ylim(-0.25, 0.25); ax.set_xlim(-0.6, 4.6)
    ax.set_ylabel(r"$\Delta$RMSE of 24-h free run (K)" + "\nsat+harm $-$ relax+harm")
    ax.set_title("filled: closed forest; open: mosaic or savanna\n(95% CI over days)", fontsize=7)
    save(fig, "fig3_free_runs")


# ---------------------------------------------------------------- figure 5: identifiability of b
def fig5(bp):
    cur = pd.read_csv(R / "b_profile_curves.csv")
    fig, axs = plt.subplots(1, 2, figsize=(180 / 25.4, 60 / 25.4), layout="constrained")
    ax = axs[0]
    for s in SITES:
        for part, ls in (("train", "-"), ("test", ":")):
            c = cur[(cur.site == s) & (cur.part == part)]
            ax.plot(c.b, c.sse_rel, ls=ls, color=COL[s], lw=1, label=LABEL[s] if part == "train" else None)
    ax.set_xscale("log"); ax.set_ylim(0.995, 1.06)
    ax.set_xlabel(r"b (m$^2$ W$^{-1}$, log$_{10}$ axis)"); ax.set_ylabel("weak-form SSE / minimum")
    ax.set_title("(a) profile of b (solid: training years; dotted: test years)")
    ax.legend(frameon=False, fontsize=6)
    ax = axs[1]
    for i, s in enumerate(SITES):
        for part, dx, fs in (("train", -0.12, "full"), ("test", 0.12, "none")):
            r = bp[(bp.site == s) & (bp.part == part)].iloc[0]
            ax.errorbar(i + dx, r.b_hat * 1e3, yerr=[[(r.b_hat - r.b_lo) * 1e3], [(r.b_hi - r.b_hat) * 1e3]],
                        fmt=MK[s], color=COL[s], fillstyle=fs, ms=4, lw=0.9, capsize=1.5)
    ax.plot([], [], "o", color="0.3", label="training years"); ax.plot([], [], "o", color="0.3", fillstyle="none", label="test years")
    ax.set_xticks(range(5), SITES)
    ax.set_ylabel(r"$\hat b$ ($10^{-3}$ m$^2$ W$^{-1}$)")
    sec = ax.secondary_yaxis("right", functions=(lambda x: np.log(2) / np.maximum(x, 1e-6) * 1e3,
                                                  lambda y: np.log(2) / np.maximum(y, 1e-6) * 1e3))
    sec.set_ylabel(r"SW$_{1/2}$ = ln2 / b (W m$^{-2}$)")
    ax.set_title("(b) fitted b, 95% daily-block bootstrap")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig4_b_profile")


# ---------------------------------------------------------------- figure 6: mechanism tests
def fig6(raw):
    blh = pd.read_csv(R / "sindy_blh_mechanism.csv")
    hf = pd.read_csv(R / "sindy_h_forcing.csv")
    n = 3 if raw else 2
    fig, axs = plt.subplots(1, n, figsize=(180 / 25.4, 60 / 25.4), layout="constrained")
    k = 0
    if raw:
        ax = axs[0]; k = 1
        out = []
        for s in SITES:
            d = HN.with_H(s, F.load_site(s), raw)
            m = d[d.hour.between(10, 14) & d.SW.notna() & d.H.notna()]
            m = m.assign(bin=(m.SW // 50) * 50 + 25)
            g = m.groupby("bin").H.agg(["mean", "count"]); g = g[g["count"] >= 30]
            ax.plot(g.index, g["mean"], marker=MK[s], ms=2.5, color=COL[s], lw=0.9, label=LABEL[s])
            out.append(g.assign(site=s).reset_index())
        pd.concat(out).to_csv(FIG / "fig5a_data.csv", index=False)
        ax.set_xlabel(r"SW$_{in}$ (W m$^{-2}$)"); ax.set_ylabel(r"measured H, 10-14 h (W m$^{-2}$)")
        ax.set_title("(a) midday H against SW (bin means)")
        ax.legend(frameon=False, fontsize=6)
    ax = axs[k]
    for i, s in enumerate(SITES):
        r = hf[hf.site == s].iloc[0]
        if s == "Ma3":
            ax.annotate("collapse\n(H < 0 in exp form)", (i, 0.0), ha="center", va="bottom", fontsize=6)
            continue
        ax.errorbar(i - 0.12, r.dR2_H, yerr=[[r.dR2_H - r.ci_lo], [r.ci_hi - r.dR2_H]], fmt=MK[s], color=COL[s], ms=4,
                    capsize=1.5, lw=0.9)
        ax.plot(i + 0.12, r.dR2_SW_same_rows, MK[s], color=COL[s], fillstyle="none", ms=4)
    ax.plot([], [], "o", color="0.3", label="forcing H (95% CI)"); ax.plot([], [], "o", color="0.3", fillstyle="none", label="forcing SW, same rows")
    ax.axhline(0, color="0.5", lw=0.6); ax.set_xticks(range(5), SITES)
    ax.set_ylabel(r"$\Delta R^2_{weak}$, saturating $-$ linear"); ax.set_title("(b) saturation against measured H")
    ax.set_ylim(-0.004, 0.04)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax = axs[k + 1]
    for i, s in enumerate(SITES):
        r = blh[blh.site == s].iloc[0]
        ax.errorbar(i - 0.15, r.P1_dR2_BL_vs_relax, yerr=[[r.P1_dR2_BL_vs_relax - r.P1_ci_lo], [r.P1_ci_hi - r.P1_dR2_BL_vs_relax]],
                    fmt="x", color=COL[s], ms=4, capsize=1.5, lw=0.9)
        ax.plot(i, r.gain_sat_over_relax, MK[s], color=COL[s], ms=4)
        ax.plot(i + 0.15, r.gain_sat_over_BL, MK[s], color=COL[s], fillstyle="none", ms=4)
    ax.plot([], [], "x", color="0.3", label="SW/h (ERA5) $-$ relax")
    ax.plot([], [], "o", color="0.3", label="saturation over relax")
    ax.plot([], [], "o", color="0.3", fillstyle="none", label="saturation over SW/h")
    ax.axhline(0, color="0.5", lw=0.6); ax.set_xticks(range(5), SITES)
    ax.set_ylabel(r"$\Delta R^2_{weak}$"); ax.set_title("(c) ERA5 boundary-layer test")
    ax.set_ylim(-0.09, 0.19)
    ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=1)
    save(fig, "fig5_mechanism_tests")


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else None
    t2, t3, bp = tables()
    print(t2.round(4).to_string(index=False)); print(t3.round(4).to_string(index=False))
    fig2(); fig3(t2); fig4(t3); fig5(bp); fig6(raw)
    print("figures written to", FIG)
