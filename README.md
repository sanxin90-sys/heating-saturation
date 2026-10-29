# Code and data for: "A saturating response of above-canopy air warming to incoming solar radiation at five tropical South American flux towers"

This archive reproduces every number, table and figure of the manuscript from AmeriFlux FLUXNET-1F
data (v1.3_r1) and ERA5 boundary-layer height. Status of the archive: prepared 2026-10-29 for review;
a persistent DOI (e.g. Zenodo) has not been minted yet.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# extract the AmeriFlux zips (see "Data") into RAW/ma2, RAW/Sa1, RAW/Ma3, RAW/Npw, then:
bash run_all.sh /path/to/RAW                 # all confirmatory and descriptive results
bash run_all.sh /path/to/RAW "/path/to/Dados Rebio Jaru 2006 a 2008.xlsx"   # + discovery runs EXP-007/008
```

Outputs go to `src/sr_benchmark/results/` (logs and CSVs) and `figures/` (PDF and 600-dpi PNG figures,
CSV tables and the data behind Figures 1 and 5a). Every run uses fixed random seeds; the reruns made on
2026-10-29 reproduced the original logs exactly (apart from timing lines).

Tested with Python 3.13.17, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, pysindy 2.1.0,
sympy 1.14.0, openpyxl 3.1.5, matplotlib 3.11.2, pillow 12.3.0, on Linux.

## What reproduces what

| Manuscript item | Script | Experiment id | Main outputs |
|---|---|---|---|
| Sec. 4.1, discovery at Jaru 2006-2008 (needs the site spreadsheet) | `sindy_weak_loyo.py`, `sindy_exp008.py` (+ `wind_check`) | EXP-007, EXP-008 | `sindy_weak_loyo.log`, `sindy_exp008.log`, `sindy_exp008_*.csv` |
| Sec. 4.1, confirmation at Jaru (2005, 2009, 2010; rules R1-R4) | `sindy_confirm_newyears.py` | EXP-009 | `sindy_confirm.log`, `sindy_confirm_*.csv` |
| Table 2, Fig. 2a, K34; Sec. 4.6 (M1 at K34, transfer) | `sindy_site_k34.py` | EXP-010 | `sindy_k34.log`, `sindy_k34_*.csv` |
| Table 2, Fig. 2a, Santarem and Colosso | `sindy_site_generic.py` | EXP-013 | `sindy_Sa1*`, `sindy_Ma3*` |
| Sec. 4.6 wind forms; Table 3 Santarem (188 d); Sec. 4.7 H/LE saturating fits | `sindy_wind_forms.py` | EXP-014 | `sindy_wind_forms.log`, `sindy_wind_*.csv`, `sindy_b_fluxes.csv` |
| Sec. 4.7 H as forcing (Fig. 5b); Table 2/3 Pantanal | `sindy_h_and_npw.py` | EXP-015, EXP-016 | `sindy_h_and_npw.log`, `sindy_h_forcing.csv`, `sindy_Npw_*.csv` |
| Sec. 4.7 ERA5 boundary-layer test (Fig. 5c) | `sindy_blh_mechanism.py` (data from `era5_blh_fetch.py`) | EXP-017 | `sindy_blh_mechanism.{log,csv}` |
| Table 4, Fig. 4 (profile of b; descriptive, added while writing) | `b_profile.py` | - | `b_profile.csv`, `b_profile_curves.csv` |
| Sec. 4.7 quadratic H check (Fig. 5a; descriptive, added while writing) | `h_shape.py` | - | `h_shape.csv` |
| Sec. 4.3, Fig. 2b (PySR) | `pysr_sites.py` (needs PySR + Julia) | pysr.md 4b | `pysr_sites_baselines.*`, `pysr_sites_pysr.csv` (included) |
| Figures 1-5, Tables 2-4 | `make_figures.py` | - | `figures/` |

Shared code: `sindy_weak_loyo.py` (weak form, test functions, STLSQ, bootstrap), `sindy_wind_forms.py`
(site configuration, model libraries, weak system), `sindy_confirm_newyears.py` (24-h RK4 free runs),
`real_discovery.py` (Savitzky-Golay derivative, spreadsheet loader), `dtdt_robustness.py` (list of suspect
months, DQ-001), `sr_benchmark/` (helpers imported by `real_discovery.py`).

## Data

* `data/fluxnet_BR-Ji3/`: the AmeriFlux FLUXNET-1F BR-Ji3 BIF files as distributed, and the FLUXMET_HH file reduced to the 13
  columns the scripts read (TIMESTAMP_END; TA, SW_IN, H, LE with their _F_MDS values and QC flags; WS, WS_F_QC, P, RH), with
  values copied unchanged. A full rerun with this reduced file reproduced all 49 result logs and tables exactly.
* `data/fluxnet_BR-{Ma2,Sa1,Ma3,Npw}/`: subsets (TIMESTAMP, TA_F_MDS, SW_IN_F_MDS, WS and their QC flags)
  of the FLUXMET_HH files of those sites.
* `RAW/` (not included, ~1 GB): the full FLUXMET_HH files of BR-Ma2, BR-Sa1, BR-Ma3 and BR-Npw, needed only for
  measured sensible heat flux (H_F_MDS). Download from AmeriFlux (DOIs below), unzip, and place each site's
  `AMF_<site>_FLUXNET_FLUXMET_HH_*.csv` in `RAW/ma2`, `RAW/Sa1`, `RAW/Ma3`, `RAW/Npw`.
* `data/era5_blh/blh_sites.csv`: ERA5 hourly boundary-layer height at the 0.25-degree cell nearest each tower,
  15, 18 and 21 UTC, all days of the years used, read from the public ARCO-ERA5 copy on Google Cloud by
  `src/era5_blh_fetch.py` (Copernicus licence; cite Hersbach et al., 2020).
* The Rebio Jaru site spreadsheet (2006-2008; Rn, 22-m temperature, wind) is **not included**: it was supplied by
  the tower team and is used only for the discovery-phase runs EXP-007/008. All confirmatory results use FLUXNET
  data only.

AmeriFlux data are licensed CC-BY-4.0. Cite each site dataset:

* BR-Ji3: Araujo, A. (2026). AmeriFlux FLUXNET-1F BR-Ji3 Ji-Paraná/RO - Reserva Biológica do Jaru (RBJ), Ver. v1.3_r1 (2004-2010) [Dataset]. AmeriFlux AMP. https://doi.org/10.17190/AMF/3027314
* BR-Ma2: Araujo, A., von Randow, C., Manzi, A., Nobre, A., Pastorello, G., de França e Silva, J. A., Leal, L. do S. M., Sá, M., Teixeira, P., & von Randow, R. (2026). AmeriFlux FLUXNET-1F BR-Ma2 Manaus - ZF2 K34, Ver. v1.3_r1 (1999-2016) [Dataset]. AmeriFlux AMP. https://doi.org/10.17190/AMF/3027671
* BR-Sa1: Restrepo-Coupe, N., & Saleska, S. (2026). AmeriFlux FLUXNET-1F BR-Sa1 Santarem-Km67-Primary Forest, Ver. v1.3_r1 (2002-2011) [Dataset]. AmeriFlux AMP. https://doi.org/10.17190/AMF/3027310
* BR-Ma3: Araujo, A. (2026). AmeriFlux FLUXNET-1F BR-Ma3 ZF3, Colosso farm, Ver. v1.3_r1 (2008-2018) [Dataset]. AmeriFlux AMP. https://doi.org/10.17190/AMF/3027359
* BR-Npw: Vourlitis, G., Dalmagro, H., Nogueira, J. de S., Johnson, M., & Arruda, P. (2026). AmeriFlux FLUXNET-1F BR-Npw Northern Pantanal Wetland, Ver. v1.3_r1 (2013-2017) [Dataset]. AmeriFlux AMP. https://doi.org/10.17190/AMF/1881563

(Author names, titles, versions and DOIs as given in the AmeriFlux download citation file `citations_for_site_data_FLUXNET_20261026.csv`; given names abbreviated.)

## Notes and known quirks

* Santarem (BR-Sa1) is run with `--no-wind-filter` (registered deviation: the wind quality filter inherited from
  Jaru would remove 49 of 72 months). The log still prints the list of months the filter *would* drop; they are not
  dropped.
* At Pantanal (BR-Npw) missing wind speed is filled with 1.0 only so that it does not remove rows; the models
  compared (relax+harm, sat+harm) do not use wind.
* `b_profile.py` and `h_shape.py` were added while writing the manuscript and are descriptive, not pre-registered.
* The decision rules of each experiment are written in the docstring of its script, in the words used before
  the run.

## Licence

Code: MIT (see LICENSE). Data: CC-BY-4.0 (AmeriFlux) and the
Copernicus licence (ERA5), as above.
