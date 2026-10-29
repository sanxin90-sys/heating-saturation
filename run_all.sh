#!/usr/bin/env bash
# Reproduces every number, table and figure cited in the manuscript (see README.md).
#
# Usage:
#   bash run_all.sh RAW [XLSX]
#     RAW  = folder with ma2/, Sa1/, Ma3/, Npw/ holding the extracted AmeriFlux FLUXNET-1F
#            FLUXMET_HH csv files (needed for measured H; see README, "Data").
#     XLSX = optional: the Rebio Jaru site spreadsheet (2006-2008). Only the discovery-phase
#            runs EXP-007/008 (manuscript Section 4.1, first paragraph) need it; it is not public.
#
# PySR (Section 4.3) is not run here: it needs Julia and ran on a separate computer.
# Its output, src/sr_benchmark/results/pysr_sites_pysr.csv, is included; rerun with
#   python src/pysr_sites.py --flux-dir RAW
set -euo pipefail
RAW=${1:?give the folder with the extracted FLUXMET_HH files}
XLSX=${2:-}
cd "$(dirname "$0")"
R=src/sr_benchmark/results
mkdir -p $R

if [ -n "$XLSX" ]; then
  python src/sindy_weak_loyo.py "$XLSX"                    > $R/sindy_weak_loyo.log    # EXP-007, Jaru 2006-08 (discovery)
  python src/sindy_exp008.py "$XLSX"                       > $R/sindy_exp008.log       # EXP-008 A-C (discovery)
  python -c "import sys; sys.path.insert(0, 'src'); import sindy_exp008 as E; E.wind_check(sys.argv[1])" "$XLSX" \
                                                          >> $R/sindy_exp008.log       # EXP-008 D (exploratory wind term)
fi
python src/sindy_confirm_newyears.py                       > $R/sindy_confirm.log      # EXP-009, Jaru confirmation years
python src/sindy_site_k34.py                               > $R/sindy_k34.log          # EXP-010, K34
python src/sindy_site_generic.py --site Sa1 --csv data/fluxnet_BR-Sa1/BR-Sa1_HH_TA_SW_WS.csv \
  --train 2002 2003 2004 --test 2009 2010 2011 --wet 12 1 2 3 4 5 --dry 8 9 10 11 --no-wind-filter > $R/sindy_Sa1.log  # EXP-013
python src/sindy_site_generic.py --site Ma3 --csv data/fluxnet_BR-Ma3/BR-Ma3_HH_TA_SW_WS.csv \
  --train 2009 2010 2011 --test 2013 2018 --wet 12 1 2 3 4 5 --dry 7 8 9 10 > $R/sindy_Ma3.log                     # EXP-013
python src/sindy_wind_forms.py "$RAW"                      > $R/sindy_wind_forms.log   # EXP-014 (wind forms, Santarem 188 d, H/LE shape)
python src/sindy_h_and_npw.py "$RAW"                       > $R/sindy_h_and_npw.log    # EXP-015 (H forcing), EXP-016 (Pantanal)
python src/sindy_blh_mechanism.py                          > $R/sindy_blh_mechanism.log # EXP-017 (ERA5 boundary-layer test)
python src/b_profile.py                                    > $R/b_profile.log          # Table 4, Figure 4 (descriptive)
python src/h_shape.py "$RAW"                               > $R/h_shape.log            # Section 4.7, quadratic H check (descriptive)
python src/make_figures.py "$RAW"                                                      # Figures 1-5, Tables 2-4
