"""
era5_blh_fetch.py — ERA5 boundary-layer height (blh, m) at the five towers, from the
public ARCO-ERA5 Zarr store on Google Cloud (gcp-public-data-arco-era5,
full_37-1h-0p25deg-chunk-1.zarr-v3; ERA5 licence: Copernicus, attribution required).
Nearest 0.25-degree grid cell. Hours 15, 18, 21 UTC (11, 14, 17 local, UTC-4) of
every day of the years used by EXP-009..016. Output: data/era5_blh/blh_sites.csv
(date, hour_utc, site, blh).
"""
import sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, numcodecs, pandas as pd

B = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
SITES = {"Jaru": (-10.078, -61.933), "K34": (-2.609, -60.209), "Sa1": (-2.857, -54.959),
         "Ma3": (-2.391, -59.884), "Npw": (-16.498, -56.412)}
YEARS = sorted(set(range(2002, 2012)) | {2013, 2014, 2015, 2016, 2017, 2018})
bl = numcodecs.Blosc()
lat = 90 - 0.25 * np.arange(721); lon = 0.25 * np.arange(1440)
IDX = {k: (int(np.argmin(abs(lat - la))), int(np.argmin(abs(lon - lo % 360)))) for k, (la, lo) in SITES.items()}


def get(ts):
    t = int((ts - pd.Timestamp("1900-01-01")).total_seconds() // 3600)
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(B + f"boundary_layer_height/{t}.0.0", timeout=90).read()
            a = np.frombuffer(bl.decode(raw), dtype="<f4").reshape(721, 1440)
            return [(ts.date(), ts.hour, k, float(a[i, j])) for k, (i, j) in IDX.items()]
        except Exception:
            pass
    return [(ts.date(), ts.hour, k, np.nan) for k in IDX]


if __name__ == "__main__":
    out = Path(sys.argv[1])
    times = [pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=d, hours=h)
             for y in YEARS for d in range(366 if y % 4 == 0 else 365) for h in (15, 18, 21)]
    with ThreadPoolExecutor(32) as ex:
        rows = [r for chunk in ex.map(get, times) for r in chunk]
    pd.DataFrame(rows, columns=["date", "hour_utc", "site", "blh"]).to_csv(out, index=False)
    print("done", len(rows), "grid cells:", IDX)
