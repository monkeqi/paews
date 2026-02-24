"""
PAEWS Phase 2 - Step 1: SST Baseline Builder (v2)
===================================================
Downloads 20 years (2003-2022) of daily SST data from NOAA OISST,
one year at a time, and saves each year as a separate NetCDF file.

VERSION 2 CHANGES:
- Extended bounding box from 5S-15S to 0S-16S (captures coastal
  El Nino hotspot at 0-5S that v1 missed entirely)
- Extended longitude from 82W-70W to 85W-70W (captures more
  offshore equatorial water for better gradient computation)
- Downloads to data/baseline_v2/ to preserve v1 data
- Clean baseline: 2003-2022 (excludes 2023+ El Nino)

BOUNDING BOX RATIONALE:
Literature review confirmed that coastal El Ninos (like 2017)
concentrate extreme warming (+5C anomalies) in the 0-5S zone.
Our v1 box at 5S-15S completely missed this. The Nino 1+2 region
(0-10S, 90W-80W) is the standard monitoring box for coastal Peru
events. Our new box (0-16S, 85W-70W) overlaps with Nino 1+2
while also covering the main anchovy fishing grounds (4S-14S).

IMPORTANT: This script will take 15-20 minutes to run because
it makes 20 HTTP requests with pauses between them. Each year's
file is larger than v1 (~500-600 KB) due to the bigger box.

Usage:
  & C:/Users/josep/miniconda3/Scripts/conda.exe run -n geosentinel python c:/Users/josep/Documents/paews/scripts/baseline_builder.py
"""

import sys
print("PAEWS Baseline Builder v2 starting...", flush=True)

import xarray as xr
import requests
from pathlib import Path
from datetime import datetime
import time

print("Imports done", flush=True)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_BASELINE = BASE_DIR / "data" / "baseline_v2"

# EXTENDED Peru coastal study area bounding box (v2)
# 0S to 16S latitude, 85W to 70W longitude
# v1 was 4S to 16S, 82W to 70W -- missed coastal El Nino zone
LAT_MIN, LAT_MAX = -16, 0
LON_MIN, LON_MAX = -85, -70

# OISST dataset details
SST_SERVER = "https://coastwatch.pfeg.noaa.gov"
SST_DATASET = "ncdcOisst21Agg_LonPM180"
SST_VARIABLE = "sst"

# Clean baseline: 2003-2022 (excludes 2023+ El Nino)
BASELINE_START_YEAR = 2003
BASELINE_END_YEAR = 2022

# Pause between downloads (seconds)
PAUSE_BETWEEN_DOWNLOADS = 3


# ============================================================
# HELPER: CHECK DATASET TIME RANGE
# ============================================================
def get_sst_time_range():
    """Check what time range the SST dataset actually has."""
    url = f"{SST_SERVER}/erddap/info/{SST_DATASET}/index.json"
    print("Checking SST dataset availability...", flush=True)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  WARNING: Could not check (HTTP {r.status_code})", flush=True)
            return None, None
        data = r.json()
        for row in data["table"]["rows"]:
            if len(row) >= 5 and row[1] == "time" and row[2] == "actual_range":
                parts = row[4].split(",")
                t_min = datetime.utcfromtimestamp(float(parts[0].strip()))
                t_max = datetime.utcfromtimestamp(float(parts[1].strip()))
                print(f"  SST available: {t_min.strftime('%Y-%m-%d')} to {t_max.strftime('%Y-%m-%d')}", flush=True)
                return t_min, t_max
    except Exception as e:
        print(f"  WARNING: Check failed: {e}", flush=True)
    return None, None


# ============================================================
# DOWNLOAD ONE YEAR OF SST
# ============================================================
def download_sst_year(year, server_max_date=None):
    """
    Download one full year of daily SST for the extended Peru box.

    v2 box is larger than v1:
      v1: 49 lat x 49 lon = 2,401 pixels per day
      v2: 65 lat x 61 lon = 3,965 pixels per day
    Files will be ~50-60% larger than v1.
    """
    start_date = f"{year}-01-01"
    stop_date = f"{year}-12-31"

    if server_max_date:
        requested_stop = datetime(year, 12, 31)
        if requested_stop > server_max_date:
            stop_date = server_max_date.strftime("%Y-%m-%d")
            print(f"  Adjusted stop date to {stop_date} (server limit)", flush=True)

    outfile = DATA_BASELINE / f"sst_{year}.nc"

    # Skip if already downloaded
    if outfile.exists():
        try:
            ds = xr.open_dataset(outfile)
            n_days = ds.dims.get("time", ds.sizes.get("time", 0))
            ds.close()
            print(f"  {year}: Already exists ({n_days} days) - skipping", flush=True)
            return True
        except Exception:
            print(f"  {year}: Existing file is corrupted - re-downloading", flush=True)

    # Build ERDDAP URL
    url = (
        f"{SST_SERVER}/erddap/griddap/{SST_DATASET}.nc"
        f"?{SST_VARIABLE}[({start_date}T12:00:00Z):1:({stop_date}T12:00:00Z)]"
        f"[0:1:0]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )

    try:
        r = requests.get(url, timeout=300)
    except requests.exceptions.Timeout:
        print(f"  {year}: TIMEOUT after 300 seconds", flush=True)
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"  {year}: CONNECTION ERROR: {e}", flush=True)
        return False

    if r.status_code == 200:
        outfile.write_bytes(r.content)
        size_kb = len(r.content) / 1024
        try:
            ds = xr.open_dataset(outfile)
            n_days = ds.dims.get("time", ds.sizes.get("time", 0))
            ds.close()
            print(f"  {year}: Downloaded {size_kb:.0f} KB ({n_days} days)", flush=True)
            return True
        except Exception as e:
            print(f"  {year}: Downloaded but file is invalid: {e}", flush=True)
            return False
    else:
        print(f"  {year}: ERROR {r.status_code}: {r.text[:200]}", flush=True)
        return False


# ============================================================
# MAIN: DOWNLOAD ALL YEARS
# ============================================================
if __name__ == "__main__":
    print("=" * 55, flush=True)
    print("PAEWS Baseline Builder v2 - Extended Box", flush=True)
    print(f"Period: {BASELINE_START_YEAR} to {BASELINE_END_YEAR}", flush=True)
    print(f"Region: {LAT_MIN}N to {LAT_MAX}N, {LON_MIN}E to {LON_MAX}E", flush=True)
    print(f"Output: {DATA_BASELINE}", flush=True)
    print(f"Dataset: {SST_DATASET}", flush=True)
    print("=" * 55, flush=True)

    DATA_BASELINE.mkdir(parents=True, exist_ok=True)
    t_min, t_max = get_sst_time_range()

    years = list(range(BASELINE_START_YEAR, BASELINE_END_YEAR + 1))
    total = len(years)
    success = 0
    failed = []

    print(f"\nDownloading {total} years of SST data...", flush=True)
    print(f"(This will take approximately {total * 30 // 60} - {total * 60 // 60} minutes)\n", flush=True)

    for i, year in enumerate(years):
        print(f"[{i+1}/{total}] Year {year}...", flush=True)
        if download_sst_year(year, t_max):
            success += 1
        else:
            failed.append(year)

        if i < total - 1:
            time.sleep(PAUSE_BETWEEN_DOWNLOADS)

    print("\n" + "=" * 55, flush=True)
    print("BASELINE DOWNLOAD SUMMARY", flush=True)
    print(f"  Successful: {success}/{total} years", flush=True)
    if failed:
        print(f"  Failed: {failed}", flush=True)
        print(f"  Re-run this script to retry failed years", flush=True)
    else:
        print("  All years downloaded successfully!", flush=True)

    print(f"\nFiles in {DATA_BASELINE}:", flush=True)
    for f in sorted(DATA_BASELINE.glob("sst_*.nc")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.0f} KB", flush=True)

    print("=" * 55, flush=True)
    if success == total:
        print("Next step: Run compute_climatology.py", flush=True)
    else:
        print("Fix failed downloads before proceeding.", flush=True)
