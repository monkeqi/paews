"""
PAEWS Phase 2 - Step 1: SST Baseline Builder
=============================================
Downloads 22 years (2003-2025) of daily SST data from NOAA OISST,
one year at a time, and saves each year as a separate NetCDF file.

WHY WE NEED THIS:
To detect anomalies, we need to know what "normal" looks like.
If February SST off Peru is usually 22C, and today it's 25C,
that's abnormal. But we can only say that if we have 20+ years
of February data to compute the average and standard deviation.

This is called a "climatological baseline" -- the historical
reference that all future measurements are compared against.

WHY YEAR BY YEAR:
The full 22-year daily dataset is too large to download in one
request -- the ERDDAP server would time out. So we break it into
annual chunks. Each year of daily SST for our Peru box is about
300-400 KB (small because OISST is 0.25 degree resolution).

IMPORTANT: This script will take 10-20 minutes to run because
it makes 23 separate HTTP requests with pauses between them
to be polite to the NOAA server. Do not interrupt it.

After this finishes, you will have 23 files:
  data/baseline/sst_2003.nc
  data/baseline/sst_2004.nc
  ...
  data/baseline/sst_2025.nc

Next step: compute_climatology.py will read all these files
and compute monthly means and standard deviations.

Usage:
  & C:/Users/josep/miniconda3/Scripts/conda.exe run -n geosentinel python c:/Users/josep/Documents/paews/scripts/baseline_builder.py
"""

import sys
print("PAEWS Baseline Builder starting...", flush=True)

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
DATA_BASELINE = BASE_DIR / "data" / "baseline"

# Same Peru bounding box as our data pipeline
LAT_MIN, LAT_MAX = -16, -4
LON_MIN, LON_MAX = -82, -70

# OISST dataset details (same server and dataset as Phase 1)
SST_SERVER = "https://coastwatch.pfeg.noaa.gov"
SST_DATASET = "ncdcOisst21Agg_LonPM180"
SST_VARIABLE = "sst"

# Baseline period: 2003 to 2025
# We start at 2003 to match MODIS Aqua chlorophyll availability.
# This gives us 23 years of data -- more than enough for robust
# climatological statistics. The WMO recommends 30 years for
# climate normals, but for ocean monitoring 20+ is standard
# practice (Hobday et al. 2016 used ~30 years where available).
BASELINE_START_YEAR = 2003
BASELINE_END_YEAR = 2025

# Pause between downloads (seconds) -- be polite to the server.
# ERDDAP servers are free public resources. Hammering them with
# rapid requests can get your IP temporarily blocked. A 3-second
# pause between annual downloads is courteous and safe.
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
    Download one full year of daily SST for the Peru box.

    Each file will contain ~365 daily grids (366 for leap years),
    each grid being 49 lat x 49 lon pixels at 0.25 degree resolution.

    The data is stored as a 3D array: (time, latitude, longitude)
    with an additional altitude dimension that we collapse.
    """
    start_date = f"{year}-01-01"
    stop_date = f"{year}-12-31"

    # If this is the most recent year, don't request beyond available data
    if server_max_date:
        requested_stop = datetime(year, 12, 31)
        if requested_stop > server_max_date:
            stop_date = server_max_date.strftime("%Y-%m-%d")
            print(f"  Adjusted stop date to {stop_date} (server limit)", flush=True)

    outfile = DATA_BASELINE / f"sst_{year}.nc"

    # Skip if already downloaded (allows resuming interrupted runs)
    if outfile.exists():
        # Verify the file is valid by trying to open it
        try:
            ds = xr.open_dataset(outfile)
            n_days = ds.dims.get("time", ds.sizes.get("time", 0))
            ds.close()
            print(f"  {year}: Already exists ({n_days} days) - skipping", flush=True)
            return True
        except Exception:
            print(f"  {year}: Existing file is corrupted - re-downloading", flush=True)

    # Build ERDDAP URL
    # OISST uses T12:00:00Z timestamps (noon UTC)
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
        # Verify the downloaded file
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
    print("PAEWS Baseline Builder - SST Historical Data", flush=True)
    print(f"Period: {BASELINE_START_YEAR} to {BASELINE_END_YEAR}", flush=True)
    print(f"Region: {LAT_MIN}N to {LAT_MAX}N, {LON_MIN}E to {LON_MAX}E", flush=True)
    print(f"Dataset: {SST_DATASET}", flush=True)
    print("=" * 55, flush=True)

    # Create output directory
    DATA_BASELINE.mkdir(parents=True, exist_ok=True)

    # Check server availability
    t_min, t_max = get_sst_time_range()

    # Track results
    years = list(range(BASELINE_START_YEAR, BASELINE_END_YEAR + 1))
    total = len(years)
    success = 0
    failed = []

    print(f"\nDownloading {total} years of SST data...", flush=True)
    print(f"(This will take approximately {total * 30 // 60} - {total * 45 // 60} minutes)\n", flush=True)

    for i, year in enumerate(years):
        print(f"[{i+1}/{total}] Year {year}...", flush=True)
        if download_sst_year(year, t_max):
            success += 1
        else:
            failed.append(year)

        # Pause between downloads (skip after last one)
        if i < total - 1:
            time.sleep(PAUSE_BETWEEN_DOWNLOADS)

    # Summary
    print("\n" + "=" * 55, flush=True)
    print("BASELINE DOWNLOAD SUMMARY", flush=True)
    print(f"  Successful: {success}/{total} years", flush=True)
    if failed:
        print(f"  Failed: {failed}", flush=True)
        print(f"  Re-run this script to retry failed years", flush=True)
    else:
        print("  All years downloaded successfully!", flush=True)

    # List what we have
    print(f"\nFiles in {DATA_BASELINE}:", flush=True)
    for f in sorted(DATA_BASELINE.glob("sst_*.nc")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.0f} KB", flush=True)

    print("=" * 55, flush=True)
    if success == total:
        print("Next step: Run compute_climatology.py", flush=True)
    else:
        print("Fix failed downloads before proceeding.", flush=True)
