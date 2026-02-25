"""
PAEWS Chlorophyll Baseline Builder
===================================
Downloads MODIS Aqua 8-day chlorophyll-a composites from ERDDAP
for the 2003-2022 baseline period.

Dataset: erdMWchla8day (MODIS Aqua, 8-day composite, ~4km resolution)
Box: 0S-16S, 85W-70W (extended to capture coastal El Nino zone)
Baseline: 2003-2022 (clean — excludes 2023+ El Nino)

WHY 8-DAY COMPOSITES?
    MODIS Aqua passes over Peru once per day, but clouds block the view
    most days — especially during La Garua season (Jun-Sep). An 8-day
    composite merges ~8 overpasses into one image, filling in cloud gaps.
    This is standard Level 3 processing. You'll still see missing pixels
    (NaN) where ALL 8 days were cloudy — that's real, not a bug.

WHY 4KM RESOLUTION?
    Chlorophyll varies sharply near the coast. The Espinoza-Morriberon (2025)
    finding — chlorophyll INCREASES within 25km of shore during El Nino while
    DECREASING offshore — requires enough spatial resolution to see that
    gradient. At 4km, 25km = ~6 pixels. Enough to detect the pattern.
    SST at 0.25° (25km) would completely miss it.

EXPECTED FILE SIZES:
    Each year: ~15-30 MB (varies with cloud cover — more NaN = smaller file)
    Total baseline: ~400-600 MB
    Download time: ~30-60 minutes depending on ERDDAP server load

Usage:
    python chl_baseline_builder.py
"""

import sys
print("PAEWS Chlorophyll Baseline Builder starting...", flush=True)

import requests
from pathlib import Path
import time

print("Imports done", flush=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_CHL_BASELINE = BASE_DIR / "data" / "baseline_v2_chl"

# Bounding box for chlorophyll
# erdMH1chla8day is global (-90 to 90), so we can use the same box as SST
# Latitude axis runs south-to-north in this dataset
LAT_MIN, LAT_MAX = -16, 0
LON_MIN, LON_MAX = -85, -70

# Clean baseline period (matches SST)
BASELINE_START_YEAR = 2003
BASELINE_END_YEAR = 2022

# ERDDAP dataset
# erdMH1chla8day = MODIS Aqua, Global, 4km, Science Quality, 8-day composite
# This is a Level 3 SMI (Standard Mapped Image) product
# Coverage: global (-90 to 90 lat, -180 to 180 lon)
# Time: 2003-01-05 to 2022-06-14
# NOTE: erdMWchla8day was WRONG — that's a US West Coast regional product!
# Units: mg/m³ (milligrams of chlorophyll-a per cubic meter of seawater)
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
DATASET_ID = "erdMH1chla8day"

# Download settings
TIMEOUT = 600       # 10 minutes per request (chlorophyll files are larger than SST)
RETRY_ATTEMPTS = 3
RETRY_DELAY = 30    # seconds between retries
PAUSE_BETWEEN = 5   # seconds between successful downloads


def download_year(year):
    """
    Download one year of MODIS chlorophyll data from ERDDAP.
    
    The URL structure for ERDDAP griddap:
        dataset.nc?variable[(time_start):stride:(time_stop)]
                           [(lat_start):stride:(lat_stop)]
                           [(lon_start):stride:(lon_stop)]
    
    Notes:
        - Dataset erdMH1chla8day is global, lat runs south-to-north
        - No altitude dimension in this dataset
        - Time uses ISO 8601 format
        - Coverage ends 2022-06-14
    """
    DATA_CHL_BASELINE.mkdir(parents=True, exist_ok=True)
    outfile = DATA_CHL_BASELINE / f"chl_{year}.nc"
    
    # Skip if already downloaded
    if outfile.exists():
        size_mb = outfile.stat().st_size / (1024 * 1024)
        print(f"  [{year}] Already exists ({size_mb:.1f} MB), skipping", flush=True)
        return True
    
    start_date = f"{year}-01-01"
    # erdMH1chla8day coverage ends 2022-06-14 — cap the last year
    if year >= 2022:
        end_date = "2022-06-14"
    else:
        end_date = f"{year}-12-31"
    
    # Build ERDDAP URL
    # erdMH1chla8day is global, latitude runs south-to-north (-90 to 90)
    # So LAT_MIN (-16) comes first, LAT_MAX (0) comes second
    # NO altitude dimension in this dataset (unlike erdMWchla8day)
    url = (
        f"{ERDDAP_BASE}/{DATASET_ID}.nc"
        f"?chlorophyll[({start_date}T00:00:00Z):1:({end_date}T00:00:00Z)]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"               # lat: south to north
        f"[({LON_MIN}):1:({LON_MAX})]"                # lon: west to east
    )
    
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            print(f"  [{year}] Downloading (attempt {attempt}/{RETRY_ATTEMPTS})...", flush=True)
            r = requests.get(url, timeout=TIMEOUT)
            
            if r.status_code == 200:
                outfile.write_bytes(r.content)
                size_mb = len(r.content) / (1024 * 1024)
                print(f"  [{year}] Success: {size_mb:.1f} MB", flush=True)
                return True
            else:
                # Extract error message
                error_msg = r.text[:300] if r.text else "No error message"
                print(f"  [{year}] HTTP {r.status_code}: {error_msg}", flush=True)
                
                if r.status_code == 404:
                    # Data doesn't exist for this year/range — skip, don't retry
                    print(f"  [{year}] Data not available, skipping", flush=True)
                    return False
                    
        except requests.exceptions.Timeout:
            print(f"  [{year}] Timeout after {TIMEOUT}s", flush=True)
        except requests.exceptions.ConnectionError as e:
            print(f"  [{year}] Connection error: {e}", flush=True)
        except Exception as e:
            print(f"  [{year}] Unexpected error: {e}", flush=True)
        
        if attempt < RETRY_ATTEMPTS:
            print(f"  [{year}] Retrying in {RETRY_DELAY}s...", flush=True)
            time.sleep(RETRY_DELAY)
    
    print(f"  [{year}] FAILED after {RETRY_ATTEMPTS} attempts", flush=True)
    return False


if __name__ == "__main__":
    years = list(range(BASELINE_START_YEAR, BASELINE_END_YEAR + 1))
    total = len(years)
    
    print("=" * 60, flush=True)
    print("PAEWS Chlorophyll Baseline Builder", flush=True)
    print(f"Dataset: {DATASET_ID} (MODIS Aqua 8-day composite, ~4km)", flush=True)
    print(f"Period: {BASELINE_START_YEAR}-{BASELINE_END_YEAR} ({total} years)", flush=True)
    print(f"Box: {LAT_MIN}S to {LAT_MAX}S (equator), {LON_MIN}W to {LON_MAX}W", flush=True)
    print(f"Output: {DATA_CHL_BASELINE}", flush=True)
    print("=" * 60, flush=True)
    
    success = 0
    failed = 0
    skipped = 0
    
    for i, year in enumerate(years, 1):
        print(f"\n[{i}/{total}] Year {year}:", flush=True)
        
        outfile = DATA_CHL_BASELINE / f"chl_{year}.nc"
        if outfile.exists():
            skipped += 1
            size_mb = outfile.stat().st_size / (1024 * 1024)
            print(f"  Already exists ({size_mb:.1f} MB), skipping", flush=True)
            continue
        
        result = download_year(year)
        if result:
            success += 1
        else:
            failed += 1
        
        # Pause between downloads to be nice to the server
        if i < total:
            time.sleep(PAUSE_BETWEEN)
    
    print("\n" + "=" * 60, flush=True)
    print("CHLOROPHYLL BASELINE DOWNLOAD COMPLETE", flush=True)
    print(f"  Success: {success}", flush=True)
    print(f"  Skipped (already had): {skipped}", flush=True)
    print(f"  Failed: {failed}", flush=True)
    print(f"  Files in: {DATA_CHL_BASELINE}", flush=True)
    print("=" * 60, flush=True)
    
    if failed > 0:
        print("\nTo retry failed years, just run this script again.", flush=True)
        print("It will skip files already downloaded.", flush=True)
