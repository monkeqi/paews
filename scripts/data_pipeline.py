import sys
print("PAEWS starting...", flush=True)

import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime, timedelta
import requests
import json

print("Imports done", flush=True)

# ============================================================
# CONFIGURATION
# ============================================================
# All paths and parameters in one place so they are easy to find
# and change later when we move to Phase 2.

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_CURRENT = BASE_DIR / "data" / "current"
OUTPUTS = BASE_DIR / "outputs"

# Peru coastal study area bounding box
# 4S to 16S latitude, 82W to 70W longitude
# This covers the main Peruvian upwelling zone where anchovies live
LAT_MIN, LAT_MAX = -16, -4
LON_MIN, LON_MAX = -82, -70


# ============================================================
# DATASET NOTES (read this - it matters for understanding)
# ============================================================
#
# CHLOROPHYLL - We use NOAA VIIRS (S-NPP satellite), NOT MODIS Aqua.
#
# Why? MODIS Aqua launched in 2002 and its global 8-day product on
# CoastWatch ERDDAP (erdMH1chla8day) stopped updating in June 2022.
# The regional West Coast version (erdMWchla8day) is still updated
# but only covers 22N-51N -- useless for Peru at 4S-16S.
#
# VIIRS is the successor sensor to MODIS for ocean color. It flies
# on the S-NPP satellite (launched 2011) and NOAA-20 (launched 2017).
# The dataset "noaacwNPPVIIRSchlaDaily" is global, 4km, near-real-time,
# and runs from 2017 to present. It uses the OC3 chlorophyll algorithm,
# which is similar to what MODIS uses.
#
# For Phase 2 backtesting (going back to 2003), we will use the
# historical MODIS dataset (erdMH1chla8day) for the long record.
# For current monitoring, we use VIIRS. This is called "sensor
# continuity" -- transitioning between satellite missions while
# maintaining consistent measurements. Understanding this is
# important for space industry work.
#
# IMPORTANT: The VIIRS dataset is on coastwatch.noaa.gov (NOAA
# CoastWatch main server), NOT coastwatch.pfeg.noaa.gov (the
# Pacific Fisheries server where SST lives). Different servers,
# different datasets. This is normal -- ERDDAP is a protocol,
# and many organizations run their own ERDDAP servers.
#
# SST - We still use NOAA OISST on coastwatch.pfeg.noaa.gov.
# This is Level 4 data (gap-filled using microwave + buoys + ships).
# It sees through clouds, unlike optical chlorophyll sensors.
# The dataset has a ~2 week processing lag, so we cannot request
# data right up to today.
#
# KEY DIFFERENCE between the two variables:
#   Chlorophyll variable name: "chlor_a" (VIIRS naming convention)
#   SST variable name: "sst" (OISST naming convention)
#   Both have an altitude dimension that we set to [0:1:0]


# ============================================================
# HELPER: CHECK DATASET INFO
# ============================================================
# Before we try to download data, we ask the ERDDAP server
# "what time range and coordinate ranges do you actually have?"
# This prevents 404 errors when our request falls outside the
# dataset's actual coverage.
#
# ERDDAP has a built-in info page for every dataset in JSON format.
# We parse that to find the actual min/max for each axis.

def get_dataset_info(server_base, dataset_id):
    """Ask ERDDAP what time and coordinate ranges a dataset has."""
    url = f"{server_base}/erddap/info/{dataset_id}/index.json"
    print(f"  Checking dataset info for {dataset_id}...", flush=True)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  WARNING: Could not get info (HTTP {r.status_code})", flush=True)
            return None
        data = r.json()
        # ERDDAP info JSON has a "table" with "columnNames" and "rows"
        # Each row is like: ["attribute", "time", "actual_range", "float64", "1.0E9, 1.7E9"]
        # We want rows where column 2 (attribute name) is "actual_range"
        info = {}
        rows = data["table"]["rows"]
        for row in rows:
            if len(row) >= 5 and row[2] == "actual_range":
                var_name = row[1]
                value = row[4]
                info[var_name] = value
        return info
    except Exception as e:
        print(f"  WARNING: Info check failed: {e}", flush=True)
        return None


def parse_time_range(info):
    """Convert epoch timestamps from info into readable dates."""
    if not info or "time" not in info:
        return None, None
    try:
        parts = info["time"].split(",")
        t_min = datetime.utcfromtimestamp(float(parts[0].strip()))
        t_max = datetime.utcfromtimestamp(float(parts[1].strip()))
        return t_min, t_max
    except Exception:
        return None, None


def print_dataset_ranges(info, dataset_name):
    """Print what ranges a dataset has so we can see what is available."""
    if not info:
        print(f"  Could not retrieve ranges for {dataset_name}", flush=True)
        return
    print(f"  {dataset_name} available ranges:", flush=True)
    for key in ["time", "latitude", "longitude", "altitude"]:
        if key in info:
            print(f"    {key}: {info[key]}", flush=True)
    # Also print human-readable time range
    t_min, t_max = parse_time_range(info)
    if t_min and t_max:
        print(f"    time (readable): {t_min.strftime('%Y-%m-%d')} to {t_max.strftime('%Y-%m-%d')}", flush=True)


# ============================================================
# CHLOROPHYLL DATA PULL
# ============================================================
# Dataset: noaacwNPPVIIRSchlaDaily (VIIRS S-NPP, daily, global 4km)
# Server: coastwatch.noaa.gov
# Variable: chlor_a
# Has altitude dimension: yes [0:1:0]
#
# This is Level 3 mapped data -- the satellite takes raw radiance
# measurements (Level 0), processes them to calibrated data (Level 1),
# derives geophysical variables like chlorophyll per pixel (Level 2),
# and then maps all orbits from one day onto a regular grid (Level 3).
#
# "Daily" means one file per day, but cloud-covered pixels will be
# NaN (no data). This is the La Garua problem -- Peru's coastal
# fog can block optical sensors for days. In Phase 2 we will handle
# this with our confidence tier system.

CHL_SERVER = "https://coastwatch.noaa.gov"
CHL_DATASET = "noaacwNPPVIIRSchlaDaily"
CHL_VARIABLE = "chlor_a"

def fetch_viirs_chlorophyll(start_date, stop_date):
    print(f"Fetching VIIRS chlorophyll: {start_date} to {stop_date}...", flush=True)

    # Step 1: Check what is actually available
    info = get_dataset_info(CHL_SERVER, CHL_DATASET)
    print_dataset_ranges(info, "VIIRS Chlorophyll")

    # Adjust stop date if needed
    t_min, t_max = parse_time_range(info)
    if t_max:
        requested_stop = datetime.strptime(stop_date, "%Y-%m-%d")
        if requested_stop > t_max:
            adjusted = t_max.strftime("%Y-%m-%d")
            print(f"  Adjusting stop date from {stop_date} to {adjusted} (server limit)", flush=True)
            stop_date = adjusted

    # Step 2: Build the ERDDAP griddap URL
    url = (
        f"{CHL_SERVER}/erddap/griddap/{CHL_DATASET}.nc"
        f"?{CHL_VARIABLE}[({start_date}T00:00:00Z):1:({stop_date}T00:00:00Z)]"
        f"[0:1:0]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )
    print(f"  URL: {url[:120]}...", flush=True)

    # Step 3: Download
    DATA_CURRENT.mkdir(parents=True, exist_ok=True)
    outfile = DATA_CURRENT / "chlorophyll_current.nc"

    try:
        r = requests.get(url, timeout=180)
    except requests.exceptions.Timeout:
        print("  ERROR: Request timed out after 180 seconds", flush=True)
        return None

    if r.status_code == 200:
        outfile.write_bytes(r.content)
        print(f"  Downloaded {len(r.content)/1024:.0f} KB", flush=True)
    else:
        error_text = r.text[:400]
        print(f"  ERROR {r.status_code}: {error_text}", flush=True)

        # If latitude order is wrong, try flipping
        if "latitude" in error_text.lower() and ("less than" in error_text.lower() or "greater than" in error_text.lower()):
            print("  Retrying with flipped latitude order...", flush=True)
            url = (
                f"{CHL_SERVER}/erddap/griddap/{CHL_DATASET}.nc"
                f"?{CHL_VARIABLE}[({start_date}T00:00:00Z):1:({stop_date}T00:00:00Z)]"
                f"[0:1:0]"
                f"[({LAT_MAX}):1:({LAT_MIN})]"
                f"[({LON_MIN}):1:({LON_MAX})]"
            )
            try:
                r = requests.get(url, timeout=180)
            except requests.exceptions.Timeout:
                print("  ERROR: Retry also timed out", flush=True)
                return None
            if r.status_code == 200:
                outfile.write_bytes(r.content)
                print(f"  Downloaded {len(r.content)/1024:.0f} KB (flipped lat worked)", flush=True)
            else:
                print(f"  RETRY ALSO FAILED {r.status_code}: {r.text[:300]}", flush=True)
                return None
        else:
            return None

    # Step 4: Open with xarray and report what we got
    ds = xr.open_dataset(outfile)
    print(f"  Got chlorophyll data: {dict(ds.dims)}", flush=True)
    return ds


# ============================================================
# SST DATA PULL
# ============================================================
# Dataset: ncdcOisst21Agg_LonPM180 (NOAA OISST v2.1, daily)
# Server: coastwatch.pfeg.noaa.gov
# Variable: sst
# Has altitude dimension: yes [0:1:0]
#
# This is Level 4 data -- it goes one step beyond Level 3 by
# INTERPOLATING to fill all gaps. It fuses satellite microwave
# measurements, ship observations, buoy data, and uses statistical
# methods to produce a complete grid with no missing values.
#
# This is why SST always works even when chlorophyll has cloud gaps.
# Microwave radiation (used by AMSR-E and similar sensors) passes
# through clouds. Visible light (used by MODIS/VIIRS for chlorophyll)
# does not.
#
# The "Agg" means aggregated -- NOAA combines daily files into one
# continuous time series on ERDDAP. The "LonPM180" means longitude
# runs from -180 to +180 (not 0 to 360), which is what we want
# since Peru is at negative longitude.
#
# OISST timestamps are at T12:00:00Z (noon UTC), not midnight.
# This is because the daily average is centered on noon.

SST_SERVER = "https://coastwatch.pfeg.noaa.gov"
SST_DATASET = "ncdcOisst21Agg_LonPM180"
SST_VARIABLE = "sst"

def fetch_noaa_sst(start_date, stop_date):
    print(f"Fetching NOAA OISST: {start_date} to {stop_date}...", flush=True)

    # Step 1: Check what is actually available
    info = get_dataset_info(SST_SERVER, SST_DATASET)
    print_dataset_ranges(info, "NOAA OISST")

    # Adjust stop date if it is beyond available data
    t_min, t_max = parse_time_range(info)
    if t_max:
        requested_stop = datetime.strptime(stop_date, "%Y-%m-%d")
        if requested_stop > t_max:
            adjusted = t_max.strftime("%Y-%m-%d")
            print(f"  Adjusting stop date from {stop_date} to {adjusted} (server limit)", flush=True)
            stop_date = adjusted

    # Step 2: Build URL
    url = (
        f"{SST_SERVER}/erddap/griddap/{SST_DATASET}.nc"
        f"?{SST_VARIABLE}[({start_date}T12:00:00Z):1:({stop_date}T12:00:00Z)]"
        f"[0:1:0]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )
    print(f"  URL: {url[:120]}...", flush=True)

    # Step 3: Download
    DATA_CURRENT.mkdir(parents=True, exist_ok=True)
    outfile = DATA_CURRENT / "sst_current.nc"

    try:
        r = requests.get(url, timeout=180)
    except requests.exceptions.Timeout:
        print("  ERROR: Request timed out after 180 seconds", flush=True)
        return None

    if r.status_code == 200:
        outfile.write_bytes(r.content)
        print(f"  Downloaded {len(r.content)/1024:.0f} KB", flush=True)
    else:
        error_text = r.text[:400]
        print(f"  ERROR {r.status_code}: {error_text}", flush=True)

        # If latitude order is wrong, try flipping
        if "latitude" in error_text.lower() and ("less than" in error_text.lower() or "greater than" in error_text.lower()):
            print("  Retrying with flipped latitude order...", flush=True)
            url = (
                f"{SST_SERVER}/erddap/griddap/{SST_DATASET}.nc"
                f"?{SST_VARIABLE}[({start_date}T12:00:00Z):1:({stop_date}T12:00:00Z)]"
                f"[0:1:0]"
                f"[({LAT_MAX}):1:({LAT_MIN})]"
                f"[({LON_MIN}):1:({LON_MAX})]"
            )
            try:
                r = requests.get(url, timeout=180)
            except requests.exceptions.Timeout:
                print("  ERROR: Retry also timed out", flush=True)
                return None
            if r.status_code == 200:
                outfile.write_bytes(r.content)
                print(f"  Downloaded {len(r.content)/1024:.0f} KB (flipped lat worked)", flush=True)
            else:
                print(f"  RETRY ALSO FAILED {r.status_code}: {r.text[:300]}", flush=True)
                return None
        else:
            return None

    # Step 4: Open with xarray and report what we got
    ds = xr.open_dataset(outfile)
    print(f"  Got SST data: {dict(ds.dims)}", flush=True)
    return ds


# ============================================================
# VISUALIZATION
# ============================================================
# Chlorophyll is plotted on a LOG scale because values span orders
# of magnitude -- open ocean might be 0.01 mg/m3 while productive
# upwelling water near shore can be 20+ mg/m3. A linear scale
# would make the offshore detail invisible.
#
# SST is plotted on a LINEAR scale with the "RdYlBu_r" colormap
# (red = warm, blue = cold). Off Peru, SST typically ranges from
# ~14C (cold upwelled water near shore) to ~28C (warm tropical
# water offshore/north). The cold upwelling water is where the
# nutrients are -- that is what feeds the phytoplankton that feeds
# the anchovies.
#
# We use .isel(time=-1) to get the LAST (most recent) time step.

def plot_chlorophyll(ds):
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # The variable name depends on the dataset
    # VIIRS uses "chlor_a", MODIS uses "chlorophyll"
    if "chlor_a" in ds:
        chl = ds["chlor_a"].isel(time=-1)
    elif "chlorophyll" in ds:
        chl = ds["chlorophyll"].isel(time=-1)
    else:
        print(f"  ERROR: Cannot find chlorophyll variable. Available: {list(ds.data_vars)}", flush=True)
        return

    # Get the date for the title
    time_val = ds["time"].values[-1]
    date_str = str(time_val)[:10]

    # Count valid (non-NaN) pixels for data coverage reporting
    total_pixels = chl.size
    valid_pixels = int(chl.count().values)
    coverage_pct = (valid_pixels / total_pixels) * 100
    print(f"  Data coverage: {valid_pixels}/{total_pixels} pixels ({coverage_pct:.1f}%)", flush=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    chl.plot(
        ax=ax,
        norm=mcolors.LogNorm(vmin=0.01, vmax=20),
        cmap="viridis",
        cbar_kwargs={"label": "Chlorophyll-a (mg/m3)"}
    )
    ax.set_title(
        f"VIIRS S-NPP Chlorophyll-a - Peru Coast\n"
        f"{date_str} | Coverage: {coverage_pct:.0f}%"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    outpath = OUTPUTS / "chlorophyll_map.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chlorophyll map saved: {outpath}", flush=True)


def plot_sst(ds):
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    sst = ds["sst"].isel(time=-1)

    # Get the date for the title
    time_val = ds["time"].values[-1]
    date_str = str(time_val)[:10]

    # SST from OISST should have near 100% coverage (it is interpolated)
    total_pixels = sst.size
    valid_pixels = int(sst.count().values)
    coverage_pct = (valid_pixels / total_pixels) * 100
    print(f"  Data coverage: {valid_pixels}/{total_pixels} pixels ({coverage_pct:.1f}%)", flush=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    sst.plot(
        ax=ax,
        vmin=14,
        vmax=28,
        cmap="RdYlBu_r",
        cbar_kwargs={"label": "SST (C)"}
    )
    ax.set_title(
        f"NOAA OISST Sea Surface Temperature - Peru Coast\n"
        f"{date_str} | Coverage: {coverage_pct:.0f}%"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    outpath = OUTPUTS / "sst_map.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SST map saved: {outpath}", flush=True)


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # We request a short recent window.
    # Both datasets have processing lag so we stay a few weeks back.
    # The functions will auto-adjust if stop_date is too recent.
    start = "2026-01-01"
    stop = datetime.now().strftime("%Y-%m-%d")

    print("=" * 50, flush=True)
    print("PAEWS Data Pipeline v0.2", flush=True)
    print(f"Period: {start} to {stop}", flush=True)
    print(f"Region: {LAT_MIN}N to {LAT_MAX}N, {LON_MIN}E to {LON_MAX}E", flush=True)
    print("=" * 50, flush=True)

    # Pull chlorophyll (VIIRS - may have cloud gaps)
    chl_ds = fetch_viirs_chlorophyll(start, stop)
    if chl_ds:
        plot_chlorophyll(chl_ds)
    else:
        print("  Chlorophyll download failed - skipping map", flush=True)

    print("-" * 50, flush=True)

    # Pull SST (OISST - gap-filled, always complete)
    sst_ds = fetch_noaa_sst(start, stop)
    if sst_ds:
        plot_sst(sst_ds)
    else:
        print("  SST download failed - skipping map", flush=True)

    print("=" * 50, flush=True)
    if chl_ds and sst_ds:
        print("SUCCESS: Both datasets downloaded. Check outputs/ folder.", flush=True)
    elif chl_ds or sst_ds:
        print("PARTIAL: One dataset downloaded. Check errors above.", flush=True)
    else:
        print("FAILED: No data downloaded. Check errors above.", flush=True)
