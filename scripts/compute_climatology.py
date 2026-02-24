"""
PAEWS Phase 2 - Step 2: Compute SST Climatology (v2)
=====================================================
Reads 20 years (2003-2022) of baseline SST data from the EXTENDED
bounding box (0S-16S, 85W-70W) and computes monthly climatological
statistics: mean and standard deviation for each pixel for each
calendar month (January through December).

VERSION 2 CHANGES:
- Reads from data/baseline_v2/ (extended box 0S-16S, 85W-70W)
- Saves to data/processed/sst_climatology_v2.nc
- Updated gradient zones for the wider box
- Clean baseline: 2003-2022 (excludes 2023+ El Nino)

OUTPUT:
  data/processed/sst_climatology_v2.nc   - climatology for extended box
  outputs/sst_climatology_preview.png    - visual check
  outputs/sst_gradient_seasonal.png      - gradient analysis

Usage:
  & C:/Users/josep/miniconda3/Scripts/conda.exe run -n geosentinel python c:/Users/josep/Documents/paews/scripts/compute_climatology.py
"""

import sys
print("PAEWS Climatology Calculator v2 starting...", flush=True)

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

print("Imports done", flush=True)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_BASELINE = BASE_DIR / "data" / "baseline_v2"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"

# Clean baseline: 2003-2022
BASELINE_START_YEAR = 2003
BASELINE_END_YEAR = 2022

# Climatology output filename (v2 to distinguish from v1)
CLIM_FILENAME = "sst_climatology_v2.nc"


# ============================================================
# STEP 1: LOAD ALL BASELINE FILES INTO ONE DATASET
# ============================================================
def load_all_baseline_data():
    """Load all yearly SST files and combine into one big dataset."""
    print("\nStep 1: Loading baseline files...", flush=True)

    datasets = []
    total_days = 0

    for year in range(BASELINE_START_YEAR, BASELINE_END_YEAR + 1):
        filepath = DATA_BASELINE / f"sst_{year}.nc"
        if not filepath.exists():
            print(f"  WARNING: {filepath.name} not found - skipping", flush=True)
            continue

        ds = xr.open_dataset(filepath)

        if "zlev" in ds.dims:
            ds = ds.squeeze("zlev", drop=True)
        if "altitude" in ds.dims:
            ds = ds.squeeze("altitude", drop=True)

        n_days = ds.sizes["time"]
        total_days += n_days
        datasets.append(ds)
        print(f"  Loaded {filepath.name}: {n_days} days", flush=True)

    if not datasets:
        print("ERROR: No baseline files found!", flush=True)
        print(f"  Expected files in: {DATA_BASELINE}", flush=True)
        print(f"  Run baseline_builder.py first.", flush=True)
        return None

    print(f"\n  Combining {len(datasets)} files ({total_days} total days)...", flush=True)
    combined = xr.concat(datasets, dim="time")
    combined = combined.sortby("time")

    print(f"  Combined dataset shape: {dict(combined.sizes)}", flush=True)
    print(f"  Time range: {str(combined.time.values[0])[:10]} to {str(combined.time.values[-1])[:10]}", flush=True)
    print(f"  Lat range: {float(combined.latitude.min()):.2f} to {float(combined.latitude.max()):.2f}", flush=True)
    print(f"  Lon range: {float(combined.longitude.min()):.2f} to {float(combined.longitude.max()):.2f}", flush=True)

    return combined


# ============================================================
# STEP 2: COMPUTE MONTHLY CLIMATOLOGY
# ============================================================
def compute_monthly_climatology(combined_ds):
    """Compute mean and std of SST for each calendar month."""
    print("\nStep 2: Computing monthly climatology...", flush=True)

    sst = combined_ds["sst"]

    print("  Computing monthly means...", flush=True)
    monthly_mean = sst.groupby("time.month").mean(dim="time")

    print("  Computing monthly standard deviations (ddof=1)...", flush=True)
    monthly_std = sst.groupby("time.month").std(dim="time", ddof=1)

    print("  Counting valid observations per month...", flush=True)
    monthly_count = sst.groupby("time.month").count(dim="time")

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    print("\n  Monthly climatology summary (spatial averages):", flush=True)
    print(f"  {'Month':<6} {'Mean SST':>10} {'Std Dev':>10} {'Avg Days':>10}", flush=True)
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}", flush=True)

    for m in range(1, 13):
        mean_val = float(monthly_mean.sel(month=m).mean(skipna=True).values)
        std_val = float(monthly_std.sel(month=m).mean(skipna=True).values)
        count_val = float(monthly_count.sel(month=m).mean(skipna=True).values)
        print(f"  {month_names[m-1]:<6} {mean_val:>10.2f}C {std_val:>10.2f}C {count_val:>10.0f}", flush=True)

    return monthly_mean, monthly_std, monthly_count


# ============================================================
# STEP 3: SAVE CLIMATOLOGY TO NETCDF
# ============================================================
def save_climatology(monthly_mean, monthly_std, monthly_count):
    """Save climatology as a NetCDF file."""
    print("\nStep 3: Saving climatology...", flush=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    clim_ds = xr.Dataset(
        {
            "sst_mean": monthly_mean,
            "sst_std": monthly_std,
            "sst_count": monthly_count,
        },
        attrs={
            "title": "PAEWS SST Monthly Climatology v2 (Extended Box)",
            "description": "Monthly mean and standard deviation of SST from NOAA OISST",
            "baseline_period": f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}",
            "baseline_note": "Clean baseline excluding 2023+ El Nino event years",
            "bounding_box": "0S-16S, 85W-70W (extended from v1: 4S-16S, 82W-70W)",
            "source": "ncdcOisst21Agg_LonPM180 via ERDDAP",
            "created_by": "PAEWS compute_climatology.py v2",
        }
    )

    outpath = DATA_PROCESSED / CLIM_FILENAME
    clim_ds.to_netcdf(outpath)
    size_kb = outpath.stat().st_size / 1024
    print(f"  Saved: {outpath} ({size_kb:.0f} KB)", flush=True)

    return clim_ds


# ============================================================
# STEP 4: VISUALIZE THE CLIMATOLOGY
# ============================================================
def plot_climatology(clim_ds):
    """Create a 12-panel plot showing monthly mean SST."""
    print("\nStep 4: Creating climatology preview...", flush=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(
        f"PAEWS SST Monthly Climatology v2 ({BASELINE_START_YEAR}-{BASELINE_END_YEAR})\n"
        f"NOAA OISST - Extended Box (0S-16S, 85W-70W)",
        fontsize=16, fontweight="bold"
    )

    for m in range(1, 13):
        row = (m - 1) // 4
        col = (m - 1) % 4
        ax = axes[row, col]

        sst_month = clim_ds["sst_mean"].sel(month=m)

        im = sst_month.plot(
            ax=ax,
            vmin=14,
            vmax=28,
            cmap="RdYlBu_r",
            add_colorbar=False
        )
        ax.set_title(month_names[m - 1], fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.2)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Mean SST (C)")

    outpath = OUTPUTS / "sst_climatology_preview.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Preview saved: {outpath}", flush=True)


# ============================================================
# STEP 5: COMPUTE AND PLOT THE COASTAL-OFFSHORE SST GRADIENT
# ============================================================
# Updated for the wider box. Nearshore and offshore zones are
# the same longitude ranges as v1, but now we have more latitude
# coverage to see how the gradient varies from equator to 16S.

def compute_gradient_climatology(clim_ds):
    """Compute the coastal-offshore SST gradient for each month."""
    print("\nStep 5: Computing coastal-offshore SST gradient...", flush=True)

    sst_mean = clim_ds["sst_mean"]

    # Same gradient zones as v1 for consistency
    nearshore = sst_mean.sel(longitude=slice(-78, -70)).mean(dim=["latitude", "longitude"])
    offshore = sst_mean.sel(longitude=slice(-85, -80)).mean(dim=["latitude", "longitude"])

    gradient = offshore - nearshore

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    print(f"\n  Monthly SST Gradient (offshore - nearshore):", flush=True)
    print(f"  {'Month':<6} {'Nearshore':>10} {'Offshore':>10} {'Gradient':>10}", flush=True)
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}", flush=True)

    for m in range(1, 13):
        ns_val = float(nearshore.sel(month=m).values)
        os_val = float(offshore.sel(month=m).values)
        gr_val = float(gradient.sel(month=m).values)
        print(f"  {month_names[m-1]:<6} {ns_val:>10.2f}C {os_val:>10.2f}C {gr_val:>+10.2f}C", flush=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    months = list(range(1, 13))

    ax1.plot(months, [float(nearshore.sel(month=m).values) for m in months],
             'b-o', linewidth=2, label='Nearshore (< 50km from coast)')
    ax1.plot(months, [float(offshore.sel(month=m).values) for m in months],
             'r-o', linewidth=2, label='Offshore (100-200km)')
    ax1.set_ylabel("SST (C)")
    ax1.set_title("Seasonal SST Cycle: Nearshore vs Offshore")
    ax1.legend()
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names)
    ax1.grid(True, alpha=0.3)

    gradient_vals = [float(gradient.sel(month=m).values) for m in months]
    colors = ['green' if g > 0 else 'red' for g in gradient_vals]
    ax2.bar(months, gradient_vals, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel("SST Gradient (C)")
    ax2.set_title("Coastal-Offshore SST Gradient (positive = active upwelling)")
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_names)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"PAEWS Upwelling Gradient Analysis v2 ({BASELINE_START_YEAR}-{BASELINE_END_YEAR})",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    outpath = OUTPUTS / "sst_gradient_seasonal.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gradient plot saved: {outpath}", flush=True)

    return gradient


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("=" * 55, flush=True)
    print("PAEWS Climatology Calculator v2 (Extended Box)", flush=True)
    print(f"Baseline: {BASELINE_START_YEAR} to {BASELINE_END_YEAR} (clean)", flush=True)
    print(f"Source: {DATA_BASELINE}", flush=True)
    print("=" * 55, flush=True)

    combined = load_all_baseline_data()
    if combined is None:
        sys.exit(1)

    monthly_mean, monthly_std, monthly_count = compute_monthly_climatology(combined)
    clim_ds = save_climatology(monthly_mean, monthly_std, monthly_count)
    plot_climatology(clim_ds)
    gradient = compute_gradient_climatology(clim_ds)

    print("\n" + "=" * 55, flush=True)
    print("CLIMATOLOGY v2 COMPLETE", flush=True)
    print(f"  Climatology file: data/processed/{CLIM_FILENAME}", flush=True)
    print(f"  Preview map: outputs/sst_climatology_preview.png", flush=True)
    print(f"  Gradient plot: outputs/sst_gradient_seasonal.png", flush=True)
    print(f"\nNext step: Run anomaly_detector.py", flush=True)
    print("=" * 55, flush=True)
