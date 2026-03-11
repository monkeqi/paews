# ============================================================
# PAEWS Data Refresh — March 2026
# ============================================================
# Run from: C:\Users\josep\Documents\paews
# Env: conda activate paews
#
# This script refreshes ALL stale data inputs. Run each section
# in order. Some steps require waiting for data availability.
# ============================================================

# ── 0. Setup ──
cd C:\Users\josep\Documents\paews
conda activate paews

# ============================================================
# 1. SST — refresh sst_current.nc from ERDDAP
#    Current: Feb 15 snapshot (3 weeks old)
#    Target: latest available daily OISST
# ============================================================

# Delete old file
# Remove-Item data\current\sst_current.nc -ErrorAction SilentlyContinue

# Download latest OISST daily for Peru box
# ERDDAP URL: NOAA OISST v2.1 daily
# This gets the last 90 days of SST data
python -c "
import xarray as xr
import pandas as pd
from pathlib import Path

url = (
    'https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.nc'
    '?sst[(2026-01-01T12:00:00Z):1:(2026-03-06T12:00:00Z)]'
    '[(0.0):1:(0.0)]'
    '[(-16.0):1:(-4.0)]'
    '[(278.0):1:(284.0)]'
)
print('Downloading OISST from ERDDAP...')
ds = xr.open_dataset(url)
out = Path('data/current/sst_current.nc')
ds.to_netcdf(out)
print(f'Saved: {out}')
print(f'Time range: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}')
print(f'Shape: {ds[\"sst\"].shape}')
ds.close()
"

# ============================================================
# 2. CHLOROPHYLL — refresh from Copernicus
#    Current: Dec 2025 monthly proxy (3 months old)
#    Target: Jan or Feb 2026 monthly
# ============================================================

# Check what's available first:
# Copernicus MY (multi-year, reprocessed) product has ~2 month latency
# Copernicus NRT (near-real-time) product has ~1 week latency
# The MY product is what we use for training consistency

# Option A: Check if Jan 2026 MY monthly is available
copernicusmarine subset `
    -i cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M `
    --variable CHL `
    -x -82 -X -76 -y -16 -Y -4 `
    -t "2026-01-01" -T "2026-01-31" `
    -o data/external -f chl_copernicus_jan2026.nc `
    --dry-run

# If available, download it:
copernicusmarine subset `
    -i cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M `
    --variable CHL `
    -x -82 -X -76 -y -16 -Y -4 `
    -t "2026-01-01" -T "2026-01-31" `
    -o data/external -f chl_copernicus_jan2026.nc

# If Jan not available yet, grab NRT daily for directional check:
copernicusmarine subset `
    -i cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D `
    --variable CHL `
    -x -82 -X -76 -y -16 -Y -4 `
    -t "2026-02-01" -T "2026-03-06" `
    -o data/external -f chl_nrt_2026_mar.nc

# Option B: Update the full Copernicus file and re-run migration
# (Do this when Jan 2026 MY monthly is confirmed available)
copernicusmarine subset `
    -i cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M `
    --variable CHL `
    -x -82 -X -76 -y -16 -Y -4 `
    -t "2003-01-01" -T "2026-02-28" `
    -o data/external -f chl_copernicus_full.nc

# Then recompute all Chl Z-scores:
python scripts/chl_migration.py

# ============================================================
# 3. NIÑO INDICES — already fresh (Feb 2026)
#    Just verify with a quick pull
# ============================================================

python scripts/external_data_puller.py

# ============================================================
# 4. GODAS THERMOCLINE — NEW PIPELINE
#    Downloads subsurface temperature, computes Z20 isotherm
#    First run takes ~10-15 minutes (OPeNDAP download)
# ============================================================

python scripts/godas_thermocline.py

# ============================================================
# 5. EVALUATE GODAS FEATURE
#    LOO test: does Z20 improve the model?
# ============================================================

python scripts/test_godas_feature.py

# ============================================================
# 6. RE-RUN PREDICTION with latest data
# ============================================================

python scripts/predict_2026_s1.py

# ============================================================
# 7. FISHMEAL PRICE — check for Jan 2026
#    World Bank Pink Sheet updates monthly
# ============================================================

# Manual: download from
# https://www.worldbank.org/en/research/commodity-markets
# File: CMO-Historical-Data-Monthly.xlsx
# Look for: Fish meal, 64-65% protein, Peru, FOB
# Save to: data/external/CMO-Historical-Data-Monthly.xlsx

# ============================================================
# 8. WEEKLY SST CHECK — manual download
#    Shows weekly Niño trajectory
# ============================================================

# Manual: download from
# https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for
# Save to: data/external/weekly_sst.txt

# ============================================================
# EXECUTION ORDER (recommended)
# ============================================================
# 1. SST refresh (5 min)
# 2. Niño indices (1 min)
# 3. GODAS pipeline (10-15 min, first run)
# 4. Test GODAS feature (1 min)
# 5. Chl refresh (when available)
# 6. Re-run prediction
# 7. Update dashboard if model changes
# ============================================================
