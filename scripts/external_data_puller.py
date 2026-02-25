"""
PAEWS External Data Puller
===========================
Downloads three external datasets for the composite score and validation:

1. NOAA Niño 1+2 SST Index (monthly, 1950-present)
   - Region: 0-10°S, 90°W-80°W (directly off Peru)
   - This is THE most relevant ENSO index for Peru anchovy
   
2. NOAA ONI (Oceanic Niño Index, Niño 3.4 region)
   - The official ENSO monitoring index
   - 3-month running mean of SST anomalies

3. World Bank Fishmeal Prices (monthly, 1960-present)
   - "Fish meal, German fishmeal, Danish 64% pro, FOB Bremen"
   - Used to validate economic impact of anomalies

Usage:
    python external_data_puller.py
    python external_data_puller.py --nino-only
    python external_data_puller.py --prices-only
"""

import sys
print("PAEWS External Data Puller starting...", flush=True)

import argparse
import os
import re
import csv
from pathlib import Path
from datetime import datetime

# Try imports, guide user if missing
try:
    import requests
    import pandas as pd
except ImportError as e:
    print(f"Missing package: {e}. Run: pip install requests pandas openpyxl", flush=True)
    sys.exit(1)

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_EXTERNAL = BASE_DIR / "data" / "external"
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. NOAA Niño Indices
# =============================================================================

# CPC monthly SST indices file — contains Niño 1+2, 3, 3.4, 4
NINO_INDICES_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"

# ONI (3-month running mean, official ENSO definition)
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# PSL monthly Niño 1+2 (backup source)
PSL_NINO12_URL = "https://psl.noaa.gov/data/correlation/nina12.anom.data"


def download_nino_indices():
    """Download Niño SST indices from CPC."""
    print("\n--- Downloading NOAA Niño Indices ---", flush=True)
    
    # Try primary source: CPC sstoi.indices
    print(f"  Fetching {NINO_INDICES_URL}...", flush=True)
    try:
        resp = requests.get(NINO_INDICES_URL, timeout=30)
        resp.raise_for_status()
        
        # Parse the fixed-width format
        # Format: YEAR MON NINO1+2_total NINO1+2_anom NINO3_total NINO3_anom NINO34_total NINO34_anom NINO4_total NINO4_anom
        # We want the ANOMALY columns (indices 3, 5, 7, 9)
        lines = resp.text.strip().split('\n')
        records = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 10:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    nino12 = float(parts[3])  # Niño 1+2 anomaly (col 3)
                    nino3 = float(parts[5])   # Niño 3 anomaly (col 5)
                    nino34 = float(parts[7])  # Niño 3.4 anomaly (col 7)
                    nino4 = float(parts[9])   # Niño 4 anomaly (col 9)
                    records.append({
                        'year': year, 'month': month,
                        'date': f"{year}-{month:02d}-01",
                        'nino12_anom': nino12,
                        'nino3_anom': nino3,
                        'nino34_anom': nino34,
                        'nino4_anom': nino4,
                    })
                except (ValueError, IndexError):
                    continue
        
        if records:
            df = pd.DataFrame(records)
            outpath = DATA_EXTERNAL / "nino_indices_monthly.csv"
            df.to_csv(outpath, index=False)
            print(f"  Saved {len(df)} months: {outpath}", flush=True)
            print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}", flush=True)
            print(f"  Latest Niño 1+2: {df['nino12_anom'].iloc[-1]:+.2f}°C", flush=True)
            print(f"  Latest Niño 3.4: {df['nino34_anom'].iloc[-1]:+.2f}°C", flush=True)
            return df
        else:
            print("  WARNING: Could not parse CPC indices", flush=True)
            
    except Exception as e:
        print(f"  CPC source failed: {e}", flush=True)
    
    # Try backup: PSL Niño 1+2
    print(f"  Trying backup: {PSL_NINO12_URL}...", flush=True)
    try:
        resp = requests.get(PSL_NINO12_URL, timeout=30)
        resp.raise_for_status()
        
        lines = resp.text.strip().split('\n')
        records = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 13:
                try:
                    year = int(parts[0])
                    if year < 1950 or year > 2030:
                        continue
                    for m in range(1, 13):
                        val = float(parts[m])
                        if val > -90:  # -99.99 = missing
                            records.append({
                                'year': year, 'month': m,
                                'date': f"{year}-{m:02d}-01",
                                'nino12_anom': val,
                            })
                except (ValueError, IndexError):
                    continue
        
        if records:
            df = pd.DataFrame(records)
            outpath = DATA_EXTERNAL / "nino12_monthly.csv"
            df.to_csv(outpath, index=False)
            print(f"  Saved {len(df)} months (Niño 1+2 only): {outpath}", flush=True)
            return df
            
    except Exception as e:
        print(f"  Backup source also failed: {e}", flush=True)
    
    print("  FAILED: Could not download any Niño data", flush=True)
    return None


def download_oni():
    """Download ONI (Oceanic Niño Index) separately."""
    print(f"\n  Fetching ONI: {ONI_URL}...", flush=True)
    try:
        resp = requests.get(ONI_URL, timeout=30)
        resp.raise_for_status()
        
        lines = resp.text.strip().split('\n')
        records = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 4:
                try:
                    season = parts[0]  # e.g., "DJF"
                    year = int(parts[1])
                    total = float(parts[2])
                    anom = float(parts[3])
                    records.append({
                        'season': season, 'year': year,
                        'sst_total': total, 'oni_anom': anom,
                    })
                except (ValueError, IndexError):
                    continue
        
        if records:
            df = pd.DataFrame(records)
            outpath = DATA_EXTERNAL / "oni_monthly.csv"
            df.to_csv(outpath, index=False)
            print(f"  Saved {len(df)} ONI records: {outpath}", flush=True)
            return df
            
    except Exception as e:
        print(f"  ONI download failed: {e}", flush=True)
    
    return None


# =============================================================================
# 2. World Bank Fishmeal Prices
# =============================================================================

PINK_SHEET_URL = "https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx"


def download_fishmeal_prices():
    """Download World Bank Pink Sheet and extract fishmeal prices."""
    print("\n--- Downloading World Bank Fishmeal Prices ---", flush=True)
    print(f"  Fetching Pink Sheet: {PINK_SHEET_URL}...", flush=True)
    
    try:
        # Need openpyxl for xlsx
        try:
            import openpyxl
        except ImportError:
            print("  Need openpyxl: pip install openpyxl", flush=True)
            return None
        
        resp = requests.get(PINK_SHEET_URL, timeout=60)
        resp.raise_for_status()
        
        # Save raw file
        raw_path = DATA_EXTERNAL / "CMO-Historical-Data-Monthly.xlsx"
        with open(raw_path, 'wb') as f:
            f.write(resp.content)
        print(f"  Saved raw: {raw_path} ({len(resp.content)/1024/1024:.1f} MB)", flush=True)
        
        # Parse — fishmeal is in the "Monthly Prices" sheet
        # The column header is "Fish meal" or "Fishmeal"
        try:
            df_raw = pd.read_excel(raw_path, sheet_name="Monthly Prices", header=None)
            
            # Find the fishmeal column
            fishmeal_col = None
            date_col = None
            header_row = None
            
            for i in range(min(10, len(df_raw))):
                for j in range(len(df_raw.columns)):
                    val = str(df_raw.iloc[i, j]).lower()
                    if 'fish' in val and 'meal' in val:
                        fishmeal_col = j
                        header_row = i
                    if val in ['period', 'month', 'date'] or '1960' in val:
                        date_col = j
            
            if fishmeal_col is None:
                # Try alternate approach — look in all cells
                for i in range(min(20, len(df_raw))):
                    row_str = ' '.join([str(x).lower() for x in df_raw.iloc[i]])
                    if 'fish' in row_str and 'meal' in row_str:
                        header_row = i
                        for j in range(len(df_raw.columns)):
                            val = str(df_raw.iloc[i, j]).lower()
                            if 'fish' in val:
                                fishmeal_col = j
                                break
                        break
            
            if header_row is not None and fishmeal_col is not None:
                # Data starts after header
                if date_col is None:
                    date_col = 0  # Usually first column
                
                records = []
                for i in range(header_row + 1, len(df_raw)):
                    date_val = df_raw.iloc[i, date_col]
                    price_val = df_raw.iloc[i, fishmeal_col]
                    
                    if pd.isna(date_val) or pd.isna(price_val):
                        continue
                    
                    try:
                        price = float(price_val)
                    except (ValueError, TypeError):
                        continue
                    
                    # Parse date — could be datetime, string like "1960M01", etc.
                    date_str = str(date_val)
                    if hasattr(date_val, 'strftime'):
                        date_parsed = date_val.strftime('%Y-%m-01')
                        year = date_val.year
                        month = date_val.month
                    elif 'M' in date_str:
                        parts = date_str.split('M')
                        year = int(parts[0])
                        month = int(parts[1])
                        date_parsed = f"{year}-{month:02d}-01"
                    else:
                        continue
                    
                    records.append({
                        'date': date_parsed,
                        'year': year,
                        'month': month,
                        'fishmeal_price_usd_mt': price,
                    })
                
                if records:
                    df = pd.DataFrame(records)
                    outpath = DATA_EXTERNAL / "fishmeal_prices_monthly.csv"
                    df.to_csv(outpath, index=False)
                    print(f"  Extracted {len(df)} months of fishmeal prices", flush=True)
                    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}", flush=True)
                    print(f"  Latest price: ${df['fishmeal_price_usd_mt'].iloc[-1]:.0f}/MT", flush=True)
                    return df
            
            print("  Could not find fishmeal column in Pink Sheet", flush=True)
            print("  Raw file saved — you can extract manually", flush=True)
            
        except Exception as e:
            print(f"  Excel parsing failed: {e}", flush=True)
            print(f"  Raw file saved at: {raw_path}", flush=True)
    
    except Exception as e:
        print(f"  Download failed: {e}", flush=True)
    
    return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAEWS External Data Puller")
    parser.add_argument("--nino-only", action="store_true", help="Only download Niño indices")
    parser.add_argument("--prices-only", action="store_true", help="Only download fishmeal prices")
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("PAEWS External Data Puller", flush=True)
    print(f"Output directory: {DATA_EXTERNAL}", flush=True)
    print("=" * 60, flush=True)
    
    if args.prices_only:
        download_fishmeal_prices()
    elif args.nino_only:
        download_nino_indices()
        download_oni()
    else:
        # Download everything
        nino_df = download_nino_indices()
        oni_df = download_oni()
        price_df = download_fishmeal_prices()
        
        print("\n" + "=" * 60, flush=True)
        print("DOWNLOAD SUMMARY", flush=True)
        print(f"  Niño indices: {'OK' if nino_df is not None else 'FAILED'}", flush=True)
        print(f"  ONI: {'OK' if oni_df is not None else 'FAILED'}", flush=True)
        print(f"  Fishmeal prices: {'OK' if price_df is not None else 'FAILED'}", flush=True)
        print(f"  Output: {DATA_EXTERNAL}", flush=True)
        print("=" * 60, flush=True)
