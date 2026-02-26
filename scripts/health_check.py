"""
PAEWS Health Check — Structural Integrity Validator
=====================================================
Run at the START of every new chat session to verify nothing
has been corrupted across sessions.

10 categories of checks:
  1. File existence
  2. Ground truth integrity
  3. Feature matrix structure
  4. Chl source consistency
  5. Physical plausibility
  6. Composite formula verification
  7. Script consistency
  8. Data coverage
  9. Known issue tracking
  10. Reproducibility hashes

Usage:
    cd C:\\Users\\josep\\Documents\\paews\\scripts
    python health_check.py
"""

import sys
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("PAEWS Health Check starting...", flush=True)

import numpy as np
import pandas as pd

print("Imports done", flush=True)

BASE_DIR = Path("c:/Users/josep/Documents/paews")
DATA_SST = BASE_DIR / "data" / "baseline_v2"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EXTERNAL = BASE_DIR / "data" / "external"
SCRIPTS = BASE_DIR / "scripts"

PASS = "✅"
WARN = "⚠️"
FAIL = "❌"

results = []


def check(category, name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((category, name, passed, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""), flush=True)
    return passed


def warn(category, name, detail=""):
    results.append((category, name, None, detail))
    print(f"  {WARN} {name}" + (f" — {detail}" if detail else ""), flush=True)


# =========================================================================
# 1. FILE EXISTENCE
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("1. FILE EXISTENCE", flush=True)
print("=" * 60, flush=True)

critical_files = {
    "Feature matrix": DATA_EXTERNAL / "paews_feature_matrix.csv",
    "Ground truth": DATA_EXTERNAL / "imarpe_ground_truth.csv",
    "Niño indices": DATA_EXTERNAL / "nino_indices_monthly.csv",
    "SST climatology v2": DATA_PROCESSED / "sst_climatology_v2.nc",
    "Chl climatology (Copernicus)": DATA_PROCESSED / "chl_climatology_copernicus.nc",
    "Copernicus Chl full": DATA_EXTERNAL / "chl_copernicus_full.nc",
    "SLA monthly": DATA_EXTERNAL / "sla_monthly_2010_2024.nc",
}

# Also check alternate SLA filename
sla_alt = DATA_EXTERNAL / "sla_2010_2024.nc"

for name, path in critical_files.items():
    exists = path.exists()
    if not exists and "SLA" in name:
        exists = sla_alt.exists()
        if exists:
            detail = f"found as {sla_alt.name}"
        else:
            detail = f"MISSING: {path}"
    else:
        detail = f"{path.stat().st_size / 1024:.0f} KB" if exists else f"MISSING: {path}"
    check("FILES", name, exists, detail)

# Scripts
critical_scripts = [
    "composite_score.py", "chl_migration.py", "sla_pipeline.py",
    "anomaly_detector.py", "health_check.py",
]
for script in critical_scripts:
    path = SCRIPTS / script
    check("FILES", f"Script: {script}", path.exists(),
          "present" if path.exists() else "MISSING")


# =========================================================================
# 2. GROUND TRUTH INTEGRITY
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("2. GROUND TRUTH INTEGRITY", flush=True)
print("=" * 60, flush=True)

gt_path = DATA_EXTERNAL / "imarpe_ground_truth.csv"
if gt_path.exists():
    gt = pd.read_csv(gt_path)
    check("GT", "Row count", len(gt) == 30, f"{len(gt)} rows (expect 30)")
    
    # Check outcomes
    outcomes = gt['outcome'].value_counts().to_dict()
    check("GT", "NORMAL count", outcomes.get('NORMAL', 0) == 18,
          f"NORMAL={outcomes.get('NORMAL', 0)} (expect 18)")
    check("GT", "DISRUPTED count", outcomes.get('DISRUPTED', 0) == 5,
          f"DISRUPTED={outcomes.get('DISRUPTED', 0)} (expect 5)")
    check("GT", "REDUCED count", outcomes.get('REDUCED', 0) == 6,
          f"REDUCED={outcomes.get('REDUCED', 0)} (expect 6)")
    
    # 2023 S1 must be CANCELLED
    s2023 = gt[(gt['year'] == 2023) & (gt['season'] == 1)]
    if len(s2023) > 0:
        check("GT", "2023 S1 = CANCELLED",
              s2023.iloc[0]['outcome'] == 'CANCELLED',
              s2023.iloc[0]['outcome'])
    else:
        check("GT", "2023 S1 exists", False, "ROW MISSING")
    
    # Year/season coverage
    years = sorted(gt['year'].unique())
    check("GT", "Year range", years[0] == 2010 and years[-1] == 2024,
          f"{years[0]}-{years[-1]}")
    
    # Each year should have exactly 2 seasons
    for y in range(2010, 2025):
        seasons = gt[gt['year'] == y]['season'].tolist()
        if sorted(seasons) != [1, 2]:
            check("GT", f"Year {y} seasons", False, f"has {seasons}")
else:
    check("GT", "File exists", False, "MISSING")


# =========================================================================
# 3. FEATURE MATRIX STRUCTURE
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("3. FEATURE MATRIX STRUCTURE", flush=True)
print("=" * 60, flush=True)

feat_path = DATA_EXTERNAL / "paews_feature_matrix.csv"
if feat_path.exists():
    df = pd.read_csv(feat_path)
    check("FEAT", "Row count", len(df) == 30, f"{len(df)} rows (expect 30)")
    
    # Required columns
    required_cols = ['year', 'season', 'outcome', 'target', 'sst_z', 'chl_z',
                     'nino12_t1', 'is_summer', 'bio_thresh_pct', 'chl_source',
                     'sla_z', 'composite_hard']
    missing_cols = [c for c in required_cols if c not in df.columns]
    check("FEAT", "Required columns", len(missing_cols) == 0,
          f"missing: {missing_cols}" if missing_cols else "all present")
    
    # Alignment with ground truth
    if gt_path.exists():
        gt_keys = set(zip(gt['year'], gt['season']))
        feat_keys = set(zip(df['year'], df['season']))
        check("FEAT", "Aligned with ground truth", gt_keys == feat_keys,
              f"match" if gt_keys == feat_keys else
              f"mismatch: GT-only={gt_keys - feat_keys}, Feat-only={feat_keys - gt_keys}")
    
    # Valid sample count (rows with complete core features)
    valid = df.dropna(subset=['sst_z', 'chl_z', 'nino12_t1'])
    check("FEAT", "Valid samples (SST+Chl+Niño)", len(valid) >= 28,
          f"{len(valid)} valid (expect ≥28)")
    
    # 2024 rows — expected to have NaN SST (no .nc files downloaded yet)
    rows_2024 = df[df['year'] == 2024]
    if len(rows_2024) > 0:
        sst_missing = rows_2024['sst_z'].isna().all()
        if sst_missing:
            warn("FEAT", "2024 SST missing (expected — need to download sst_2024.nc)")
else:
    check("FEAT", "File exists", False, "MISSING")
    df = None


# =========================================================================
# 4. CHL SOURCE CONSISTENCY
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("4. CHL SOURCE CONSISTENCY", flush=True)
print("=" * 60, flush=True)

if df is not None and 'chl_source' in df.columns:
    source_counts = df['chl_source'].value_counts().to_dict()
    cop_count = source_counts.get('Copernicus', 0) + source_counts.get('COP', 0)
    modis_count = source_counts.get('MODIS', 0)
    none_count = source_counts.get('NONE', 0) + source_counts.get(np.nan, 0)
    
    check("CHL", "All seasons use Copernicus", cop_count == 30,
          f"Copernicus={cop_count}, MODIS={modis_count}, NONE={none_count}")
    
    if cop_count < 30:
        warn("CHL", "OVERWRITE BUG DETECTED",
             "Run chl_migration.py BEFORE composite_score.py to fix")
    
    # Check no mixed baseline (critical bug from Session 3)
    if modis_count > 0 and cop_count > 0:
        check("CHL", "No mixed baseline", False,
              f"MIXED: {modis_count} MODIS + {cop_count} Copernicus — INVALID")
else:
    warn("CHL", "chl_source column missing", "cannot verify sensor consistency")


# =========================================================================
# 5. PHYSICAL PLAUSIBILITY
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("5. PHYSICAL PLAUSIBILITY", flush=True)
print("=" * 60, flush=True)

if df is not None:
    valid = df.dropna(subset=['sst_z', 'chl_z'])
    
    # Z-scores should be in reasonable range
    sst_range = (valid['sst_z'].min(), valid['sst_z'].max())
    check("PHYS", "SST Z-score range", -3 < sst_range[0] and sst_range[1] < 4,
          f"[{sst_range[0]:.2f}, {sst_range[1]:.2f}]")
    
    chl_range = (valid['chl_z'].min(), valid['chl_z'].max())
    check("PHYS", "Chl Z-score range", -3 < chl_range[0] and chl_range[1] < 3,
          f"[{chl_range[0]:.2f}, {chl_range[1]:.2f}]")
    
    # 2023 S1 should have the worst conditions
    s2023_s1 = df[(df['year'] == 2023) & (df['season'] == 1)]
    if len(s2023_s1) > 0 and not pd.isna(s2023_s1.iloc[0].get('sst_z')):
        sst_2023 = s2023_s1.iloc[0]['sst_z']
        check("PHYS", "2023 S1 has high SST", sst_2023 > 1.5,
              f"SST_Z={sst_2023:.2f} (expect >1.5)")
        
        bio_2023 = s2023_s1.iloc[0].get('bio_thresh_pct', np.nan)
        if not pd.isna(bio_2023):
            check("PHYS", "2023 S1 bio threshold extreme", bio_2023 > 90,
                  f"Bio>23°C = {bio_2023:.0f}% (expect >90%)")
    
    # Bio threshold: summer (S1, month=3) should generally be higher than winter (S2, month=10)
    s1_bio = valid[valid['season'] == 1]['bio_thresh_pct'].mean()
    s2_bio = valid[valid['season'] == 2]['bio_thresh_pct'].mean()
    if not np.isnan(s1_bio) and not np.isnan(s2_bio):
        check("PHYS", "S1 warmer than S2 (bio threshold)", s1_bio > s2_bio,
              f"S1 mean={s1_bio:.1f}%, S2 mean={s2_bio:.1f}%")
    
    # Nino indices should be correlated with SST
    nino_sst_corr = valid[['sst_z', 'nino12_t1']].corr().iloc[0, 1]
    check("PHYS", "SST-Niño correlation positive", nino_sst_corr > 0,
          f"r={nino_sst_corr:.3f}")


# =========================================================================
# 6. COMPOSITE FORMULA VERIFICATION
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("6. COMPOSITE FORMULA VERIFICATION", flush=True)
print("=" * 60, flush=True)

if df is not None:
    valid = df.dropna(subset=['sst_z', 'chl_z', 'composite_hard'])
    
    mismatches = 0
    for _, row in valid.iterrows():
        nino = row.get('nino12_t1', 0)
        if pd.isna(nino):
            nino = 0
        expected = 0.4 * row['sst_z'] + 0.4 * (-row['chl_z']) + 0.2 * nino
        actual = row['composite_hard']
        if abs(expected - actual) > 0.01:
            mismatches += 1
    
    check("COMP", "Hardcoded composite formula matches",
          mismatches == 0,
          f"{mismatches} mismatches" if mismatches > 0 else
          f"all {len(valid)} rows verified")


# =========================================================================
# 7. SCRIPT CONSISTENCY
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("7. SCRIPT CONSISTENCY", flush=True)
print("=" * 60, flush=True)

cs_path = SCRIPTS / "composite_score.py"
if cs_path.exists():
    cs_text = cs_path.read_text(encoding='utf-8', errors='replace')
    
    # 5 features, not 6 or 7
    check("SCRIPT", "Uses 5 features (not 6 or 7)",
          "['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']" in cs_text,
          "5-feature list found" if "['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']" in cs_text
          else "WRONG feature list — check composite_score.py")
    
    # SLA should NOT be in the regression feature list
    # (stored in CSV but excluded from regression)
    if "'sla_z', 'chl_z'" in cs_text or "'sla_z', 'sst_z'" in cs_text:
        check("SCRIPT", "SLA excluded from regression", False,
              "sla_z found in feature list — should be excluded")
    else:
        check("SCRIPT", "SLA excluded from regression", True, "not in feature list")
    
    # CSV-first architecture
    check("SCRIPT", "CSV-first (reads ledger)",
          "ledger" in cs_text.lower() or "paews_feature_matrix.csv" in cs_text,
          "CSV ledger pattern found")
    
    # class_weight='balanced'
    check("SCRIPT", "Balanced class weights",
          "class_weight='balanced'" in cs_text or 'class_weight="balanced"' in cs_text,
          "balanced weights confirmed")

cm_path = SCRIPTS / "chl_migration.py"
if cm_path.exists():
    cm_text = cm_path.read_text(encoding='utf-8', errors='replace')
    
    check("SCRIPT", "Coastal productivity mask present",
          "PRODUCTIVE_PERCENTILE" in cm_text or "productive_mask" in cm_text,
          "mask logic found")
    
    # Check percentile value
    import re
    pct_match = re.search(r'PRODUCTIVE_PERCENTILE\s*=\s*(\d+)', cm_text)
    if pct_match:
        pct_val = int(pct_match.group(1))
        check("SCRIPT", f"Mask percentile = {pct_val}",
              pct_val in [30, 40, 50, 60], f"set to {pct_val}%")


# =========================================================================
# 8. DATA COVERAGE
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("8. DATA COVERAGE", flush=True)
print("=" * 60, flush=True)

# SST baseline files
sst_files = sorted(DATA_SST.glob("sst_*.nc")) if DATA_SST.exists() else []
check("DATA", f"SST baseline files", len(sst_files) >= 20,
      f"{len(sst_files)} files (expect 20 for 2003-2022)")

# Check for 2024 SST (needed for full feature matrix)
sst_2024 = DATA_SST / "sst_2024.nc"
if not sst_2024.exists():
    warn("DATA", "sst_2024.nc missing",
         "2024 rows will have NaN SST — download to complete")

# Niño indices coverage
nino_path = DATA_EXTERNAL / "nino_indices_monthly.csv"
if nino_path.exists():
    nino = pd.read_csv(nino_path)
    latest = nino.iloc[-1]
    check("DATA", "Niño indices current",
          latest['year'] >= 2025,
          f"latest: {int(latest['year'])}-{int(latest['month']):02d}")

# Copernicus Chl
cop_path = DATA_EXTERNAL / "chl_copernicus_full.nc"
if cop_path.exists():
    size_mb = cop_path.stat().st_size / (1024 * 1024)
    check("DATA", "Copernicus Chl full dataset", size_mb > 100,
          f"{size_mb:.0f} MB (expect ~145 MB)")
else:
    check("DATA", "Copernicus Chl full dataset", False, "MISSING")


# =========================================================================
# 9. KNOWN ISSUE TRACKING
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("9. KNOWN ISSUE STATUS", flush=True)
print("=" * 60, flush=True)

if df is not None:
    valid = df.dropna(subset=['sst_z', 'chl_z', 'nino12_t1'])
    
    # 2014 S1 ghost miss — should have prob ~0.16 (low)
    s2014 = valid[(valid['year'] == 2014) & (valid['season'] == 1)]
    if len(s2014) > 0:
        warn("KNOWN", "2014 S1 ghost miss (subsurface Kelvin wave)",
             f"SST_Z={s2014.iloc[0]['sst_z']:+.2f}, SLA_Z={s2014.iloc[0].get('sla_z', np.nan):+.2f} — "
             "needs weekly SLA or biomass to fix")
    
    # 2022 S2 cold disruption
    s2022 = valid[(valid['year'] == 2022) & (valid['season'] == 2)]
    if len(s2022) > 0:
        warn("KNOWN", "2022 S2 cold-water disruption",
             f"SST_Z={s2022.iloc[0]['sst_z']:+.2f}, Chl_Z={s2022.iloc[0]['chl_z']:+.2f} — "
             "needs biomass data to fix")
    
    # False alarm cluster 2018-2020
    fa_years = [(2018, 2), (2019, 1), (2019, 2), (2020, 1)]
    fa_count = 0
    for y, s in fa_years:
        row = valid[(valid['year'] == y) & (valid['season'] == s)]
        if len(row) > 0 and row.iloc[0]['target'] == 0:
            fa_count += 1
    warn("KNOWN", f"False alarm cluster (2018-2020)",
         f"{fa_count} NORMAL seasons with elevated risk — needs biomass")


# =========================================================================
# 10. REPRODUCIBILITY HASHES
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("10. REPRODUCIBILITY HASHES", flush=True)
print("=" * 60, flush=True)

hash_files = [
    DATA_EXTERNAL / "paews_feature_matrix.csv",
    DATA_EXTERNAL / "imarpe_ground_truth.csv",
    SCRIPTS / "composite_score.py",
    SCRIPTS / "chl_migration.py",
]

for fpath in hash_files:
    if fpath.exists():
        md5 = hashlib.md5(fpath.read_bytes()).hexdigest()[:12]
        print(f"  {fpath.name}: {md5}", flush=True)
    else:
        print(f"  {fpath.name}: FILE MISSING", flush=True)


# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("HEALTH CHECK SUMMARY", flush=True)
print("=" * 60, flush=True)

total = len(results)
passed = sum(1 for _, _, p, _ in results if p is True)
failed = sum(1 for _, _, p, _ in results if p is False)
warned = sum(1 for _, _, p, _ in results if p is None)

print(f"  {PASS} Passed: {passed}", flush=True)
print(f"  {FAIL} Failed: {failed}", flush=True)
print(f"  {WARN} Warnings: {warned}", flush=True)
print(f"  Total checks: {total}", flush=True)

if failed > 0:
    print(f"\n  {FAIL} FAILURES:", flush=True)
    for cat, name, p, detail in results:
        if p is False:
            print(f"    [{cat}] {name}: {detail}", flush=True)

if failed == 0:
    print(f"\n  {PASS} ALL CHECKS PASSED — system is clean", flush=True)
else:
    print(f"\n  {FAIL} FIX FAILURES BEFORE PROCEEDING", flush=True)
    print(f"  Most common fix: run chl_migration.py then composite_score.py", flush=True)

print("=" * 60, flush=True)
