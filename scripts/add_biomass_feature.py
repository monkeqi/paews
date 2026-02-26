"""
add_biomass_feature.py - Add lagged biomass to PAEWS feature matrix

Usage:
    cd C:/Users/josep/Documents/paews/scripts
    python add_biomass_feature.py

What it does:
    1. Reads imarpe_biomass_verified.csv (10 verified values, 16 NaN)
    2. Reads paews_feature_matrix.csv (30 seasons)
    3. Adds 'biomass_lag1' column: previous season's biomass predicts current outcome
    4. Reruns logistic regression with biomass as additional feature
    5. Reports PR-AUC with and without biomass
    6. Handles NaN gracefully (only trains on rows where lag is available)

The lag logic:
    - 2022 S2 outcome is predicted using 2022 S1 biomass (7.13 MMT)
    - 2022 S1 outcome is predicted using 2021 S2 biomass (8.03 MMT)
    - If previous season biomass is NaN, that row is excluded from training
      but can still get a prediction using the non-biomass features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

# === PATHS (adjust if your layout differs) ===
BASE = Path(__file__).resolve().parent.parent
BIOMASS_CSV = BASE / "data" / "external" / "imarpe_biomass_verified.csv"
FEATURE_CSV = BASE / "data" / "external" / "paews_feature_matrix.csv"
OUTPUT_CSV  = BASE / "data" / "external" / "paews_feature_matrix_with_biomass.csv"

def load_biomass():
    """Load biomass CSV and create a lookup dict: (year, season) -> biomass_mmt"""
    df = pd.read_csv(BIOMASS_CSV)
    lookup = {}
    for _, row in df.iterrows():
        if row['verified'] == True or str(row['verified']).upper() == 'TRUE':
            bm = row['biomass_mmt']
            if pd.notna(bm):
                lookup[(int(row['year']), row['season'])] = float(bm)
    return lookup

def get_previous_season(year, season):
    """Return (year, season) of the previous season.
    Handles both string ('S1','S2') and int (1,2) formats."""
    s = str(season).strip()
    if s in ('S1', '1'):
        return (year - 1, 'S2')
    else:
        return (year, 'S1')

def add_lagged_biomass(features_df, biomass_lookup):
    """Add biomass_lag1 column to feature matrix."""
    lag_values = []
    for _, row in features_df.iterrows():
        yr = int(row['year'])
        sn = row['season']
        prev_yr, prev_sn = get_previous_season(yr, sn)
        bm = biomass_lookup.get((prev_yr, prev_sn), np.nan)
        lag_values.append(bm)
    features_df['biomass_lag1'] = lag_values
    return features_df

def run_comparison(df):
    """Run LOO-CV with and without biomass, compare PR-AUC."""
    
    # Identify target column (try common names)
    target_col = None
    for candidate in ['target', 'at_risk', 'label', 'reduced']:
        if candidate in df.columns:
            target_col = candidate
            break
    
    if target_col is None:
        # Try to create binary target from outcome text
        if 'outcome' in df.columns:
            df['target'] = df['outcome'].apply(
                lambda x: 1 if str(x).upper() in ['REDUCED', 'DISRUPTED', 'CANCELLED', 'AT-RISK', 'AT_RISK'] else 0
            )
            target_col = 'target'
        else:
            print("ERROR: Cannot find target column. Available columns:")
            print(list(df.columns))
            return
    
    y = df[target_col].values
    
    # Base features (the ones currently in the model)
    # Try to identify them from column names
    base_candidates = ['chl_z', 'sst_z', 'chl_anom', 'sst_anom', 
                       'Chl_Z', 'SST_Z', 'Chl_anom', 'SST_anom',
                       'bio_thresh_pct', 'nino12_t1', 'is_summer']
    base_features = [c for c in base_candidates if c in df.columns]
    
    if len(base_features) == 0:
        print("WARNING: Could not auto-detect base features.")
        print("Available columns:", list(df.columns))
        print("Please edit the script to specify your feature columns.")
        return
    
    print(f"Base features found: {base_features}")
    print(f"Target column: {target_col}")
    print(f"Total samples: {len(df)}")
    print(f"  AT-RISK: {sum(y)}, NORMAL: {len(y) - sum(y)}")
    print()
    
    # === Model A: Without biomass (baseline) ===
    print("=" * 60)
    print("MODEL A: Without biomass (current baseline)")
    print("=" * 60)
    prauc_a = loo_cv(df, base_features, target_col)
    
    # === Model B: With lagged biomass ===
    print()
    print("=" * 60)
    print("MODEL B: With lagged biomass")
    print("=" * 60)
    
    has_biomass = df['biomass_lag1'].notna()
    n_with = has_biomass.sum()
    n_without = (~has_biomass).sum()
    print(f"Samples with biomass_lag1: {n_with}")
    print(f"Samples without (NaN): {n_without}")
    
    if n_with < 8:
        print("Too few samples with biomass to run meaningful comparison.")
        return
    
    # Only train/evaluate on rows with biomass available
    df_bm = df[has_biomass].copy()
    features_bm = base_features + ['biomass_lag1']
    prauc_b = loo_cv(df_bm, features_bm, target_col)
    
    # === Model C: Baseline but same subset (fair comparison) ===
    print()
    print("=" * 60)
    print("MODEL C: Without biomass, SAME SUBSET (fair comparison)")
    print("=" * 60)
    prauc_c = loo_cv(df_bm, base_features, target_col)
    
    # === Summary ===
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model A (no biomass, all 30):     PR-AUC = {prauc_a:.3f}")
    print(f"Model C (no biomass, {n_with} subset): PR-AUC = {prauc_c:.3f}")
    print(f"Model B (with biomass, {n_with} subset): PR-AUC = {prauc_b:.3f}")
    print()
    if prauc_b > prauc_c:
        print(f"✅ Biomass IMPROVED PR-AUC by {prauc_b - prauc_c:+.3f} on the same subset")
    else:
        print(f"⚠️  Biomass did not improve PR-AUC ({prauc_b - prauc_c:+.3f})")
    print()
    print("NOTE: The subset comparison (B vs C) is the fair test.")
    print("If B > C, finding the remaining 16 biomass values is worth the effort.")

def loo_cv(df, features, target_col):
    """Leave-one-out cross-validation, returns PR-AUC."""
    X = df[features].values
    y = df[target_col].values
    n = len(y)
    
    probs = np.zeros(n)
    skipped = 0
    
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]
        
        # Skip if only one class in training set
        if len(np.unique(y_train)) < 2:
            # Assign base rate as probability
            probs[i] = np.mean(y_train)
            skipped += 1
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_train_s, y_train)
        probs[i] = model.predict_proba(X_test_s)[0, 1]
    
    n_pos = int(sum(y))
    try:
        prauc = average_precision_score(y, probs)
    except:
        prauc = float('nan')
    try:
        rocauc = roc_auc_score(y, probs)
    except:
        rocauc = float('nan')
    
    print(f"  PR-AUC:  {prauc:.3f}")
    print(f"  ROC-AUC: {rocauc:.3f}")
    print(f"  Samples: {n} ({n_pos} positive, {n - n_pos} negative)")
    if skipped > 0:
        print(f"  Skipped folds (single-class): {skipped}")
    if n_pos <= 2:
        print(f"  ⚠️  Only {n_pos} positive sample(s) — metrics are unreliable at this sample size")
    
    # Show per-season probabilities for the subset
    if n <= 20:
        print(f"  Per-season:")
        for idx, (_, row) in enumerate(df.iterrows()):
            yr = int(row['year'])
            sn = row['season']
            actual = int(row[target_col])
            prob = probs[idx]
            flag = "✓" if (prob >= 0.38) == (actual == 1) else "✗"
            bm_str = f" [lag={row['biomass_lag1']:.1f}]" if 'biomass_lag1' in row.index and pd.notna(row.get('biomass_lag1')) else ""
            print(f"    {yr} {sn}: actual={actual} prob={prob:.2f} {flag}{bm_str}")
    
    return prauc

def main():
    print("PAEWS Biomass Integration Test")
    print("=" * 60)
    
    # Check files exist
    if not BIOMASS_CSV.exists():
        print(f"ERROR: {BIOMASS_CSV} not found")
        print("Copy imarpe_biomass_verified.csv to data/external/")
        return
    
    if not FEATURE_CSV.exists():
        print(f"ERROR: {FEATURE_CSV} not found")
        return
    
    # Load data
    biomass_lookup = load_biomass()
    print(f"Loaded {len(biomass_lookup)} verified biomass values")
    for (yr, sn), bm in sorted(biomass_lookup.items()):
        print(f"  {yr} {sn}: {bm} MMT")
    print()
    
    features_df = pd.read_csv(FEATURE_CSV)
    print(f"Loaded feature matrix: {len(features_df)} rows, {len(features_df.columns)} columns")
    print(f"Columns: {list(features_df.columns)}")
    print()
    
    # Add lagged biomass
    features_df = add_lagged_biomass(features_df, biomass_lookup)
    
    # Show coverage
    has_lag = features_df['biomass_lag1'].notna()
    print(f"Lagged biomass coverage: {has_lag.sum()}/{len(features_df)} seasons")
    for _, row in features_df.iterrows():
        yr = int(row['year'])
        sn = row['season']
        lag = row['biomass_lag1']
        status = f"{lag:.2f} MMT" if pd.notna(lag) else "NaN"
        print(f"  {yr} {sn}: biomass_lag1 = {status}")
    print()
    
    # Save augmented CSV
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved augmented feature matrix to: {OUTPUT_CSV}")
    print()
    
    # Run comparison
    run_comparison(features_df)

if __name__ == "__main__":
    main()
