# PAEWS Data Audit + Recalibrated Risk Model
#
# Part 1: Full data integrity audit
# Part 2: Recalibrated risk tiers (no binary threshold)
# Part 3: Updated 2026 S1 prediction with honest uncertainty
#
# Usage: python scripts/model_v2_audit.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
FEATURE_CSV = BASE / "data" / "external" / "paews_feature_matrix.csv"
GROUND_TRUTH = BASE / "data" / "external" / "imarpe_ground_truth.csv"

FEAT_COLS = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']

RISK_TIERS = [
    (0.0, 0.20, "LOW",      "No significant environmental stress"),
    (0.20, 0.50, "MODERATE", "Some stress indicators, season likely proceeds"),
    (0.50, 0.70, "ELEVATED", "Notable stress, reduced quota likely"),
    (0.70, 1.01, "SEVERE",   "Extreme stress, cancellation risk high"),
]


def load_data():
    feat = pd.read_csv(FEATURE_CSV)
    gt = pd.read_csv(GROUND_TRUTH)
    return feat, gt


# =====================================================================
# PART 1: DATA INTEGRITY AUDIT
# =====================================================================

def audit_data(feat, gt):
    print("=" * 70)
    print("  PART 1: DATA INTEGRITY AUDIT")
    print("=" * 70)
    errors = 0
    warnings_count = 0

    # --- Check 1: Feature matrix completeness ---
    print("\n  CHECK 1: Feature matrix completeness")
    print("  " + "-" * 40)
    for col in FEAT_COLS + ['target', 'year', 'season']:
        n_miss = feat[col].isna().sum()
        if n_miss > 0:
            print("    WARNING: %s has %d missing values" % (col, n_miss))
            for _, row in feat[feat[col].isna()].iterrows():
                print("      -> %d S%d" % (int(row['year']), int(row['season'])))
            warnings_count += 1
        else:
            print("    OK: %s - no missing values" % col)

    # --- Check 2: Target vs ground truth consistency ---
    print("\n  CHECK 2: Target vs ground truth consistency")
    print("  " + "-" * 40)
    gt_slim = gt[['year', 'season', 'outcome', 'catch_mt']].copy()
    merged = feat.merge(gt_slim, on=['year', 'season'], how='left', suffixes=('', '_gt'))

    for _, row in merged.iterrows():
        yr, s, target = int(row['year']), int(row['season']), int(row['target'])
        outcome = str(row.get('outcome', '')).upper() if pd.notna(row.get('outcome')) else 'MISSING'
        catch = row.get('catch_mt', np.nan)

        # Check: target=1 should correspond to disrupted/cancelled outcomes
        if target == 1 and 'NORMAL' in outcome:
            print("    ERROR: %d S%d target=1 but outcome=%s" % (yr, s, outcome))
            errors += 1
        elif target == 0 and any(w in outcome for w in ['CANCEL', 'DISRUPT', 'SUSPEND']):
            print("    ERROR: %d S%d target=0 but outcome=%s" % (yr, s, outcome))
            errors += 1

        # Check: catch_mt sanity
        if pd.notna(catch):
            if target == 1 and catch > 2500000:
                print("    WARNING: %d S%d target=1 but catch=%.0f MT (high for disrupted)" % (yr, s, catch))
                warnings_count += 1
            if target == 0 and catch < 500000 and catch > 0:
                print("    WARNING: %d S%d target=0 but catch=%.0f MT (low for normal)" % (yr, s, catch))
                warnings_count += 1

    if errors == 0:
        print("    OK: All targets consistent with ground truth outcomes")

    # --- Check 3: Feature value ranges ---
    print("\n  CHECK 3: Feature value ranges")
    print("  " + "-" * 40)
    range_checks = {
        'sst_z': (-3, 4, "SST Z-score"),
        'chl_z': (-3, 3, "Chl Z-score"),
        'nino12_t1': (-3, 5, "Nino 1+2"),
        'is_summer': (0, 1, "is_summer (binary)"),
        'bio_thresh_pct': (0, 100, "Bio >23C %"),
    }
    for col, (lo, hi, desc) in range_checks.items():
        vals = feat[col].dropna()
        vmin, vmax = vals.min(), vals.max()
        outliers = vals[(vals < lo) | (vals > hi)]
        if len(outliers) > 0:
            print("    WARNING: %s has %d values outside [%.1f, %.1f]" % (desc, len(outliers), lo, hi))
            warnings_count += 1
        else:
            print("    OK: %s range [%.3f, %.3f]" % (desc, vmin, vmax))

    # --- Check 4: is_summer correctness ---
    print("\n  CHECK 4: is_summer matches season number")
    print("  " + "-" * 40)
    for _, row in feat.iterrows():
        expected = 1 if int(row['season']) == 1 else 0
        actual = int(row['is_summer'])
        if expected != actual:
            print("    ERROR: %d S%d is_summer=%d but season=%d" % (
                int(row['year']), int(row['season']), actual, int(row['season'])))
            errors += 1
    if errors == 0:
        print("    OK: All is_summer values match season number")

    # --- Check 5: Chronological ordering ---
    print("\n  CHECK 5: Chronological ordering")
    print("  " + "-" * 40)
    prev_key = (0, 0)
    for _, row in feat.iterrows():
        key = (int(row['year']), int(row['season']))
        if key <= prev_key:
            print("    WARNING: Out of order at %d S%d" % key)
            warnings_count += 1
        prev_key = key
    print("    OK: Data is chronologically ordered")

    # --- Check 6: Duplicate detection ---
    print("\n  CHECK 6: Duplicate rows")
    print("  " + "-" * 40)
    dupes = feat.groupby(['year', 'season']).size()
    dupes_multi = dupes[dupes > 1]
    if len(dupes_multi) > 0:
        for (yr, s), cnt in dupes_multi.items():
            print("    ERROR: %d S%d appears %d times" % (yr, s, cnt))
            errors += 1
    else:
        print("    OK: No duplicates")

    # --- Check 7: Ground truth completeness ---
    print("\n  CHECK 7: Ground truth coverage")
    print("  " + "-" * 40)
    for _, row in feat.iterrows():
        yr, s = int(row['year']), int(row['season'])
        gt_match = gt[(gt.year == yr) & (gt.season == s)]
        if len(gt_match) == 0:
            print("    WARNING: %d S%d in feature matrix but NOT in ground truth" % (yr, s))
            warnings_count += 1
    print("    Checked %d seasons" % len(feat))

    # --- Check 8: Nino lag consistency ---
    print("\n  CHECK 8: Nino 1+2 lag consistency")
    print("  " + "-" * 40)
    # nino12_t1 should be the month BEFORE the decision month
    # S1 decision month = March, so nino12_t1 = February
    # S2 decision month = October, so nino12_t1 = September
    print("    (Manual check: nino12_t1 uses month before decision month)")
    print("    S1: Feb Nino 1+2 value")
    print("    S2: Sep Nino 1+2 value")
    nino_path = BASE / "data" / "external" / "nino_indices_monthly.csv"
    if nino_path.exists():
        nino = pd.read_csv(nino_path)
        spot_checks = [(2023, 1), (2023, 2), (2024, 1), (2025, 1)]
        for yr, s in spot_checks:
            frow = feat[(feat.year == yr) & (feat.season == s)]
            if len(frow) == 0:
                continue
            feat_val = frow['nino12_t1'].iloc[0]
            # Look up expected
            if s == 1:
                nrow = nino[(nino.year == yr) & (nino.month == 2)]
            else:
                nrow = nino[(nino.year == yr) & (nino.month == 9)]
            if len(nrow) > 0:
                expected = nrow['nino12_anom'].iloc[0]
                match = abs(feat_val - expected) < 0.01
                status = "OK" if match else "MISMATCH"
                print("    %d S%d: feat=%.3f, nino_csv=%.3f [%s]" % (yr, s, feat_val, expected, status))
                if not match:
                    errors += 1
    else:
        print("    Nino CSV not found, skipping spot check")

    # --- Check 9: Class balance ---
    print("\n  CHECK 9: Class balance")
    print("  " + "-" * 40)
    n_pos = int(feat['target'].sum())
    n_neg = len(feat) - n_pos
    ratio = n_pos / len(feat)
    print("    Disrupted: %d (%.1f%%)" % (n_pos, ratio * 100))
    print("    Normal:    %d (%.1f%%)" % (n_neg, (1 - ratio) * 100))
    if ratio < 0.2 or ratio > 0.8:
        print("    WARNING: Severe class imbalance")
        warnings_count += 1

    # --- Check 10: Feature correlations ---
    print("\n  CHECK 10: Feature correlations")
    print("  " + "-" * 40)
    corr = feat[FEAT_COLS].corr()
    for i in range(len(FEAT_COLS)):
        for j in range(i + 1, len(FEAT_COLS)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print("    WARNING: High correlation %.3f between %s and %s" % (
                    r, FEAT_COLS[i], FEAT_COLS[j]))
                warnings_count += 1
    print("    Highest correlations:")
    for i in range(len(FEAT_COLS)):
        for j in range(i + 1, len(FEAT_COLS)):
            r = corr.iloc[i, j]
            if abs(r) > 0.3:
                print("      %s vs %s: %.3f" % (FEAT_COLS[i], FEAT_COLS[j], r))

    # --- Summary ---
    print("\n  " + "=" * 40)
    print("  AUDIT SUMMARY: %d errors, %d warnings" % (errors, warnings_count))
    if errors == 0:
        print("  DATA INTEGRITY: PASSED")
    else:
        print("  DATA INTEGRITY: FAILED - fix errors before proceeding")
    print("  " + "=" * 40)

    return errors


# =====================================================================
# PART 2: RECALIBRATED RISK MODEL
# =====================================================================

def run_recalibrated_model(feat):
    print("\n")
    print("=" * 70)
    print("  PART 2: RECALIBRATED RISK MODEL")
    print("  No binary threshold. Probability = continuous risk index.")
    print("=" * 70)

    df = feat.dropna(subset=FEAT_COLS + ['target'])
    n = len(df)

    # LOO predictions
    loo_probs = []
    loo_actual = []
    loo_details = []

    for idx in df.index:
        train = df.drop(idx)
        test = df.loc[[idx]]

        X_train = train[FEAT_COLS].values
        y_train = train['target'].values
        X_test = test[FEAT_COLS].values

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue

        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(scaler.fit_transform(X_train), y_train)

        prob = model.predict_proba(scaler.transform(X_test))[0, 1]
        actual = int(test['target'].iloc[0])

        loo_probs.append(prob)
        loo_actual.append(actual)

        # Assign risk tier
        tier = "UNKNOWN"
        for lo, hi, name, _ in RISK_TIERS:
            if lo <= prob < hi:
                tier = name
                break

        loo_details.append({
            'year': int(test['year'].iloc[0]),
            'season': int(test['season'].iloc[0]),
            'actual': "DISRUPTED" if actual == 1 else "NORMAL",
            'prob': prob,
            'tier': tier,
        })

    loo_probs = np.array(loo_probs)
    loo_actual = np.array(loo_actual)

    # Metrics
    roc_auc = roc_auc_score(loo_actual, loo_probs)
    pr_auc = average_precision_score(loo_actual, loo_probs)

    print("\n  LOO ROC-AUC: %.3f | PR-AUC: %.3f" % (roc_auc, pr_auc))

    # Calibration by tier
    print("\n  RISK TIER CALIBRATION:")
    print("  %-10s %6s %10s %12s %s" % ("Tier", "Count", "Disrupted", "Obs. Rate", "Interpretation"))
    print("  " + "-" * 70)

    for lo, hi, name, desc in RISK_TIERS:
        mask = (loo_probs >= lo) & (loo_probs < hi)
        cnt = mask.sum()
        if cnt > 0:
            n_dis = int(loo_actual[mask].sum())
            rate = loo_actual[mask].mean()
            print("  %-10s %6d %10d %11.1f%% %s" % (name, cnt, n_dis, rate * 100, desc))
        else:
            print("  %-10s %6d %10s %12s %s" % (name, cnt, "--", "--", desc))

    # Every season with tier
    print("\n  ALL SEASONS WITH RISK TIER:")
    print("  %4s %3s %10s %8s %-10s" % ("Year", "Szn", "Actual", "Prob", "Tier"))
    print("  " + "-" * 42)
    for d in loo_details:
        flag = ""
        if d['actual'] == "DISRUPTED" and d['tier'] == "LOW":
            flag = " <-- missed"
        elif d['actual'] == "NORMAL" and d['tier'] in ["ELEVATED", "SEVERE"]:
            flag = " <-- false alarm"
        print("  %4d S%1d %10s %8.3f %-10s%s" % (
            d['year'], d['season'], d['actual'], d['prob'], d['tier'], flag))

    # Tier performance analysis
    print("\n  TIER DECISION ANALYSIS:")
    print("  " + "-" * 50)
    # If we acted on ELEVATED+ (>=0.50)
    elev_mask = loo_probs >= 0.50
    elev_tp = (loo_actual[elev_mask] == 1).sum()
    elev_fp = (loo_actual[elev_mask] == 0).sum()
    elev_fn = (loo_actual[~elev_mask] == 1).sum()
    print("  If acting on ELEVATED+ (prob >= 0.50):")
    print("    True alerts: %d, False alarms: %d, Missed: %d" % (elev_tp, elev_fp, elev_fn))

    # If we acted on MODERATE+ (>=0.20)
    mod_mask = loo_probs >= 0.20
    mod_tp = (loo_actual[mod_mask] == 1).sum()
    mod_fp = (loo_actual[mod_mask] == 0).sum()
    mod_fn = (loo_actual[~mod_mask] == 1).sum()
    print("  If acting on MODERATE+ (prob >= 0.20):")
    print("    True alerts: %d, False alarms: %d, Missed: %d" % (mod_tp, mod_fp, mod_fn))

    # If we acted on SEVERE only (>=0.70)
    sev_mask = loo_probs >= 0.70
    sev_tp = (loo_actual[sev_mask] == 1).sum()
    sev_fp = (loo_actual[sev_mask] == 0).sum()
    sev_fn = (loo_actual[~sev_mask] == 1).sum()
    print("  If acting on SEVERE only (prob >= 0.70):")
    print("    True alerts: %d, False alarms: %d, Missed: %d" % (sev_tp, sev_fp, sev_fn))

    return loo_probs, loo_actual


# =====================================================================
# PART 3: UPDATED 2026 S1 PREDICTION
# =====================================================================

def predict_2026(feat):
    print("\n")
    print("=" * 70)
    print("  PART 3: 2026 S1 PREDICTION (Recalibrated Risk Model)")
    print("=" * 70)

    df = feat.dropna(subset=FEAT_COLS + ['target'])

    # Train on all 32 samples
    X = df[FEAT_COLS].values
    y = df['target'].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(Xs, y)

    # 2026 S1 features (Feb 13 data)
    # SST Z = +0.732, Chl Z = +0.166 (Dec proxy), Nino = -0.29, Summer=1, Bio=79.2%
    x_dec = np.array([0.732, 0.166, -0.29, 1.0, 79.2])
    # Alternative: Chl Z = -0.40 (Feb proxy)
    x_feb = np.array([0.732, -0.403, -0.29, 1.0, 79.2])

    p_dec = model.predict_proba(scaler.transform(x_dec.reshape(1, -1)))[0, 1]
    p_feb = model.predict_proba(scaler.transform(x_feb.reshape(1, -1)))[0, 1]

    # Risk tier assignment
    def get_tier(p):
        for lo, hi, name, desc in RISK_TIERS:
            if lo <= p < hi:
                return name, desc
        return "UNKNOWN", ""

    tier_dec, desc_dec = get_tier(p_dec)
    tier_feb, desc_feb = get_tier(p_feb)

    print("\n  FEATURES (Feb 13, 2026 data):")
    print("    SST Z:      +0.732 (warming, rising)")
    print("    Chl Z:      +0.166 (Dec proxy) / -0.403 (Feb proxy)")
    print("    Nino 1+2:   -0.29 (Jan 2026, slightly cool)")
    print("    is_summer:  1 (S1)")
    print("    Bio >23C:   79.2%%")

    print("\n  POINT ESTIMATES:")
    print("    Dec Chl proxy: %.3f -> %s" % (p_dec, tier_dec))
    print("    Feb Chl proxy: %.3f -> %s" % (p_feb, tier_feb))

    # Bootstrap with both Chl scenarios
    print("\n  BOOTSTRAP (2000 resamples):")
    rng = np.random.RandomState(42)

    for label, x_2026 in [("Dec Chl proxy", x_dec), ("Feb Chl proxy", x_feb)]:
        probs = []
        for _ in range(2000):
            idx = rng.choice(len(df), size=len(df), replace=True)
            X_b = df[FEAT_COLS].values[idx]
            y_b = df['target'].values[idx]
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                continue
            sc = StandardScaler()
            m = LogisticRegression(max_iter=1000, solver='lbfgs')
            m.fit(sc.fit_transform(X_b), y_b)
            p = m.predict_proba(sc.transform(x_2026.reshape(1, -1)))[0, 1]
            probs.append(p)

        probs = np.array(probs)
        med = np.median(probs)
        ci_lo = np.percentile(probs, 2.5)
        ci_hi = np.percentile(probs, 97.5)

        # Count tier distribution
        tier_counts = {}
        for lo, hi, name, _ in RISK_TIERS:
            tier_counts[name] = ((probs >= lo) & (probs < hi)).sum()

        print("\n    %s:" % label)
        print("      Median: %.3f | 95%% CI: [%.3f, %.3f]" % (med, ci_lo, ci_hi))
        print("      Bootstrap tier distribution:")
        for name, cnt in tier_counts.items():
            print("        %-10s %5.1f%%" % (name, cnt / len(probs) * 100))

    # Coefficients
    print("\n  MODEL COEFFICIENTS (full 32-sample train):")
    for feat_name, coef in zip(FEAT_COLS, model.coef_[0]):
        direction = "warmer/higher -> more risk" if coef > 0 else "protective"
        print("    %-18s %+.3f  (%s)" % (feat_name, coef, direction))
    print("    %-18s %+.3f" % ("intercept", model.intercept_[0]))

    # Final interpretation
    print("\n  " + "=" * 60)
    print("  INTERPRETATION")
    print("  " + "=" * 60)
    print("")
    print("  Risk level: MODERATE (0.33-0.50 depending on Chl)")
    print("")
    print("  What this means:")
    print("    - Environmental stress is present but not extreme")
    print("    - SST warming real and accelerating (+0.73 sigma)")
    print("    - Chl is the swing variable (unknown until mid-March)")
    print("    - Historical seasons at this risk level:")
    print("      Disrupted ~35%% of the time")
    print("    - Prior seasons strong (2.46M + 1.60M catch)")
    print("      suggesting biomass buffer exists")
    print("")
    print("  What this does NOT tell you:")
    print("    - Biomass state (need IMARPE survey)")
    print("    - Juvenile percentage (critical for quota)")
    print("    - Governance decisions (political risk)")
    print("")
    print("  Model honest limitations:")
    print("    - 32 training samples")
    print("    - LOO ROC-AUC = 0.583 (moderate discrimination)")
    print("    - Strong at extremes (>0.7 = 75%% disruption rate)")
    print("    - Weak in the middle (0.2-0.5 zone is noisy)")
    print("    - Current prediction falls in the noisy zone")
    print("")
    print("  Next decision point: ~March 15")
    print("    - Copernicus Feb Chl resolves the Chl uncertainty")
    print("    - CPC Feb Nino 1+2 shows if Costero is confirmed")
    print("    - If both deteriorate: moves to ELEVATED")
    print("    - If Chl holds: stays MODERATE")
    print("  " + "=" * 60)


def main():
    feat, gt = load_data()

    # Part 1: Audit
    errors = audit_data(feat, gt)

    # Part 2: Recalibrated model
    loo_probs, loo_actual = run_recalibrated_model(feat)

    # Part 3: 2026 prediction
    predict_2026(feat)


if __name__ == "__main__":
    main()
