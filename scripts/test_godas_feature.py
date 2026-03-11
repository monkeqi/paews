"""
test_godas_feature.py — LOO evaluation of GODAS Z20 as 4th feature

Tests whether adding thermocline depth (z20_z) improves the model:
  1. Must improve ROC-AUC over 0.629 baseline
  2. SEVERE tier must stay at 100% (non-negotiable)
  3. 2014 S1 should improve (the target case for GODAS)

Run AFTER godas_thermocline.py has populated z20_z in the feature matrix.

Usage:
    cd C:\\Users\\josep\\Documents\\paews
    conda activate paews
    python scripts/test_godas_feature.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Load data ──
gt = pd.read_csv("data/external/imarpe_ground_truth.csv")
fm = pd.read_csv("data/external/paews_feature_matrix.csv")

# ── Check z20_z exists ──
if 'z20_z' not in fm.columns:
    print("ERROR: z20_z column not found in feature matrix.")
    print("Run godas_thermocline.py first.")
    raise SystemExit(1)

# ── Report coverage ──
z20_valid = fm['z20_z'].notna().sum()
z20_missing = fm['z20_z'].isna().sum()
print(f"Z20 coverage: {z20_valid}/{len(fm)} seasons ({z20_missing} missing)")

if z20_missing > 0:
    missing_rows = fm[fm['z20_z'].isna()][['year', 'season']]
    print("Missing seasons:")
    for _, row in missing_rows.iterrows():
        print(f"  {int(row['year'])} S{int(row['season'])}")

# ══════════════════════════════════════════════════════════════
# LOO EVALUATION (same framework as test_prev_feature_variants.py)
# ══════════════════════════════════════════════════════════════

feat_base = ["sst_z", "chl_z", "nino12_t1"]
feat_godas = ["sst_z", "chl_z", "nino12_t1", "z20_z"]

# Drop rows with missing z20_z for fair comparison
fm_test = fm.dropna(subset=feat_godas + ['target']).copy()
n_test = len(fm_test)
print(f"\nUsable samples (no NaN in any feature): {n_test}")


def loo_evaluate(X, y, return_probs=False):
    """Leave-One-Out evaluation with balanced logistic regression."""
    n = len(y)
    probs = np.zeros(n)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs"
        )
        model.fit(X_train_s, y_train)
        probs[i] = model.predict_proba(X_test_s)[0, 1]

    roc = roc_auc_score(y, probs)
    pr = average_precision_score(y, probs)

    # Tier stats
    severe_total = 0
    severe_correct = 0
    false_alarms = 0  # NORMAL seasons scored ELEVATED+

    for p, actual in zip(probs, y):
        if p >= 0.70:
            severe_total += 1
            if actual == 1:
                severe_correct += 1
            else:
                false_alarms += 1
        elif p >= 0.50:
            if actual == 0:
                false_alarms += 1

    severe_str = f"{severe_correct}/{severe_total}" if severe_total > 0 else "0/0"
    severe_pct = severe_correct / severe_total * 100 if severe_total > 0 else 0

    if return_probs:
        # Also get full-model coefficients
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
        model.fit(X_s, y)
        return roc, pr, probs, severe_str, severe_pct, false_alarms, model.coef_[0], model.intercept_[0]

    return roc, pr, severe_str, severe_pct, false_alarms


# ══════════════════════════════════════════════════════════════
# RUN COMPARISON
# ══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  GODAS Z20 FEATURE EVALUATION")
print("  Non-negotiable constraint: SEVERE tier must stay at 100%")
print("=" * 70)

X_base = fm_test[feat_base].values
X_godas = fm_test[feat_godas].values
y = fm_test['target'].values

# Baseline (3-feature) on this subset
roc_b, pr_b, probs_b, sev_b, sev_pct_b, fa_b, coefs_b, int_b = loo_evaluate(X_base, y, return_probs=True)

# GODAS (4-feature)
roc_g, pr_g, probs_g, sev_g, sev_pct_g, fa_g, coefs_g, int_g = loo_evaluate(X_godas, y, return_probs=True)

delta_roc = roc_g - roc_b
delta_pr = pr_g - pr_b

print(f"\n  {'Metric':<25} {'3-feature':>12} {'4-feat +Z20':>12} {'Delta':>10}")
print(f"  {'-'*62}")
print(f"  {'ROC-AUC':<25} {roc_b:>12.3f} {roc_g:>12.3f} {delta_roc:>+10.3f}")
print(f"  {'PR-AUC':<25} {pr_b:>12.3f} {pr_g:>12.3f} {delta_pr:>+10.3f}")
print(f"  {'SEVERE accuracy':<25} {sev_b:>12} {sev_g:>12}")
print(f"  {'False alarms (ELEV+)':<25} {fa_b:>12} {fa_g:>12}")
print(f"  {'Samples':<25} {n_test:>12} {n_test:>12}")

# Coefficients
print(f"\n  Full-model coefficients (4-feature):")
for i, feat in enumerate(feat_godas):
    print(f"    {feat:<15} {coefs_g[i]:+.4f}")
print(f"    {'intercept':<15} {int_g:+.4f}")

# Expected: z20_z coefficient should be POSITIVE
# (deeper thermocline = more disruption risk)
z20_coef = coefs_g[feat_godas.index('z20_z')]
if z20_coef > 0:
    print(f"\n  ✓ z20_z coefficient is positive ({z20_coef:+.3f})")
    print(f"    Deeper thermocline → higher disruption risk (physically correct)")
else:
    print(f"\n  ⚠ z20_z coefficient is NEGATIVE ({z20_coef:+.3f})")
    print(f"    This is physically unexpected. Review the Z20 computation.")

# Collinearity check
print(f"\n  Collinearity with z20_z:")
for feat in feat_base:
    r = fm_test[feat].corr(fm_test['z20_z'])
    flag = " ⚠ DANGER" if abs(r) > 0.8 else (" ~ moderate" if abs(r) > 0.5 else "")
    print(f"    vs {feat:<15} r = {r:+.3f}{flag}")

# ══════════════════════════════════════════════════════════════
# PER-SEASON DETAIL
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PER-SEASON COMPARISON")
print(f"{'='*70}")

print(f"\n  {'Season':<10} {'Actual':<10} {'3-feat':>8} {'4-feat':>8} {'Δ':>8} "
      f"{'z20_z':>7} {'Direction':>10}")
print(f"  {'-'*68}")

improved = 0
degraded = 0
unchanged = 0

for i, (_, row) in enumerate(fm_test.iterrows()):
    yr = int(row['year'])
    sn = int(row['season'])
    actual = int(row['target'])
    z20_val = row['z20_z']
    d = probs_g[i] - probs_b[i]

    # Determine if change is good or bad
    if abs(d) < 0.01:
        direction = "-"
        unchanged += 1
    elif (d > 0 and actual == 1) or (d < 0 and actual == 0):
        direction = "✓ better"
        improved += 1
    else:
        direction = "✗ worse"
        degraded += 1

    # Highlight key seasons
    highlight = ""
    if yr == 2014 and sn == 1:
        highlight = " ← GHOST MISS (target)"
    elif yr == 2022 and sn == 2:
        highlight = " ← juvenile disruption"
    elif yr == 2011 and sn == 2:
        highlight = " ← stock depletion"

    # Only print if meaningful change OR key season
    if abs(d) > 0.02 or highlight:
        outcome = gt[(gt['year'] == yr) & (gt['season'] == sn)]['outcome'].values[0]
        print(f"  {yr} S{sn:<6} {outcome:<10} {probs_b[i]:>8.3f} {probs_g[i]:>8.3f} "
              f"{d:>+8.3f} {z20_val:>+7.3f} {direction:>10}{highlight}")

print(f"\n  Summary of changes (|Δ| > 0.01):")
print(f"    Directionally correct: {improved}")
print(f"    Directionally wrong:   {degraded}")
print(f"    Unchanged:             {unchanged}")

# ══════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  VERDICT")
print(f"{'='*70}")

passes = True
reasons = []

if sev_pct_g < 100:
    passes = False
    reasons.append(f"SEVERE dropped to {sev_g} ({sev_pct_g:.0f}%)")

if delta_roc < 0:
    passes = False
    reasons.append(f"ROC-AUC decreased ({delta_roc:+.3f})")

# Check 2014 S1 specifically
entry_2014 = fm_test[(fm_test['year'] == 2014) & (fm_test['season'] == 1)]
if len(entry_2014) > 0:
    idx = entry_2014.index[0]
    pos = list(fm_test.index).index(idx)
    delta_2014 = probs_g[pos] - probs_b[pos]
    if delta_2014 > 0.02:
        print(f"\n  ✓ 2014 S1 improved: {probs_b[pos]:.3f} → {probs_g[pos]:.3f} ({delta_2014:+.3f})")
    elif delta_2014 > -0.01:
        print(f"\n  ~ 2014 S1 unchanged: {probs_b[pos]:.3f} → {probs_g[pos]:.3f}")
    else:
        print(f"\n  ✗ 2014 S1 got WORSE: {probs_b[pos]:.3f} → {probs_g[pos]:.3f} ({delta_2014:+.3f})")
        reasons.append("2014 S1 (target case) got worse")

if passes and delta_roc > 0.005:
    print(f"\n  ✓✓ PASSES — GODAS Z20 improves the model")
    print(f"     ROC-AUC: {roc_b:.3f} → {roc_g:.3f} ({delta_roc:+.3f})")
    print(f"     SEVERE:  {sev_g} (100%)")
    print(f"     Ready for production as v3 model (4 features)")
    print(f"\n  Next steps:")
    print(f"    1. Update predict_2026_s1.py to include z20_z")
    print(f"    2. Rerun scenario_analysis.py with 4 features")
    print(f"    3. Update dashboard")
elif passes:
    print(f"\n  ~ MARGINAL — GODAS Z20 passes constraints but minimal improvement")
    print(f"     ROC-AUC: {roc_b:.3f} → {roc_g:.3f} ({delta_roc:+.3f})")
    print(f"     SEVERE:  {sev_g}")
    print(f"     Consider keeping 3-feature model for simplicity")
else:
    print(f"\n  ✗ FAILS — GODAS Z20 should NOT be added to production")
    for r in reasons:
        print(f"     - {r}")
    print(f"\n  Keep the 3-feature model.")
    print(f"  GODAS still has portfolio value as a technical demonstration")
    print(f"  even if it doesn't improve the LOO score.")

print(f"\n{'='*70}")
