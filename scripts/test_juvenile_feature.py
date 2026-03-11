"""
test_juvenile_feature.py — LOO evaluation of juvenile % as 4th feature

Run AFTER build_juvenile_feature.py has populated juv_pct in feature matrix.
Same framework as test_prev_feature_variants.py and test_godas_feature.py.

Usage:
    cd C:\\Users\\josep\\Documents\\paews
    python scripts/test_juvenile_feature.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

fm = pd.read_csv("data/external/paews_feature_matrix.csv")
gt = pd.read_csv("data/external/imarpe_ground_truth.csv")

feat_base = ["sst_z", "chl_z", "nino12_t1"]
feat_juv = ["sst_z", "chl_z", "nino12_t1", "juv_pct"]

if 'juv_pct' not in fm.columns:
    print("ERROR: juv_pct not in feature matrix. Run build_juvenile_feature.py first.")
    raise SystemExit(1)

valid = fm['juv_pct'].notna().sum()
missing = fm['juv_pct'].isna().sum()
print(f"Juvenile % coverage: {valid}/{len(fm)} ({missing} missing)")

if valid < 20:
    print(f"\nOnly {valid} seasons have data. Need at least 20 for meaningful LOO.")
    print("Go collect more data from PRODUCE resolutions / IMARPE reports.")
    raise SystemExit(1)

fm_test = fm.dropna(subset=feat_juv + ['target']).copy()
n = len(fm_test)
print(f"Usable samples: {n}")


def loo_evaluate(X, y, return_probs=False):
    n = len(y)
    probs = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]
        scaler = StandardScaler()
        m = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
        m.fit(scaler.fit_transform(X_train), y_train)
        probs[i] = m.predict_proba(scaler.transform(X_test))[0, 1]

    roc = roc_auc_score(y, probs)
    pr = average_precision_score(y, probs)

    severe_total = severe_correct = false_alarms = 0
    for p, actual in zip(probs, y):
        if p >= 0.70:
            severe_total += 1
            if actual == 1: severe_correct += 1
            else: false_alarms += 1
        elif p >= 0.50:
            if actual == 0: false_alarms += 1

    severe_str = f"{severe_correct}/{severe_total}" if severe_total > 0 else "0/0"
    severe_pct = severe_correct / severe_total * 100 if severe_total > 0 else 0

    if return_probs:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
        model.fit(X_s, y)
        return roc, pr, probs, severe_str, severe_pct, false_alarms, model.coef_[0], model.intercept_[0]
    return roc, pr, severe_str, severe_pct, false_alarms


X_base = fm_test[feat_base].values
X_juv = fm_test[feat_juv].values
y = fm_test['target'].values

print()
print("=" * 70)
print("  JUVENILE % FEATURE EVALUATION")
print("  Non-negotiable: SEVERE must stay at 100%")
print("=" * 70)

roc_b, pr_b, probs_b, sev_b, sev_pct_b, fa_b, coefs_b, int_b = loo_evaluate(X_base, y, return_probs=True)
roc_j, pr_j, probs_j, sev_j, sev_pct_j, fa_j, coefs_j, int_j = loo_evaluate(X_juv, y, return_probs=True)

delta_roc = roc_j - roc_b

print(f"\n  {'Metric':<25} {'3-feature':>12} {'4-feat +juv':>12} {'Delta':>10}")
print(f"  {'-'*62}")
print(f"  {'ROC-AUC':<25} {roc_b:>12.3f} {roc_j:>12.3f} {delta_roc:>+10.3f}")
print(f"  {'PR-AUC':<25} {pr_b:>12.3f} {pr_j:>12.3f} {pr_j-pr_b:>+10.3f}")
print(f"  {'SEVERE accuracy':<25} {sev_b:>12} {sev_j:>12}")
print(f"  {'False alarms (ELEV+)':<25} {fa_b:>12} {fa_j:>12}")
print(f"  {'Samples':<25} {n:>12} {n:>12}")

print(f"\n  Coefficients (4-feature):")
for i, feat in enumerate(feat_juv):
    print(f"    {feat:<15} {coefs_j[i]:+.4f}")
print(f"    {'intercept':<15} {int_j:+.4f}")

# Expected: juv_pct coefficient should be POSITIVE
# (more juveniles → more disruption risk)
juv_coef = coefs_j[feat_juv.index('juv_pct')]
if juv_coef > 0:
    print(f"\n  ✓ juv_pct coefficient is positive ({juv_coef:+.3f})")
    print(f"    More juveniles → higher disruption risk (biologically correct)")
else:
    print(f"\n  ⚠ juv_pct coefficient is NEGATIVE ({juv_coef:+.3f})")
    print(f"    Review the data — this is biologically unexpected")

# Collinearity
print(f"\n  Collinearity:")
for feat in feat_base:
    r = fm_test[feat].corr(fm_test['juv_pct'])
    flag = " ⚠ HIGH" if abs(r) > 0.8 else (" ~ moderate" if abs(r) > 0.5 else "")
    print(f"    vs {feat:<15} r = {r:+.3f}{flag}")
r_target = fm_test['juv_pct'].corr(fm_test['target'])
print(f"    vs target          r = {r_target:+.3f}")

# Per-season detail
print(f"\n{'='*70}")
print(f"  PER-SEASON COMPARISON")
print(f"{'='*70}")
print(f"\n  {'Season':<10} {'Actual':<10} {'3-feat':>8} {'4-feat':>8} {'Δ':>8} "
      f"{'juv%':>6} {'Dir':>8}")
print(f"  {'-'*62}")

improved = degraded = unchanged = 0
for i, (_, row) in enumerate(fm_test.iterrows()):
    yr = int(row['year'])
    sn = int(row['season'])
    actual = int(row['target'])
    juv = row['juv_pct']
    d = probs_j[i] - probs_b[i]

    if abs(d) < 0.01: unchanged += 1
    elif (d > 0 and actual == 1) or (d < 0 and actual == 0):
        improved += 1
    else:
        degraded += 1

    highlight = ""
    if yr == 2022 and sn == 2: highlight = " ← juvenile disruption (TARGET)"
    elif yr == 2023 and sn == 1: highlight = " ← 86% juveniles (TARGET)"
    elif yr == 2014 and sn == 1: highlight = " ← ghost miss"
    elif yr == 2011 and sn == 2: highlight = " ← stock depletion"

    if abs(d) > 0.02 or highlight:
        outcome = gt[(gt['year']==yr)&(gt['season']==sn)]['outcome'].values[0]
        direction = "✓" if (d > 0 and actual == 1) or (d < 0 and actual == 0) else "✗"
        if abs(d) < 0.01: direction = "-"
        print(f"  {yr} S{sn:<6} {outcome:<10} {probs_b[i]:>8.3f} {probs_j[i]:>8.3f} "
              f"{d:>+8.3f} {juv:>5.0f}% {direction:>8}{highlight}")

print(f"\n  Correct/Wrong/Unchanged: {improved}/{degraded}/{unchanged}")

# Verdict
print(f"\n{'='*70}")
print(f"  VERDICT")
print(f"{'='*70}")

passes = True
reasons = []

if sev_pct_j < 100:
    passes = False
    reasons.append(f"SEVERE dropped to {sev_j}")
if delta_roc < 0:
    passes = False
    reasons.append(f"ROC-AUC decreased ({delta_roc:+.3f})")

if passes and delta_roc > 0.01:
    print(f"\n  ✓✓ PASSES — juvenile % improves the model")
    print(f"     ROC-AUC: {roc_b:.3f} → {roc_j:.3f} ({delta_roc:+.3f})")
    print(f"     SEVERE: {sev_j}")
elif passes:
    print(f"\n  ~ MARGINAL — passes constraints, minimal improvement")
    print(f"     ROC-AUC: {roc_b:.3f} → {roc_j:.3f} ({delta_roc:+.3f})")
else:
    print(f"\n  ✗ FAILS")
    for r in reasons:
        print(f"     - {r}")
print(f"{'='*70}")
