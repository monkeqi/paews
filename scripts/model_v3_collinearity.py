# PAEWS Model v3: Fix Multicollinearity
#
# Drops is_summer (r=0.963 with bio_thresh_pct)
# Tests 4-feature model vs original 5-feature
# Also tests dropping bio_thresh_pct instead
#
# Usage: python scripts/model_v3_collinearity.py

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

RISK_TIERS = [
    (0.0, 0.20, "LOW"),
    (0.20, 0.50, "MODERATE"),
    (0.50, 0.70, "ELEVATED"),
    (0.70, 1.01, "SEVERE"),
]

# Feature sets to compare
MODELS = {
    "A: Original 5":        ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct'],
    "B: Drop is_summer":    ['sst_z', 'chl_z', 'nino12_t1', 'bio_thresh_pct'],
    "C: Drop bio_thresh":   ['sst_z', 'chl_z', 'nino12_t1', 'is_summer'],
    "D: Drop both":         ['sst_z', 'chl_z', 'nino12_t1'],
}

X_2026_DEC = {
    "A: Original 5":        [0.732, 0.166, -0.29, 1.0, 79.2],
    "B: Drop is_summer":    [0.732, 0.166, -0.29, 79.2],
    "C: Drop bio_thresh":   [0.732, 0.166, -0.29, 1.0],
    "D: Drop both":         [0.732, 0.166, -0.29],
}

X_2026_FEB = {
    "A: Original 5":        [0.732, -0.403, -0.29, 1.0, 79.2],
    "B: Drop is_summer":    [0.732, -0.403, -0.29, 79.2],
    "C: Drop bio_thresh":   [0.732, -0.403, -0.29, 1.0],
    "D: Drop both":         [0.732, -0.403, -0.29],
}


def loo_evaluate(df, feat_cols):
    valid = df.dropna(subset=feat_cols + ['target'])
    probs = []
    actuals = []
    details = []

    for idx in valid.index:
        train = valid.drop(idx)
        test = valid.loc[[idx]]
        X_train = train[feat_cols].values
        y_train = train['target'].values
        X_test = test[feat_cols].values

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue

        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(scaler.fit_transform(X_train), y_train)
        prob = model.predict_proba(scaler.transform(X_test))[0, 1]
        actual = int(test['target'].iloc[0])

        probs.append(prob)
        actuals.append(actual)

        tier = "?"
        for lo, hi, name in RISK_TIERS:
            if lo <= prob < hi:
                tier = name
                break

        details.append({
            'year': int(test['year'].iloc[0]),
            'season': int(test['season'].iloc[0]),
            'actual': int(test['target'].iloc[0]),
            'prob': prob,
            'tier': tier,
        })

    return np.array(probs), np.array(actuals), details


def tier_calibration(probs, actuals):
    results = {}
    for lo, hi, name in RISK_TIERS:
        mask = (probs >= lo) & (probs < hi)
        cnt = mask.sum()
        if cnt > 0:
            n_dis = int(actuals[mask].sum())
            rate = actuals[mask].mean()
            results[name] = (cnt, n_dis, rate)
        else:
            results[name] = (0, 0, 0.0)
    return results


def main():
    df = pd.read_csv(FEATURE_CSV)

    print("=" * 70)
    print("  PAEWS MODEL v3: MULTICOLLINEARITY FIX")
    print("  is_summer vs bio_thresh_pct correlation = 0.963")
    print("  Testing which to drop")
    print("=" * 70)

    # Confirm the correlation
    corr = df[['is_summer', 'bio_thresh_pct']].corr().iloc[0, 1]
    print("\n  Confirmed correlation: %.3f" % corr)

    # Show why they're correlated
    print("\n  WHY THEY CORRELATE:")
    print("  %4s %3s %10s %10s" % ("Year", "Szn", "is_summer", "bio>23%"))
    print("  " + "-" * 32)
    for _, row in df.iterrows():
        print("  %4d S%1d %10d %9.1f%%" % (
            int(row['year']), int(row['season']),
            int(row['is_summer']), row['bio_thresh_pct']))

    # S1 vs S2 averages
    s1_bio = df[df.is_summer == 1]['bio_thresh_pct'].mean()
    s2_bio = df[df.is_summer == 0]['bio_thresh_pct'].mean()
    s1_std = df[df.is_summer == 1]['bio_thresh_pct'].std()
    s2_std = df[df.is_summer == 0]['bio_thresh_pct'].std()
    print("\n  S1 mean bio: %.1f%% (std %.1f%%)" % (s1_bio, s1_std))
    print("  S2 mean bio: %.1f%% (std %.1f%%)" % (s2_bio, s2_std))
    print("  S1 has %.1fx more within-season variance" % (s1_std / s2_std if s2_std > 0 else 0))

    # =====================================================
    # Compare all models
    # =====================================================
    print("\n")
    print("=" * 70)
    print("  MODEL COMPARISON (LOO)")
    print("=" * 70)

    all_metrics = []

    for label, feat_cols in MODELS.items():
        probs, actuals, details = loo_evaluate(df, feat_cols)
        if len(np.unique(actuals)) < 2:
            continue

        roc = roc_auc_score(actuals, probs)
        pr = average_precision_score(actuals, probs)
        cal = tier_calibration(probs, actuals)

        # Separation score: how different are tier disruption rates?
        rates = [cal[name][2] for name in ['LOW', 'MODERATE', 'ELEVATED', 'SEVERE'] if cal[name][0] > 0]
        separation = max(rates) - min(rates) if len(rates) > 1 else 0

        # False alarms at ELEVATED+
        elev_mask = probs >= 0.50
        elev_fp = int((actuals[elev_mask] == 0).sum())
        elev_tp = int((actuals[elev_mask] == 1).sum())

        # Missed at SEVERE
        sev_mask = probs >= 0.70
        sev_tp = int((actuals[sev_mask] == 1).sum())
        sev_fp = int((actuals[sev_mask] == 0).sum())

        metrics = {
            'label': label,
            'n_feat': len(feat_cols),
            'roc_auc': roc,
            'pr_auc': pr,
            'separation': separation,
            'cal': cal,
            'elev_tp': elev_tp,
            'elev_fp': elev_fp,
            'sev_tp': sev_tp,
            'sev_fp': sev_fp,
            'details': details,
            'probs': probs,
            'actuals': actuals,
        }
        all_metrics.append(metrics)

        print("\n  %s (%d features)" % (label, len(feat_cols)))
        print("  " + "-" * 50)
        print("  ROC-AUC: %.3f | PR-AUC: %.3f | Tier separation: %.3f" % (roc, pr, separation))
        print("  ELEVATED+: %d true / %d false | SEVERE: %d true / %d false" % (
            elev_tp, elev_fp, sev_tp, sev_fp))
        print("  Tier calibration:")
        for name in ['LOW', 'MODERATE', 'ELEVATED', 'SEVERE']:
            cnt, n_dis, rate = cal[name]
            if cnt > 0:
                print("    %-10s n=%2d  disrupted=%d  rate=%.0f%%" % (name, cnt, n_dis, rate * 100))
            else:
                print("    %-10s n=%2d  --" % (name, cnt))

    # =====================================================
    # Summary table
    # =====================================================
    print("\n")
    print("=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print("\n  %-22s %5s %7s %7s %7s %6s %6s" % (
        "Model", "Feat", "ROC", "PR", "Sep", "SevTP", "SevFP"))
    print("  " + "-" * 64)
    for m in all_metrics:
        print("  %-22s %5d %7.3f %7.3f %7.3f %6d %6d" % (
            m['label'], m['n_feat'], m['roc_auc'], m['pr_auc'],
            m['separation'], m['sev_tp'], m['sev_fp']))

    # =====================================================
    # Best model detail
    # =====================================================
    # Pick model with best tier separation (what we care about)
    best = max(all_metrics, key=lambda m: m['separation'])
    print("\n  BEST TIER SEPARATION: %s (%.3f)" % (best['label'], best['separation']))

    # Pick model with best ROC
    best_roc = max(all_metrics, key=lambda m: m['roc_auc'])
    print("  BEST ROC-AUC: %s (%.3f)" % (best_roc['label'], best_roc['roc_auc']))

    # =====================================================
    # Season detail for best model
    # =====================================================
    print("\n")
    print("=" * 70)
    print("  BEST MODEL SEASON DETAIL: %s" % best['label'])
    print("=" * 70)
    print("\n  %4s %3s %10s %8s %-10s" % ("Year", "Szn", "Actual", "Prob", "Tier"))
    print("  " + "-" * 42)
    for d in best['details']:
        actual = "DISRUPTED" if d['actual'] == 1 else "NORMAL"
        flag = ""
        if d['actual'] == 1 and d['tier'] == "LOW":
            flag = " <-- missed"
        elif d['actual'] == 0 and d['tier'] in ["ELEVATED", "SEVERE"]:
            flag = " <-- false alarm"
        print("  %4d S%1d %10s %8.3f %-10s%s" % (
            d['year'], d['season'], actual, d['prob'], d['tier'], flag))

    # =====================================================
    # Coefficients for best model
    # =====================================================
    best_feats = MODELS[best['label']]
    valid = df.dropna(subset=best_feats + ['target'])
    X = valid[best_feats].values
    y = valid['target'].values
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(scaler.fit_transform(X), y)

    print("\n  COEFFICIENTS:")
    for feat, coef in zip(best_feats, model.coef_[0]):
        print("    %-18s %+.3f" % (feat, coef))
    print("    %-18s %+.3f" % ("intercept", model.intercept_[0]))

    # Correlation check
    corr = valid[best_feats].corr()
    print("\n  REMAINING CORRELATIONS:")
    for i in range(len(best_feats)):
        for j in range(i + 1, len(best_feats)):
            r = corr.iloc[i, j]
            if abs(r) > 0.3:
                flag = " *** HIGH" if abs(r) > 0.7 else ""
                print("    %s vs %s: %.3f%s" % (best_feats[i], best_feats[j], r, flag))

    # =====================================================
    # 2026 S1 with best model
    # =====================================================
    print("\n")
    print("=" * 70)
    print("  2026 S1 PREDICTION (BEST MODEL: %s)" % best['label'])
    print("=" * 70)

    x_dec = np.array(X_2026_DEC[best['label']])
    x_feb = np.array(X_2026_FEB[best['label']])

    p_dec = model.predict_proba(scaler.transform(x_dec.reshape(1, -1)))[0, 1]
    p_feb = model.predict_proba(scaler.transform(x_feb.reshape(1, -1)))[0, 1]

    def get_tier(p):
        for lo, hi, name in RISK_TIERS:
            if lo <= p < hi:
                return name
        return "?"

    print("\n  Dec Chl proxy: %.3f -> %s" % (p_dec, get_tier(p_dec)))
    print("  Feb Chl proxy: %.3f -> %s" % (p_feb, get_tier(p_feb)))

    # Bootstrap
    rng = np.random.RandomState(42)
    for label_chl, x_2026 in [("Dec Chl", x_dec), ("Feb Chl", x_feb)]:
        boot_probs = []
        for _ in range(2000):
            idx = rng.choice(len(valid), size=len(valid), replace=True)
            X_b = valid[best_feats].values[idx]
            y_b = valid['target'].values[idx]
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                continue
            sc = StandardScaler()
            m = LogisticRegression(max_iter=1000, solver='lbfgs')
            m.fit(sc.fit_transform(X_b), y_b)
            p = m.predict_proba(sc.transform(x_2026.reshape(1, -1)))[0, 1]
            boot_probs.append(p)

        boot_probs = np.array(boot_probs)
        med = np.median(boot_probs)
        ci_lo = np.percentile(boot_probs, 2.5)
        ci_hi = np.percentile(boot_probs, 97.5)

        tier_pcts = {}
        for lo, hi, name in RISK_TIERS:
            tier_pcts[name] = ((boot_probs >= lo) & (boot_probs < hi)).mean() * 100

        print("\n  %s bootstrap:" % label_chl)
        print("    Median: %.3f | 95%% CI: [%.3f, %.3f]" % (med, ci_lo, ci_hi))
        for name, pct in tier_pcts.items():
            bar = "#" * int(pct / 2)
            print("    %-10s %5.1f%% %s" % (name, pct, bar))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
