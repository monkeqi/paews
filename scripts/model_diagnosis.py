# PAEWS Model Diagnosis & Enhancement
#
# Three model variants compared via leave-one-out:
#   A) Baseline: original 5 features, binary target
#   B) +Lagged landings: adds catch(t-1) and catch(t-2) as biomass proxy
#   C) Severe target: redefines disruption as <1M MT or cancelled
#
# Usage: python scripts/model_diagnosis.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

BASE = Path(__file__).resolve().parent.parent
FEATURE_CSV = BASE / "data" / "external" / "paews_feature_matrix.csv"
GROUND_TRUTH = BASE / "data" / "external" / "imarpe_ground_truth.csv"

FEAT_BASE = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']
THRESHOLD = 0.38


def load_and_merge():
    feat = pd.read_csv(FEATURE_CSV)
    gt = pd.read_csv(GROUND_TRUTH)

    # Merge catch_mt from ground truth into feature matrix
    gt_slim = gt[['year', 'season', 'catch_mt', 'quota_mt']].copy()
    df = feat.merge(gt_slim, on=['year', 'season'], how='left')

    # Sort chronologically
    df = df.sort_values(['year', 'season']).reset_index(drop=True)

    # Create lagged landings (previous 1 and 2 seasons)
    df['catch_t1'] = df['catch_mt'].shift(1)   # previous season
    df['catch_t2'] = df['catch_mt'].shift(2)   # two seasons ago

    # Normalize catch to millions for cleaner coefficients
    df['catch_t1_M'] = df['catch_t1'] / 1e6
    df['catch_t2_M'] = df['catch_t2'] / 1e6

    # Create severe target: cancelled or <1M MT catch
    df['target_severe'] = 0
    df.loc[df['catch_mt'] < 1000000, 'target_severe'] = 1
    # Also flag NaN catch (cancelled seasons have 0 or NaN)
    df.loc[df['catch_mt'].isna(), 'target_severe'] = 1
    # Seasons with 0 catch
    df.loc[df['catch_mt'] == 0, 'target_severe'] = 1

    return df


def loo_evaluate(df, feat_cols, target_col, threshold, label):
    valid = df.dropna(subset=feat_cols + [target_col])
    n = len(valid)
    n_pos = int(valid[target_col].sum())

    probs = []
    actuals = []
    details = []

    for idx in valid.index:
        train = valid.drop(idx)
        test = valid.loc[[idx]]

        X_train = train[feat_cols].values
        y_train = train[target_col].values
        X_test = test[feat_cols].values

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_tr_s, y_train)

        prob = model.predict_proba(X_te_s)[0, 1]
        actual = int(test[target_col].iloc[0])

        probs.append(prob)
        actuals.append(actual)

        predicted = "AT RISK" if prob >= threshold else "NORMAL"
        actual_label = "DISRUPTED" if actual == 1 else "NORMAL"
        correct = (actual == 1 and prob >= threshold) or (actual == 0 and prob < threshold)

        details.append({
            'year': int(test['year'].iloc[0]),
            'season': int(test['season'].iloc[0]),
            'actual': actual_label,
            'prob': prob,
            'predicted': predicted,
            'correct': correct,
            'catch_mt': test['catch_mt'].iloc[0] if 'catch_mt' in test.columns else None,
        })

    probs = np.array(probs)
    actuals = np.array(actuals)
    preds = (probs >= threshold).astype(int)

    if len(np.unique(actuals)) < 2:
        return None, details

    roc = roc_auc_score(actuals, probs)
    pr = average_precision_score(actuals, probs)
    tn, fp, fn, tp = confusion_matrix(actuals, preds).ravel()
    acc = (tp + tn) / len(actuals)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'label': label,
        'n': n, 'n_pos': n_pos, 'n_features': len(feat_cols),
        'roc_auc': roc, 'pr_auc': pr,
        'accuracy': acc, 'recall': recall, 'precision': precision,
        'specificity': specificity,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }
    return metrics, details


def print_results(metrics, details, show_details=True):
    m = metrics
    print("")
    print("  %s" % m['label'])
    print("  %s" % ("-" * len(m['label'])))
    print("  Samples: %d (%d positive, %d features)" % (m['n'], m['n_pos'], m['n_features']))
    print("  ROC-AUC:     %.3f" % m['roc_auc'])
    print("  PR-AUC:      %.3f" % m['pr_auc'])
    print("  Accuracy:    %.1f%%" % (m['accuracy'] * 100))
    print("  Recall:      %.1f%% (%d/%d caught)" % (m['recall'] * 100, m['tp'], m['tp'] + m['fn']))
    print("  Precision:   %.1f%% (%d/%d real)" % (m['precision'] * 100, m['tp'], m['tp'] + m['fp']))
    print("  Specificity: %.1f%% (%d/%d cleared)" % (m['specificity'] * 100, m['tn'], m['tn'] + m['fp']))
    print("  FP: %d | FN: %d" % (m['fp'], m['fn']))

    if show_details:
        print("")
        print("  %4s %3s %10s %8s %12s %8s %10s" % (
            "Year", "Szn", "Actual", "Prob", "Predicted", "OK?", "Catch MT"))
        print("  %s" % ("-" * 62))
        for d in details:
            mark = "YES" if d['correct'] else "MISS"
            catch = "%10.0f" % d['catch_mt'] if d['catch_mt'] and not np.isnan(d['catch_mt']) else "       N/A"
            print("  %4d S%1d %10s %8.3f %12s %8s %s" % (
                d['year'], d['season'], d['actual'], d['prob'], d['predicted'], mark, catch))


def print_coefficients(df, feat_cols, target_col, label):
    valid = df.dropna(subset=feat_cols + [target_col])
    X = valid[feat_cols].values
    y = valid[target_col].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(Xs, y)

    print("")
    print("  Coefficients (%s):" % label)
    for feat, coef in zip(feat_cols, model.coef_[0]):
        direction = "-> risk" if coef > 0 else "-> safe"
        print("    %-18s %+.3f  %s" % (feat, coef, direction))
    print("    %-18s %+.3f" % ("intercept", model.intercept_[0]))


def main():
    df = load_and_merge()

    print("=" * 70)
    print("  PAEWS MODEL DIAGNOSIS")
    print("  Root cause analysis: missing biomass state + target definition")
    print("=" * 70)

    # Show lagged data
    print("")
    print("  LAGGED LANDINGS DATA:")
    print("  %4s %3s %10s %10s %10s %6s" % ("Year", "Szn", "Catch MT", "Lag-1 MT", "Lag-2 MT", "Target"))
    print("  %s" % ("-" * 52))
    for _, row in df.iterrows():
        catch = "%10.0f" % row['catch_mt'] if not np.isnan(row['catch_mt']) else "       N/A"
        lag1 = "%10.0f" % row['catch_t1'] if not np.isnan(row['catch_t1']) else "       N/A"
        lag2 = "%10.0f" % row['catch_t2'] if not np.isnan(row['catch_t2']) else "       N/A"
        print("  %4d S%1d %s %s %s %6d" % (
            int(row['year']), int(row['season']), catch, lag1, lag2, int(row['target'])))

    # === MODEL A: Baseline (original) ===
    print("")
    print("=" * 70)
    print("  MODEL A: BASELINE (original 5 features, binary target)")
    print("=" * 70)
    m_a, d_a = loo_evaluate(df, FEAT_BASE, 'target', THRESHOLD, "Baseline")
    if m_a:
        print_results(m_a, d_a, show_details=False)
        print_coefficients(df, FEAT_BASE, 'target', "Baseline")

    # === MODEL B: + Lagged landings ===
    FEAT_LAG = FEAT_BASE + ['catch_t1_M', 'catch_t2_M']
    print("")
    print("=" * 70)
    print("  MODEL B: + LAGGED LANDINGS (biomass proxy)")
    print("=" * 70)
    m_b, d_b = loo_evaluate(df, FEAT_LAG, 'target', THRESHOLD, "+Lagged Landings")
    if m_b:
        print_results(m_b, d_b, show_details=False)
        print_coefficients(df, FEAT_LAG, 'target', "+Lagged Landings")

    # === MODEL C: Severe target ===
    print("")
    print("=" * 70)
    print("  MODEL C: SEVERE TARGET (<1M MT or cancelled)")
    print("=" * 70)

    # Show severe vs original target
    print("")
    print("  TARGET COMPARISON:")
    print("  %4s %3s %10s %8s %8s" % ("Year", "Szn", "Catch MT", "Original", "Severe"))
    print("  %s" % ("-" * 42))
    for _, row in df.iterrows():
        catch = "%10.0f" % row['catch_mt'] if not np.isnan(row['catch_mt']) else "       N/A"
        orig = "DISRUPT" if row['target'] == 1 else "normal"
        sev = "SEVERE" if row['target_severe'] == 1 else "normal"
        if row['target'] != row['target_severe']:
            flag = " <-- CHANGED"
        else:
            flag = ""
        print("  %4d S%1d %s %8s %8s%s" % (
            int(row['year']), int(row['season']), catch, orig, sev, flag))

    m_c, d_c = loo_evaluate(df, FEAT_BASE, 'target_severe', THRESHOLD,
                            "Severe Target (base features)")
    if m_c:
        print_results(m_c, d_c, show_details=False)

    # === MODEL D: Severe + Lagged ===
    print("")
    print("=" * 70)
    print("  MODEL D: SEVERE TARGET + LAGGED LANDINGS (best of both)")
    print("=" * 70)
    m_d, d_d = loo_evaluate(df, FEAT_LAG, 'target_severe', THRESHOLD,
                            "Severe + Lagged Landings")
    if m_d:
        print_results(m_d, d_d, show_details=True)
        print_coefficients(df, FEAT_LAG, 'target_severe', "Severe + Lagged")

    # === COMPARISON TABLE ===
    print("")
    print("=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print("")
    print("  %-25s %8s %8s %8s %8s %5s %5s" % (
        "Model", "ROC-AUC", "PR-AUC", "Recall", "Precis", "FP", "FN"))
    print("  %s" % ("-" * 70))
    for m in [m_a, m_b, m_c, m_d]:
        if m:
            print("  %-25s %8.3f %8.3f %7.1f%% %7.1f%% %5d %5d" % (
                m['label'], m['roc_auc'], m['pr_auc'],
                m['recall'] * 100, m['precision'] * 100, m['fp'], m['fn']))

    # === 2026 IMPLICATION ===
    print("")
    print("=" * 70)
    print("  2026 S1 IMPLICATION")
    print("=" * 70)
    print("")
    print("  For 2026 S1 prediction, lagged landings would be:")
    print("    catch_t1 = 2025 S2: ~1,600,000 MT (1.60M)")
    print("    catch_t2 = 2025 S1: ~2,457,487 MT (2.46M)")
    print("  Both strong seasons -> high biomass buffer -> protective")
    print("")
    print("  If Model B/D shows improved AUC, these lag values would")
    print("  reduce 2026 disruption probability (strong prior seasons")
    print("  = more resilient stock = less vulnerable to warming).")
    print("=" * 70)


if __name__ == "__main__":
    main()
