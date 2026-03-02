# PAEWS Hindcast Validation
#
# For each season from 2023 S1 to 2025 S2, trains on ALL OTHER data
# and predicts that season out-of-sample. This is the honest test:
# could the model have predicted these seasons in real time?
#
# Also runs full leave-one-out cross-validation for proper metrics.
#
# Usage:
#     python scripts/hindcast_validation.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent.parent
FEATURE_CSV = BASE / "data" / "external" / "paews_feature_matrix.csv"
FEATURE_COLS = ['sst_z', 'chl_z', 'nino12_t1', 'is_summer', 'bio_thresh_pct']
THRESHOLD = 0.38


def load_data():
    df = pd.read_csv(FEATURE_CSV)
    df = df.dropna(subset=FEATURE_COLS + ['target'])
    return df


def train_predict_loo(df, test_idx):
    train = df.drop(test_idx)
    test = df.loc[[test_idx]]

    X_train = train[FEATURE_COLS].values
    y_train = train['target'].values
    X_test = test[FEATURE_COLS].values

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_s, y_train)

    prob = model.predict_proba(X_test_s)[0, 1]
    return prob


def main():
    df = load_data()
    n = len(df)
    n_pos = int(df['target'].sum())

    print("=" * 64)
    print("  PAEWS HINDCAST VALIDATION")
    print("  Dataset: %d samples (%d disrupted, %d normal)" % (n, n_pos, n - n_pos))
    print("  Features: %s" % FEATURE_COLS)
    print("  Threshold: %s" % THRESHOLD)
    print("=" * 64)

    # === PART 1: Recent Season Hindcasts ===
    print("")
    print("=" * 64)
    print("  PART 1: RECENT SEASON HINDCASTS (2023-2025)")
    print("  Could the model have predicted these in real time?")
    print("=" * 64)

    recent = df[df['year'] >= 2023].copy()
    print("")
    print("  %4s %3s %10s %8s %12s %8s" % ("Year", "Szn", "Actual", "Prob", "Predicted", "Correct"))
    print("  %s %s %s %s %s %s" % ("-"*4, "-"*3, "-"*10, "-"*8, "-"*12, "-"*8))

    recent_correct = 0
    for idx, row in recent.iterrows():
        prob = train_predict_loo(df, idx)
        if prob is None:
            continue
        actual = "DISRUPTED" if row['target'] == 1 else "NORMAL"
        predicted = "AT RISK" if prob >= THRESHOLD else "NORMAL"
        correct = (row['target'] == 1 and prob >= THRESHOLD) or \
                  (row['target'] == 0 and prob < THRESHOLD)
        mark = "YES" if correct else "NO !!!"
        if correct:
            recent_correct += 1
        print("  %4d S%1d %10s %8.3f %12s %8s" % (int(row['year']), int(row['season']), actual, prob, predicted, mark))

    print("")
    print("  Recent accuracy: %d/%d correct" % (recent_correct, len(recent)))

    # === PART 2: Full Leave-One-Out ===
    print("")
    print("=" * 64)
    print("  PART 2: FULL LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 64)

    loo_probs = []
    loo_actual = []
    loo_details = []

    for idx, row in df.iterrows():
        prob = train_predict_loo(df, idx)
        if prob is None:
            continue
        loo_probs.append(prob)
        loo_actual.append(int(row['target']))
        predicted = "AT RISK" if prob >= THRESHOLD else "NORMAL"
        actual = "DISRUPTED" if row['target'] == 1 else "NORMAL"
        correct = (row['target'] == 1 and prob >= THRESHOLD) or \
                  (row['target'] == 0 and prob < THRESHOLD)
        loo_details.append({
            'year': int(row['year']),
            'season': int(row['season']),
            'actual': actual,
            'prob': prob,
            'predicted': predicted,
            'correct': correct
        })

    loo_probs = np.array(loo_probs)
    loo_actual = np.array(loo_actual)
    loo_preds = (loo_probs >= THRESHOLD).astype(int)

    roc_auc = roc_auc_score(loo_actual, loo_probs)
    pr_auc = average_precision_score(loo_actual, loo_probs)
    tn, fp, fn, tp = confusion_matrix(loo_actual, loo_preds).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("")
    print("  ROC-AUC:       %.3f" % roc_auc)
    print("  PR-AUC:        %.3f" % pr_auc)
    print("  Accuracy:      %.1f%% (%d/%d)" % (accuracy * 100, tp + tn, n))
    print("  Recall:        %.1f%% (%d/%d disrupted seasons caught)" % (recall * 100, tp, tp + fn))
    print("  Precision:     %.1f%% (%d/%d alerts were real)" % (precision * 100, tp, tp + fp))
    print("  Specificity:   %.1f%% (%d/%d normal seasons correctly cleared)" % (specificity * 100, tn, tn + fp))
    print("  False Pos:     %d (normal seasons flagged as risk)" % fp)
    print("  False Neg:     %d (disrupted seasons missed)" % fn)

    print("")
    print("  Confusion Matrix:")
    print("  %12s Pred NORMAL  Pred AT RISK" % "")
    print("  %12s    %4d          %4d" % ("Actual NORM", tn, fp))
    print("  %12s    %4d          %4d" % ("Actual DISR", fn, tp))

    # === PART 3: Every Season Detail ===
    print("")
    print("  %4s %3s %10s %8s %12s %8s" % ("Year", "Szn", "Actual", "Prob", "Predicted", "Correct"))
    print("  %s %s %s %s %s %s" % ("-"*4, "-"*3, "-"*10, "-"*8, "-"*12, "-"*8))

    for d in loo_details:
        mark = "YES" if d['correct'] else "MISS"
        print("  %4d S%1d %10s %8.3f %12s %8s" % (d['year'], d['season'], d['actual'], d['prob'], d['predicted'], mark))

    # === PART 4: Misclassifications ===
    misses = [d for d in loo_details if not d['correct']]
    if misses:
        print("")
        print("  MISCLASSIFICATIONS (%d):" % len(misses))
        for d in misses:
            direction = "False Positive" if d['actual'] == 'NORMAL' else "False Negative"
            print("    %d S%d: %s (prob=%.3f, actual=%s)" % (d['year'], d['season'], direction, d['prob'], d['actual']))

    # === PART 5: Calibration ===
    print("")
    print("  CALIBRATION CHECK:")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("  %12s %6s %18s %14s" % ("Prob Range", "Count", "Actual Disrupted", "Observed Rate"))
    for lo, hi in bins:
        mask = (loo_probs >= lo) & (loo_probs < hi)
        cnt = mask.sum()
        if cnt > 0:
            obs_rate = loo_actual[mask].mean()
            n_dis = int(loo_actual[mask].sum())
            print("  %12s %6d %18d %13.1f%%" % ("[%.1f, %.1f)" % (lo, hi), cnt, n_dis, obs_rate * 100))
        else:
            print("  %12s %6d %18s %14s" % ("[%.1f, %.1f)" % (lo, hi), cnt, "--", "--"))

    # === SUMMARY ===
    print("")
    print("=" * 64)
    print("  SUMMARY")
    print("  LOO ROC-AUC: %.3f | PR-AUC: %.3f" % (roc_auc, pr_auc))
    print("  Recall: %d/%d disrupted caught | FP: %d false alarms" % (tp, tp + fn, fp))
    print("  Recent (2023-2025): %d/%d correct" % (recent_correct, len(recent)))
    print("=" * 64)


if __name__ == "__main__":
    main()
