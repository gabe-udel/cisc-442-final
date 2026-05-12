# evaluate.py - scores the saved model on the two holdout dogs

import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset import RESULTS_DIR
from features import FEATURES_CSV
from train import HOLDOUT_PATH, MODEL_PATH


TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.json"
TEST_PREDS_PATH = RESULTS_DIR / "test_predictions.csv"


def main():
    p = argparse.ArgumentParser(description="Evaluate the trained classifier on holdout.")
    p.parse_args()

    for required in (FEATURES_CSV, MODEL_PATH, HOLDOUT_PATH):
        if not required.exists():
            raise SystemExit(f"Missing {required}. Run features.py and train.py first.")

    df = pd.read_csv(FEATURES_CSV)
    artifact = joblib.load(MODEL_PATH)
    pipe = artifact["pipeline"]
    feat_cols = artifact["feature_columns"]
    model_name = artifact.get("model_name", "?")
    holdout = json.loads(HOLDOUT_PATH.read_text())

    # pull out only the holdout dogs from the full feature table
    test_df = df[df["dog_id"].isin([holdout["ccl"], holdout["normal"]])].reset_index(drop=True)
    if test_df.empty:
        raise SystemExit(f"No clips in features.csv match holdout dogs {holdout}.")

    X = test_df[feat_cols]
    y = test_df["label"].astype(int).to_numpy()
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"Model:  {model_name}")
    print(f"Holdout dogs:  CCL = {holdout['ccl']!r}    Normal = {holdout['normal']!r}")
    print(f"Test set: {len(test_df)} clips ({(y == 1).sum()} CCL, {(y == 0).sum()} Normal)")
    print()

    # clip-level metrics
    metrics = {"model_name": model_name, "holdout": holdout, "clip": {}, "dog": {}}
    metrics["clip"]["accuracy"] = float(accuracy_score(y, pred))
    try:
        metrics["clip"]["auc"] = float(roc_auc_score(y, proba))
    except ValueError:
        metrics["clip"]["auc"] = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    metrics["clip"]["precision_normal"] = float(prec[0])
    metrics["clip"]["precision_ccl"] = float(prec[1])
    metrics["clip"]["recall_normal"] = float(rec[0])
    metrics["clip"]["recall_ccl"] = float(rec[1])
    metrics["clip"]["f1_normal"] = float(f1[0])
    metrics["clip"]["f1_ccl"] = float(f1[1])

    cm = confusion_matrix(y, pred, labels=[0, 1])
    metrics["clip"]["confusion_matrix"] = cm.tolist()

    print("=== Clip-level ===")
    print(f"  Accuracy         : {metrics['clip']['accuracy']:.3f}")
    print(f"  AUC              : {metrics['clip']['auc']:.3f}")
    print(f"  Recall  (Normal) : {metrics['clip']['recall_normal']:.3f}")
    print(f"  Recall  (CCL)    : {metrics['clip']['recall_ccl']:.3f}")
    print(f"  Precision (Normal): {metrics['clip']['precision_normal']:.3f}")
    print(f"  Precision (CCL)   : {metrics['clip']['precision_ccl']:.3f}")
    print(f"  Confusion matrix [rows=true, cols=pred, order=(Normal, CCL)]:")
    print(f"     true=Normal  -> pred(Norm)={cm[0,0]}  pred(CCL)={cm[0,1]}")
    print(f"     true=CCL     -> pred(Norm)={cm[1,0]}  pred(CCL)={cm[1,1]}")
    print()

    # dog-level metrics - average clip probability per dog then threshold
    dog_df = pd.DataFrame({
        "dog_id": test_df["dog_id"],
        "true": y,
        "proba": proba,
    })
    agg = dog_df.groupby("dog_id").agg(
        true=("true", "first"),
        proba_mean=("proba", "mean"),
        n_clips=("true", "size"),
    )
    agg["pred"] = (agg["proba_mean"] >= 0.5).astype(int)
    agg["correct"] = (agg["pred"] == agg["true"]).astype(int)

    print("=== Dog-level ===")
    for dog_id, row in agg.iterrows():
        label_str = "CCL" if row["true"] == 1 else "Normal"
        pred_str = "CCL" if row["pred"] == 1 else "Normal"
        ok = "OK" if row["correct"] else "WRONG"
        print(f"  {dog_id:35s}  true={label_str:6s}  pred={pred_str:6s}  "
              f"P(CCL)={row['proba_mean']:.3f}  n={int(row['n_clips'])}  [{ok}]")

    metrics["dog"]["accuracy"] = float(agg["correct"].mean())
    try:
        metrics["dog"]["auc"] = float(roc_auc_score(agg["true"], agg["proba_mean"]))
    except ValueError:
        metrics["dog"]["auc"] = float("nan")

    print(f"\n  Dog accuracy: {metrics['dog']['accuracy']:.3f}")
    print(f"  Dog AUC     : {metrics['dog']['auc']:.3f}")

    # write output files
    test_df_out = test_df[["video_path", "dog_id", "dog_name", "label"]].copy()
    test_df_out["pred"] = pred
    test_df_out["proba_ccl"] = proba
    test_df_out.to_csv(TEST_PREDS_PATH, index=False)
    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {TEST_PREDS_PATH}")
    print(f"wrote {TEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
