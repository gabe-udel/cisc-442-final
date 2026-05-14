"""Train the CCL-vs-healthy binary classifier.

Reads results/features.csv (built by `features.py`), splits off a held-out
test set of 1 CCL dog + 1 Normal dog, then runs leave-one-dog-out (LOO) CV
on the rest of the dogs to compare three classifiers:

    - LogisticRegression  (cheap, interpretable, baseline)
    - RandomForest        (handles non-linear interactions, robust to scale)
    - XGBoost             (typically the strongest tabular model on small data)

Why grouped CV (by dog) instead of random?
    Two clips from the same dog share traits — gait kinematics, body size,
    camera setup. A random clip-level split would let the model "memorize"
    the dog instead of learning injury signal, inflating apparent accuracy
    by 10-30 points. The reported AUC under LOO-by-dog is the best
    out-of-distribution-dog estimate

Per-dog scoring:
    Each fold tests on all clips from one held-out dog. We report both
    clip-level AUC/acc (treating every clip independently) and dog-level
    accuracy (majority-vote across the dog's clips).

Outputs:
    results/cv_metrics.json     # per-model LOO scores
    results/model.joblib        # the winning classifier, refit on full train set
    results/holdout_dogs.json   # which two dogs are in the test set

Run:
    python src/train.py
    python src/train.py --holdout-ccl "Reggie Bell 927479" --holdout-normal "..."
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from dataset import RESULTS_DIR
from features import feature_columns, ALL_COLUMNS, FEATURES_CSV


CV_METRICS_PATH = RESULTS_DIR / "cv_metrics.json"
MODEL_PATH = RESULTS_DIR / "model.joblib"
HOLDOUT_PATH = RESULTS_DIR / "holdout_dogs.json"


# sklearn-compatible Pipeline that includes imputation

def _make_models() -> dict[str, Pipeline]:
    """Three classifiers with sensible small-data hyperparameters.

    All use class_weight / scale_pos_weight = balanced so the slight class
    imbalance (7 CCL vs 8 Normal dogs) doesn't bias the boundary.
    """
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced", solver="liblinear",
            )),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=3,
                class_weight="balanced", random_state=0, n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline([ # ALSO --- we **COULD** do bayesian parameter optimization using hyperopt, at least for random forest and xgboost.
            # these parameters can be heavily messed with and we could probably see some pretty large improvements (nclass = 1200?)
            ("clf", xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=0, n_jobs=-1, eval_metric="logloss",
            )),
        ]),
    }


def _pick_holdout(df: pd.DataFrame,
                  ccl_dog: str | None,
                  normal_dog: str | None) -> tuple[str, str]:
    """Return (ccl_dog_id, normal_dog_id) chosen for the held-out test set.

    Defaults: pick the lexicographically last dog in each class — deterministic
    and reproducible. Override with explicit dog ids on the command line if you
    want to use specific dogs.
    """
    ccl_dogs = sorted(df.loc[df["label"] == 1, "dog_id"].unique())
    norm_dogs = sorted(df.loc[df["label"] == 0, "dog_id"].unique())
    if not ccl_dogs or not norm_dogs:
        raise RuntimeError("Need at least one CCL and one Normal dog in features.csv.")
    chosen_ccl = ccl_dog or ccl_dogs[-1]
    chosen_norm = normal_dog or norm_dogs[-1]
    if chosen_ccl not in ccl_dogs:
        raise ValueError(f"--holdout-ccl '{chosen_ccl}' not in CCL dogs: {ccl_dogs}")
    if chosen_norm not in norm_dogs:
        raise ValueError(f"--holdout-normal '{chosen_norm}' not in Normal dogs: {norm_dogs}")
    return chosen_ccl, chosen_norm


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _per_dog_predictions(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run leave-one-dog-out CV. Returns (clip_pred, clip_proba, clip_true,
    clip_group) — one entry per clip across all folds.

    Suppresses LR convergence warnings on tiny folds — they don't indicate a
    real problem here.
    """
    logo = LeaveOneGroupOut()
    n = len(X)
    pred = np.zeros(n, dtype=int)
    proba = np.zeros(n, dtype=float)
    true = y.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fold_idx, (tr, te) in enumerate(logo.split(X, y, groups)):
            pipe.fit(X.iloc[tr], y[tr])
            pred[te] = pipe.predict(X.iloc[te])
            # predict_proba returns shape (n, 2); column 1 is P(label=1) = injured.
            proba[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return pred, proba, true, groups


def _summarize_cv(pred, proba, true, groups) -> dict:
    """Compute clip-level and dog-level metrics from out-of-fold predictions."""
    out = {
        "n_clips": int(len(true)),
        "n_dogs": int(len(np.unique(groups))),
        "clip_accuracy": float(accuracy_score(true, pred)),
    }
    # AUC requires both classes to be present and probabilities. If one class
    # is missing in `true` (shouldn't happen for sensible holdouts) we return
    # NaN.
    try:
        out["clip_auc"] = float(roc_auc_score(true, proba))
    except ValueError:
        out["clip_auc"] = float("nan")

    # Aggregate to dog level: a dog is predicted CCL if MEAN of its clip
    # probabilities is >= 0.5. Dog true label is the same across its clips.
    df = pd.DataFrame({"true": true, "proba": proba, "group": groups})
    dog_agg = df.groupby("group").agg(
        true=("true", "first"),
        proba_mean=("proba", "mean"),
    )
    dog_agg["pred"] = (dog_agg["proba_mean"] >= 0.5).astype(int)
    out["dog_accuracy"] = float((dog_agg["pred"] == dog_agg["true"]).mean())
    try:
        out["dog_auc"] = float(roc_auc_score(dog_agg["true"], dog_agg["proba_mean"]))
    except ValueError:
        out["dog_auc"] = float("nan")
    return out

def main():
    p = argparse.ArgumentParser(description="Train CCL-vs-healthy classifier.")
    p.add_argument("--holdout-ccl", type=str, default=None,
                   help="Dog id (folder name) of the CCL dog to hold out for final test.")
    p.add_argument("--holdout-normal", type=str, default=None,
                   help="Dog id of the Normal dog to hold out for final test.")
    args = p.parse_args()

    if not FEATURES_CSV.exists():
        raise SystemExit(f"{FEATURES_CSV} missing — run features.py first.")
    df = pd.read_csv(FEATURES_CSV)
    if df.empty:
        raise SystemExit("features.csv is empty.")

    feat_cols = feature_columns()
    missing_cols = [c for c in feat_cols if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"features.csv is missing columns: {missing_cols}")

    holdout_ccl, holdout_norm = _pick_holdout(df, args.holdout_ccl, args.holdout_normal)
    print(f"Holdout dogs:  CCL = {holdout_ccl!r}    Normal = {holdout_norm!r}")

    is_holdout = df["dog_id"].isin([holdout_ccl, holdout_norm])
    train_df = df.loc[~is_holdout].reset_index(drop=True)
    test_df = df.loc[is_holdout].reset_index(drop=True)
    print(f"Train: {len(train_df)} clips across {train_df['dog_id'].nunique()} dogs "
          f"({(train_df['label'] == 1).sum()} CCL clips, "
          f"{(train_df['label'] == 0).sum()} Normal clips)")
    print(f"Test:  {len(test_df)} clips across {test_df['dog_id'].nunique()} dogs")
    print()

    if train_df["dog_id"].nunique() < 3:
        raise SystemExit("Need at least 3 dogs in the training set for LOO-CV.")

    X_train = train_df[feat_cols]
    y_train = train_df["label"].astype(int).to_numpy()
    groups_train = train_df["dog_id"].to_numpy()

    print("=== Leave-one-dog-out cross-validation ===")
    cv_results: dict[str, dict] = {}
    for name, pipe in _make_models().items():
        try:
            pred, proba, true, groups = _per_dog_predictions(pipe, X_train, y_train, groups_train)
            metrics = _summarize_cv(pred, proba, true, groups)
        except Exception as e:
            metrics = {"error": str(e)}
        cv_results[name] = metrics
        if "error" in metrics:
            print(f"  {name:14s}  ERROR: {metrics['error']}")
        else:
            print(f"  {name:14s}  clip AUC={metrics['clip_auc']:.3f}  "
                  f"clip acc={metrics['clip_accuracy']:.3f}    "
                  f"DOG AUC={metrics['dog_auc']:.3f}  DOG acc={metrics['dog_accuracy']:.3f}")

    #  Pick winner by DOG-level AUC
    valid = {n: m for n, m in cv_results.items()
             if "error" not in m and not np.isnan(m.get("dog_auc", np.nan))}
    if not valid:
        valid = {n: m for n, m in cv_results.items() if "error" not in m}
    if not valid:
        raise SystemExit("All models errored during CV.")
    winner = max(valid, key=lambda n: valid[n].get("dog_auc",
                                                    valid[n].get("clip_auc", -1)))
    print(f"\nBest model by dog-level AUC: {winner}  "
          f"(dog AUC={valid[winner].get('dog_auc'):.3f}, "
          f"dog acc={valid[winner].get('dog_accuracy'):.3f})")

    winning_pipe = _make_models()[winner]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        winning_pipe.fit(X_train, y_train)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": winning_pipe,
        "feature_columns": feat_cols,
        "model_name": winner,
    }, MODEL_PATH)
    with open(CV_METRICS_PATH, "w") as f:
        json.dump(cv_results, f, indent=2)
    with open(HOLDOUT_PATH, "w") as f:
        json.dump({"ccl": holdout_ccl, "normal": holdout_norm}, f, indent=2)

    print(f"\nwrote {MODEL_PATH}")
    print(f"wrote {CV_METRICS_PATH}")
    print(f"wrote {HOLDOUT_PATH}")
    print(f"\nNext: python src/evaluate.py")


if __name__ == "__main__":
    main()
