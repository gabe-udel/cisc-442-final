# train.py - runs cross-validation across three classifiers and saves the best one

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


# build the three candidate classifier pipelines
def _make_models():
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", solver="liblinear",
        )),
    ])

    random_forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=0, n_jobs=-1,
        )),
    ])

    # xgboost handles nan natively so no imputer needed
    xgboost = Pipeline([
        ("clf", xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=0, n_jobs=-1, eval_metric="logloss",
        )),
    ])

    return {
        "logreg": logreg,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }


# pick which dogs to hold out for the final test set
def _pick_holdout(df, ccl_dog, normal_dog):
    ccl_rows = df.loc[df["label"] == 1]
    ccl_dogs = sorted(ccl_rows["dog_id"].unique())
    norm_rows = df.loc[df["label"] == 0]
    norm_dogs = sorted(norm_rows["dog_id"].unique())
    if not ccl_dogs or not norm_dogs:
        raise RuntimeError("Need at least one CCL and one Normal dog in features.csv.")
    # default to the last dog alphabetically if not specified
    if ccl_dog is not None:
        chosen_ccl = ccl_dog
    else:
        chosen_ccl = ccl_dogs[-1]
    if normal_dog is not None:
        chosen_norm = normal_dog
    else:
        chosen_norm = norm_dogs[-1]
    if chosen_ccl not in ccl_dogs:
        raise ValueError(f"--holdout-ccl '{chosen_ccl}' not in CCL dogs: {ccl_dogs}")
    if chosen_norm not in norm_dogs:
        raise ValueError(f"--holdout-normal '{chosen_norm}' not in Normal dogs: {norm_dogs}")
    return chosen_ccl, chosen_norm


# run leave-one-dog-out cross-validation and collect out-of-fold predictions
def _per_dog_predictions(pipe, X, y, groups):
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
            proba[te] = pipe.predict_proba(X.iloc[te])[:, 1]

    return pred, proba, true, groups


# compute clip-level and dog-level metrics from out-of-fold predictions
def _summarize_cv(pred, proba, true, groups):
    out = {
        "n_clips": int(len(true)),
        "n_dogs": int(len(np.unique(groups))),
        "clip_accuracy": float(accuracy_score(true, pred)),
    }
    try:
        out["clip_auc"] = float(roc_auc_score(true, proba))
    except ValueError:
        out["clip_auc"] = float("nan")

    # aggregate clip probabilities per dog and score at the dog level
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


# holdout_ccl and holdout_normal can be passed directly from main.py
# if not passed, we read them from command line arguments
def main(holdout_ccl=None, holdout_normal=None):
    p = argparse.ArgumentParser(description="Train CCL-vs-healthy classifier.")
    p.add_argument("--holdout-ccl", type=str, default=None)
    p.add_argument("--holdout-normal", type=str, default=None)
    args, _ = p.parse_known_args()
    if args.holdout_ccl is not None:
        holdout_ccl = args.holdout_ccl
    if args.holdout_normal is not None:
        holdout_normal = args.holdout_normal

    if not FEATURES_CSV.exists():
        raise SystemExit(f"{FEATURES_CSV} missing — run features.py first.")
    df = pd.read_csv(FEATURES_CSV)
    if df.empty:
        raise SystemExit("features.csv is empty.")

    feat_cols = feature_columns()
    missing_cols = []
    for c in feat_cols:
        if c not in df.columns:
            missing_cols.append(c)
    if missing_cols:
        raise SystemExit(f"features.csv is missing columns: {missing_cols}")

    holdout_ccl, holdout_norm = _pick_holdout(df, holdout_ccl, holdout_normal)
    print(f"Holdout dogs:  CCL = {holdout_ccl!r}    Normal = {holdout_norm!r}")

    # split off the holdout dogs - they won't be used for training or cv
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
    cv_results = {}
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

    # pick the winner by dog-level auc (clinically meaningful)
    valid = {}
    for n, m in cv_results.items():
        if "error" not in m and not np.isnan(m.get("dog_auc", np.nan)):
            valid[n] = m
    if not valid:
        valid = {}
        for n, m in cv_results.items():
            if "error" not in m:
                valid[n] = m
    if not valid:
        raise SystemExit("All models errored during CV.")

    # find the model with the highest dog-level auc score
    best_name = None
    best_score = -1.0
    for n in valid:
        score = valid[n].get("dog_auc", valid[n].get("clip_auc", -1))
        if score > best_score:
            best_score = score
            best_name = n
    winner = best_name
    print(f"\nBest model by dog-level AUC: {winner}  "
          f"(dog AUC={valid[winner].get('dog_auc'):.3f}, "
          f"dog acc={valid[winner].get('dog_accuracy'):.3f})")

    # refit the winning model on all training data and save it
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
