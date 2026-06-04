"""Reproducible traffic demand modelling pipeline for the Gridathon ML challenge.

The script trains several strong tree models with leakage-safe cross-validated target
encoding and writes a competition-ready submission file.

Usage:
    python src/traffic_demand_solution.py --data-dir dataset --output submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

TARGET = "demand"
ID_COL = "Index"
RANDOM_STATE = 42
MISSING_CATEGORY = "__missing__"


def stringify_categorical(series: pd.Series) -> pd.Series:
    """Return a categorical series containing only plain Python strings.

    Pandas string columns keep missing values as ``pd.NA``. Scikit-learn
    encoders sort category values internally and can crash when a column mixes
    ``pd.NA`` with strings, so every categorical path normalises missing values
    before model encoding.
    """
    return series.astype("string").fillna(MISSING_CATEGORY).astype(str)


def categorical_columns(frame: pd.DataFrame, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    return [
        col
        for col in frame.columns
        if col not in excluded and (frame[col].dtype == "object" or str(frame[col].dtype).startswith("string"))
    ]


def normalize_categorical_columns(train: pd.DataFrame, test: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    for col in columns:
        train_out[col] = stringify_categorical(train_out[col])
        test_out[col] = stringify_categorical(test_out[col])
    return train_out, test_out


def read_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path) if sample_path.exists() else None
    return train, test, sample


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        out["timestamp_hour"] = ts.dt.hour
        out["timestamp_minute"] = ts.dt.minute
        out["timestamp_dayofweek"] = ts.dt.dayofweek
        out["timestamp_month"] = ts.dt.month
        out["timestamp_dayofyear"] = ts.dt.dayofyear
        out["timestamp_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype("int8")
        seconds = ts.astype("int64").where(ts.notna(), np.nan) // 1_000_000_000
        out["timestamp_seconds"] = seconds
        out["hour_sin"] = np.sin(2 * np.pi * out["timestamp_hour"].fillna(0) / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["timestamp_hour"].fillna(0) / 24)
    if "day" in out.columns:
        day_as_dt = pd.to_datetime(out["day"], errors="coerce")
        out["day_dayofweek"] = day_as_dt.dt.dayofweek
        out["day_month"] = day_as_dt.dt.month
        out["day_dayofyear"] = day_as_dt.dt.dayofyear
    return out


def add_frequency_features(train: pd.DataFrame, test: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    combined = pd.concat([train_out[list(columns)], test_out[list(columns)]], axis=0, ignore_index=True)
    for col in columns:
        freq = stringify_categorical(combined[col]).value_counts(dropna=False)
        feature = f"{col}_frequency"
        train_out[feature] = stringify_categorical(train_out[col]).map(freq).astype("float32")
        test_out[feature] = stringify_categorical(test_out[col]).map(freq).astype("float32")
    return train_out, test_out


def add_interaction_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_signature_cols = [col for col in out.columns if col not in {TARGET, ID_COL}]
    if base_signature_cols:
        out["context_signature"] = out[base_signature_cols].apply(stringify_categorical).agg("|".join, axis=1)
    if {"geohash", "timestamp_hour"}.issubset(out.columns):
        out["geohash_hour"] = stringify_categorical(out["geohash"]) + "_" + stringify_categorical(out["timestamp_hour"])
    if {"geohash", "day", "timestamp_hour"}.issubset(out.columns):
        out["geohash_day_hour"] = (
            stringify_categorical(out["geohash"]) + "_" + stringify_categorical(out["day"]) + "_" + stringify_categorical(out["timestamp_hour"])
        )
    if {"RoadType", "NumberOfLanes", "LargeVehicles"}.issubset(out.columns):
        out["road_lane_vehicle"] = (
            stringify_categorical(out["RoadType"])
            + "_"
            + stringify_categorical(out["NumberOfLanes"])
            + "_"
            + stringify_categorical(out["LargeVehicles"])
        )
    return out


def add_cv_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Iterable[str],
    n_splits: int,
    smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    global_mean = float(train_out[TARGET].mean())
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for col in columns:
        encoded = pd.Series(global_mean, index=train_out.index, dtype="float32")
        for fit_idx, valid_idx in kfold.split(train_out):
            fit_part = train_out.iloc[fit_idx]
            stats = fit_part.groupby(col, dropna=False)[TARGET].agg(["mean", "count"])
            smooth_mean = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            encoded.iloc[valid_idx] = train_out.iloc[valid_idx][col].map(smooth_mean).fillna(global_mean).astype("float32")
        full_stats = train_out.groupby(col, dropna=False)[TARGET].agg(["mean", "count"])
        full_smooth = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)
        train_out[f"{col}_target_mean"] = encoded
        test_out[f"{col}_target_mean"] = test_out[col].map(full_smooth).fillna(global_mean).astype("float32")
    return train_out, test_out


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, n_splits: int, smoothing: float) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = add_interaction_keys(add_time_features(train))
    test = add_interaction_keys(add_time_features(test))
    categorical = categorical_columns(train, excluded={TARGET})
    train, test = normalize_categorical_columns(train, test, [col for col in categorical if col in test.columns])
    train, test = add_frequency_features(train, test, categorical)
    target_encoded = [col for col in categorical if col in test.columns]
    train, test = add_cv_target_encoding(train, test, target_encoded, n_splits=n_splits, smoothing=smoothing)

    y = train[TARGET].astype("float64")
    X = train.drop(columns=[TARGET])
    X_test = test[X.columns]
    return X, y, X_test


def make_numeric_matrix(X: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical = sorted(set(categorical_columns(X)).union(categorical_columns(X_test)))
    numeric = [col for col in X.columns if col not in categorical]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    if categorical:
        X_norm, X_test_norm = normalize_categorical_columns(X, X_test, categorical)
        X_cat = pd.DataFrame(encoder.fit_transform(X_norm[categorical]), columns=categorical, index=X.index)
        T_cat = pd.DataFrame(encoder.transform(X_test_norm[categorical]), columns=categorical, index=X_test.index)
    else:
        X_cat = pd.DataFrame(index=X.index)
        T_cat = pd.DataFrame(index=X_test.index)

    X_num = X[numeric].apply(pd.to_numeric, errors="coerce")
    T_num = X_test[numeric].apply(pd.to_numeric, errors="coerce")
    X_final = pd.concat([X_num, X_cat], axis=1)
    T_final = pd.concat([T_num, T_cat], axis=1)
    return X_final, T_final


def build_models() -> list[tuple[str, object]]:
    return [
        (
            "extra_trees",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesRegressor(
                    n_estimators=700,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    max_features=0.85,
                    min_samples_leaf=1,
                    bootstrap=False,
                ),
            ),
        ),
        (
            "random_forest",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=RANDOM_STATE + 1,
                    n_jobs=-1,
                    max_features=0.85,
                    min_samples_leaf=1,
                ),
            ),
        ),
        (
            "hist_gradient_boosting",
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    learning_rate=0.045,
                    max_iter=900,
                    l2_regularization=0.01,
                    random_state=RANDOM_STATE + 2,
                ),
            ),
        ),
    ]


def cross_validate_and_predict(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, n_splits: int) -> tuple[np.ndarray, pd.DataFrame]:
    models = build_models()
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    test_predictions = []
    cv_rows = []

    for model_name, model in models:
        oof = np.zeros(len(X), dtype="float64")
        fold_test = np.zeros((len(X_test), n_splits), dtype="float64")
        for fold, (fit_idx, valid_idx) in enumerate(folds.split(X), start=1):
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            oof[valid_idx] = model.predict(X.iloc[valid_idx])
            fold_test[:, fold - 1] = model.predict(X_test)
            fold_score = r2_score(y.iloc[valid_idx], oof[valid_idx])
            cv_rows.append({"model": model_name, "fold": fold, "r2": fold_score})
        cv_rows.append({"model": model_name, "fold": "overall", "r2": r2_score(y, oof)})
        test_predictions.append(fold_test.mean(axis=1))

    prediction_frame = pd.DataFrame({name: preds for (name, _), preds in zip(models, test_predictions)})
    prediction_frame["ensemble_mean"] = prediction_frame.mean(axis=1)
    return prediction_frame["ensemble_mean"].to_numpy(), pd.DataFrame(cv_rows)


def write_submission(test: pd.DataFrame, sample: pd.DataFrame | None, predictions: np.ndarray, output: Path) -> None:
    if sample is not None:
        submission = sample.copy()
        target_column = [col for col in submission.columns if col != ID_COL][0]
        submission[target_column] = predictions
        if ID_COL in test.columns and ID_COL in submission.columns and len(submission) == len(test):
            submission[ID_COL] = test[ID_COL].values
    else:
        submission = pd.DataFrame({ID_COL: test[ID_COL] if ID_COL in test.columns else np.arange(len(test)), TARGET: predictions})
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ensemble demand model and create submission.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"), help="Folder containing train.csv, test.csv, and sample_submission.csv")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Submission CSV path")
    parser.add_argument("--cv-report", type=Path, default=Path("cv_report.csv"), help="Cross-validation report path")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--smoothing", type=float, default=20.0, help="Target-encoding smoothing strength")
    args = parser.parse_args()

    train, test, sample = read_dataset(args.data_dir)
    X, y, X_test = prepare_features(train, test, n_splits=args.folds, smoothing=args.smoothing)
    X_matrix, X_test_matrix = make_numeric_matrix(X, X_test)
    predictions, cv_report = cross_validate_and_predict(X_matrix, y, X_test_matrix, n_splits=args.folds)
    write_submission(test, sample, predictions, args.output)
    args.cv_report.parent.mkdir(parents=True, exist_ok=True)
    cv_report.to_csv(args.cv_report, index=False)
    print(f"Wrote {args.output} and {args.cv_report}")


if __name__ == "__main__":
    main()
