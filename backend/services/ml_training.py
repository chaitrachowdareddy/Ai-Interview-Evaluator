"""Reproducible Phase 4 training pipeline for the interview-score model.

The repository does not ship a labelled dataset. This module is the supported
way to train a model once an operator has a licensed CSV with one row per
answer and a human-provided 0-100 overall score. It refuses to train a model
claimed as validated when the data is too small or the held-out result is no
better than predicting the mean.

Expected CSV columns::

    group_id,target_score,semantic_score,keyword_score,...

``group_id`` identifies a candidate or interview and is used only for a
group-aware split. It is never a model feature. Identity-like columns
(``sample_id``, ``candidate_id``, ``interview_id``, ``answer_id``) are accepted
but deliberately ignored so they cannot leak into the model.

Usage::

    python -m services.ml_training --data labeled_scores.csv
    python -m services.ml_training --data labeled_scores.csv --allow-small
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

import config
from services.ml_features import (
    FEATURE_NAMES,
    FeatureValidationError,
    validate_feature_value,
)

IGNORED_COLUMNS = {
    "sample_id",
    "candidate_id",
    "interview_id",
    "answer_id",
}
REQUIRED_COLUMNS = {"group_id", "target_score", *FEATURE_NAMES}
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}

MIN_TEST_SAMPLES = 3


class TrainingDataError(Exception):
    """The labelled dataset cannot produce a trustworthy model."""


@dataclass(frozen=True)
class TrainingRow:
    sample_id: str
    group_id: str
    target: float
    features: dict[str, float | None]


@dataclass(frozen=True)
class TrainingDataset:
    rows: tuple[TrainingRow, ...]
    ignored_columns: tuple[str, ...]
    dataset_sha256: str
    warnings: tuple[str, ...]
    target_stats: dict[str, float]
    source_name: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_feature(
    raw: str | None,
    name: str,
    row_number: int,
) -> float | None:
    text = (raw or "").strip()
    if text.lower() in MISSING_TOKENS:
        value = None
    else:
        try:
            value = float(text)
        except ValueError as exc:
            raise TrainingDataError(
                f"Row {row_number}: feature '{name}' is not numeric: {text!r}."
            ) from exc
    try:
        return validate_feature_value(name, value)
    except FeatureValidationError as exc:
        raise TrainingDataError(
            f"Row {row_number}: {exc.message}"
        ) from exc


def load_training_dataset(path: str | Path) -> TrainingDataset:
    """Read and validate a labelled CSV without fabricating missing values."""
    source = Path(path)
    if not source.is_file():
        raise TrainingDataError(f"Training data file not found: {source.name}.")

    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [str(name or "").strip() for name in (reader.fieldnames or [])]
            raw_rows = list(reader)
    except OSError as exc:
        raise TrainingDataError(
            "The training data file could not be read."
        ) from exc
    except csv.Error as exc:
        raise TrainingDataError("The training data is not valid CSV.") from exc

    if not fieldnames:
        raise TrainingDataError("The training data has no header row.")
    if len(fieldnames) != len(set(fieldnames)):
        raise TrainingDataError("The training data has duplicate column names.")

    columns = set(fieldnames)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise TrainingDataError(
            "The training data is missing required column(s): "
            + ", ".join(missing)
            + "."
        )

    unknown = sorted(columns - REQUIRED_COLUMNS - IGNORED_COLUMNS)
    if unknown:
        raise TrainingDataError(
            "The training data has unrecognized column(s): "
            + ", ".join(unknown)
            + ". Rename them or remove them so no unintended feature or target "
            "leaks into the model."
        )

    warnings: list[str] = []
    ignored = tuple(sorted(columns & IGNORED_COLUMNS))
    if ignored:
        warnings.append(
            "Ignored identity-like column(s), never used as features: "
            + ", ".join(ignored)
            + "."
        )

    rows: list[TrainingRow] = []
    seen_sample_ids: dict[str, int] = {}
    seen_rows: dict[tuple[Any, ...], int] = {}
    for index, raw in enumerate(raw_rows, start=2):
        group_id = str(raw.get("group_id") or "").strip()
        if not group_id:
            raise TrainingDataError(f"Row {index}: group_id is required.")

        target_text = str(raw.get("target_score") or "").strip()
        try:
            target = float(target_text)
        except ValueError as exc:
            raise TrainingDataError(
                f"Row {index}: target_score is not numeric: {target_text!r}."
            ) from exc
        if (
            not math.isfinite(target)
            or not config.ML_TARGET_MIN <= target <= config.ML_TARGET_MAX
        ):
            raise TrainingDataError(
                f"Row {index}: target_score must be between "
                f"{config.ML_TARGET_MIN:g} and {config.ML_TARGET_MAX:g}."
            )

        features = {
            name: _parse_feature(raw.get(name), name, index)
            for name in FEATURE_NAMES
        }

        sample_id = str(raw.get("sample_id") or f"row-{index}").strip()
        if sample_id in seen_sample_ids:
            raise TrainingDataError(
                f"Row {index}: duplicate sample_id '{sample_id}' "
                f"(first seen on row {seen_sample_ids[sample_id]})."
            )
        seen_sample_ids[sample_id] = index

        duplicate_key = (
            group_id,
            target,
            *(features[name] for name in FEATURE_NAMES),
        )
        if duplicate_key in seen_rows:
            raise TrainingDataError(
                f"Row {index}: exact duplicate of row {seen_rows[duplicate_key]} "
                "within the same group. Remove duplicates so the same answer is "
                "not counted twice."
            )
        seen_rows[duplicate_key] = index

        rows.append(
            TrainingRow(
                sample_id=sample_id,
                group_id=group_id,
                target=target,
                features=features,
            )
        )

    if not rows:
        raise TrainingDataError("The training data contains no rows.")

    targets = np.asarray([row.target for row in rows], dtype=float)
    target_stats = {
        "mean": round(float(targets.mean()), 4),
        "std": round(float(targets.std(ddof=0)), 4),
        "min": round(float(targets.min()), 4),
        "max": round(float(targets.max()), 4),
        "p25": round(float(np.percentile(targets, 25)), 4),
        "median": round(float(np.median(targets)), 4),
        "p75": round(float(np.percentile(targets, 75)), 4),
    }
    if target_stats["std"] < 1.0:
        warnings.append(
            "Target scores have almost no variation. A model cannot learn a "
            "meaningful ranking from this dataset."
        )

    return TrainingDataset(
        rows=tuple(rows),
        ignored_columns=ignored,
        dataset_sha256=_sha256_file(source),
        warnings=tuple(warnings),
        target_stats=target_stats,
        source_name=source.name,
    )


def split_rows(
    rows: list[TrainingRow] | tuple[TrainingRow, ...],
    *,
    test_fraction: float,
    random_seed: int,
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """Group-aware split so one candidate/interview never spans both sides."""
    if not rows:
        raise TrainingDataError("No rows are available to split.")
    groups = np.asarray([row.group_id for row in rows])
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        raise TrainingDataError(
            "At least two distinct group_id values are required for a "
            "group-aware train/test split."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=random_seed,
    )
    train_index, test_index = next(
        splitter.split(np.zeros(len(rows)), groups=groups)
    )
    train = [rows[int(index)] for index in train_index]
    test = [rows[int(index)] for index in test_index]

    train_groups = {row.group_id for row in train}
    test_groups = {row.group_id for row in test}
    overlap = train_groups & test_groups
    if overlap:
        # This should be impossible with GroupShuffleSplit; fail loudly rather
        # than train on leaked data if the implementation ever changes.
        raise TrainingDataError(
            "Group leakage detected between train and test splits: "
            + ", ".join(sorted(overlap))
        )
    if not train or not test:
        raise TrainingDataError(
            "The configured test fraction produced an empty train or test split."
        )
    return train, test


def _matrix(rows: list[TrainingRow]) -> np.ndarray:
    return np.asarray(
        [
            [
                math.nan if row.features[name] is None else float(row.features[name])
                for name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=float,
    )


def _targets(rows: list[TrainingRow]) -> np.ndarray:
    return np.asarray([row.target for row in rows], dtype=float)


def _regression_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    mae = float(mean_absolute_error(expected, predicted))
    rmse = float(math.sqrt(mean_squared_error(expected, predicted)))
    r2 = (
        round(float(r2_score(expected, predicted)), 4)
        if len(expected) >= 2
        else None
    )
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": r2,
    }


def _atomic_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _atomic_write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def train_from_file(
    data_path: str | Path,
    *,
    output_path: str | Path | None = None,
    allow_small: bool = False,
    overwrite: bool = False,
    test_fraction: float | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and save one versioned artifact bundle.

    A ``validated`` artifact requires: both sample/group thresholds, enough
    held-out rows, a non-degenerate target, and a held-out MAE better than a
    mean-prediction baseline. Otherwise the artifact is marked ``unvalidated``
    and inference refuses to use it.
    """
    dataset = load_training_dataset(data_path)
    rows = list(dataset.rows)
    fraction = config.ML_TEST_FRACTION if test_fraction is None else test_fraction
    seed = config.ML_RANDOM_SEED if random_seed is None else random_seed
    if not 0.1 <= fraction <= 0.5:
        raise TrainingDataError("test_fraction must be between 0.1 and 0.5.")

    destination = Path(output_path or config.ML_MODEL_PATH)
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if not overwrite and (destination.exists() or metadata_path.exists()):
        raise TrainingDataError(
            f"An artifact already exists at {destination.name}. Pass --force to "
            "replace it deliberately."
        )

    warnings = list(dataset.warnings)
    sample_count = len(rows)
    group_count = len({row.group_id for row in rows})

    if sample_count < config.ML_MIN_TRAINING_SAMPLES:
        message = (
            f"Only {sample_count} labelled sample(s) are available; at least "
            f"{config.ML_MIN_TRAINING_SAMPLES} are required for a validated "
            "model."
        )
        if not allow_small:
            raise TrainingDataError(message + " Pass --allow-small to produce "
                                    "an explicitly unvalidated experimental artifact.")
        warnings.append(message)
    if group_count < config.ML_MIN_TRAINING_GROUPS:
        message = (
            f"Only {group_count} group(s) are available; at least "
            f"{config.ML_MIN_TRAINING_GROUPS} are required for a validated "
            "model."
        )
        if not allow_small:
            raise TrainingDataError(message + " Pass --allow-small to produce "
                                    "an explicitly unvalidated experimental artifact.")
        warnings.append(message)

    train_rows, test_rows = split_rows(
        rows,
        test_fraction=fraction,
        random_seed=seed,
    )
    train_groups = {row.group_id for row in train_rows}
    test_groups = {row.group_id for row in test_rows}
    if not train_groups or not test_groups:
        raise TrainingDataError("The split produced an empty group set.")

    x_train = _matrix(train_rows)
    y_train = _targets(train_rows)
    x_test = _matrix(test_rows)
    y_test = _targets(test_rows)

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    train_predicted = pipeline.predict(x_train)
    test_predicted = pipeline.predict(x_test)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x_train, y_train)
    dummy_predicted = dummy.predict(x_test)

    metrics = {
        "test": _regression_metrics(y_test, test_predicted),
        "train": _regression_metrics(y_train, train_predicted),
        "mean_baseline_test": _regression_metrics(y_test, dummy_predicted),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "test_fraction": fraction,
        "random_seed": seed,
        "target_stats": dataset.target_stats,
    }
    metrics["improves_on_mean_baseline"] = bool(
        metrics["test"]["mae"] < metrics["mean_baseline_test"]["mae"]
    )

    validation_reasons: list[str] = []
    if sample_count < config.ML_MIN_TRAINING_SAMPLES:
        validation_reasons.append("below the minimum sample count")
    if group_count < config.ML_MIN_TRAINING_GROUPS:
        validation_reasons.append("below the minimum group count")
    if len(test_rows) < MIN_TEST_SAMPLES:
        validation_reasons.append("too few held-out samples")
    if dataset.target_stats["std"] < 1.0:
        validation_reasons.append("target scores have almost no variation")
    if not metrics["improves_on_mean_baseline"]:
        validation_reasons.append(
            "held-out MAE does not beat the mean-prediction baseline"
        )
    if metrics["test"]["r2"] is not None and metrics["test"]["r2"] < 0:
        validation_reasons.append("held-out R-squared is negative")

    if (
        metrics["train"]["mae"] is not None
        and metrics["test"]["mae"] > max(5.0, metrics["train"]["mae"] * 2.0)
    ):
        warnings.append(
            "Possible overfitting: held-out MAE is more than twice the training "
            "MAE. Review dataset size and feature coverage before use."
        )

    validation_status = "validated" if not validation_reasons else "unvalidated"
    if validation_reasons:
        warnings.append(
            "Artifact marked unvalidated: " + "; ".join(validation_reasons) + "."
        )

    model_version = (
        "rf-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + dataset.dataset_sha256[:8]
    )
    limitations = [
        f"Trained on {sample_count} labelled answer(s) from {group_count} "
        "group(s) and evaluated with one group-aware holdout split.",
        "Metrics are estimates on this dataset only; they are not a guarantee "
        "of performance on other candidates, roles, or recording conditions.",
        "The model predicts a human-provided 0-100 score. It does not measure "
        "truthfulness, personality, or fitness for a job.",
        "Review subgroup fairness and calibration on representative data before "
        "using predictions in any decision.",
    ]
    trained_at = _utc_now()

    bundle = {
        "model": pipeline,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_version": config.ML_FEATURE_SCHEMA_VERSION,
        "methodology_version": config.ML_PREDICTION_METHODOLOGY_VERSION,
        "model_version": model_version,
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
        "validation_status": validation_status,
        "metrics": metrics,
        "limitations": limitations,
        "target_min": config.ML_TARGET_MIN,
        "target_max": config.ML_TARGET_MAX,
    }
    metadata = {
        **{key: value for key, value in bundle.items() if key != "model"},
        "artifact_filename": destination.name,
        "dataset": {
            "name": dataset.source_name,
            "sha256": dataset.dataset_sha256,
            "samples": sample_count,
            "groups": group_count,
            "ignored_columns": list(dataset.ignored_columns),
        },
        "validation_reasons": validation_reasons,
        "warnings": warnings,
    }

    _atomic_dump(bundle, destination)
    _atomic_write_json(metadata, metadata_path)

    return {
        "model_version": model_version,
        "validation_status": validation_status,
        "validation_reasons": validation_reasons,
        "feature_schema_version": config.ML_FEATURE_SCHEMA_VERSION,
        "methodology_version": config.ML_PREDICTION_METHODOLOGY_VERSION,
        "output_path": str(destination),
        "metadata_path": str(metadata_path),
        "dataset": metadata["dataset"],
        "metrics": metrics,
        "warnings": warnings,
        "limitations": limitations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Phase 4 interview-score Random Forest from a labelled CSV."
        )
    )
    parser.add_argument("--data", required=True, help="Path to the labelled CSV.")
    parser.add_argument(
        "--output",
        default=None,
        help="Artifact path (default: configured ML_MODEL_PATH).",
    )
    parser.add_argument(
        "--allow-small",
        action="store_true",
        help="Produce an explicitly unvalidated artifact from a small dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing artifact deliberately.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="Held-out fraction between 0.1 and 0.5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic random seed.",
    )
    arguments = parser.parse_args(argv)

    try:
        report = train_from_file(
            arguments.data,
            output_path=arguments.output,
            allow_small=arguments.allow_small,
            overwrite=arguments.force,
            test_fraction=arguments.test_fraction,
            random_seed=arguments.seed,
        )
    except TrainingDataError as exc:
        print(f"Training refused: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
