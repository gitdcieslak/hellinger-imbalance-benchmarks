"""Loader for curated legacy HDDT datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _normalize_target_column_name(target_column: str, columns: list[str]) -> str:
    if target_column in columns:
        return target_column
    try:
        idx = int(target_column)
    except ValueError as exc:
        raise ValueError(f"target column {target_column!r} not found in dataset") from exc
    as_str = str(idx)
    if as_str in columns:
        return as_str
    raise ValueError(f"target column {target_column!r} not found in dataset")


def _deterministic_encode_column(series: pd.Series) -> np.ndarray:
    missing_mask = series.isna()
    text_values = series.astype(str)
    text_values = text_values.where(~missing_mask, "__MISSING__")
    unique_values = sorted(text_values.unique().tolist())
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    return text_values.map(mapping).to_numpy(dtype=float)


def load_legacy_hddt_dataset(
    extracted_dir: Path,
    dataset_entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one legacy HDDT dataset into numeric X and binary y."""

    extracted_dir = extracted_dir.resolve()
    relative_path = str(dataset_entry["relative_path"])
    dataset_path = (extracted_dir / relative_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")

    frame = pd.read_csv(
        dataset_path,
        sep=",",
        header=None,
        na_values=["?", "", "NA", "N/A", "null", "None"],
        keep_default_na=True,
        dtype=str,
    )
    frame.columns = [str(column) for column in frame.columns]

    target_col = _normalize_target_column_name(str(dataset_entry["target_column"]), list(frame.columns))
    positive_class = str(dataset_entry["positive_class"])

    target_series = frame[target_col].astype(str).str.strip()
    available_classes = set(target_series.unique().tolist())
    if positive_class not in available_classes:
        found = ", ".join(sorted(available_classes))
        raise ValueError(
            f"configured positive class {positive_class!r} not found for {dataset_entry['dataset_id']}; "
            f"available classes: {found}"
        )

    y = (target_series == positive_class).astype(int).to_numpy(dtype=int)
    if int(np.unique(y).shape[0]) != 2:
        raise ValueError(f"dataset {dataset_entry['dataset_id']} does not resolve to binary labels")

    X_frame = frame.drop(columns=[target_col]).copy()
    for column in X_frame.columns:
        stripped = X_frame[column].astype(str).str.strip()
        stripped = stripped.replace({"?": np.nan, "": np.nan, "None": np.nan, "null": np.nan})

        numeric = pd.to_numeric(stripped, errors="coerce")
        numeric_non_missing = int(numeric.notna().sum())
        if numeric_non_missing >= int(max(1, 0.8 * len(stripped.dropna()))):
            if numeric.notna().sum() == 0:
                X_frame[column] = 0.0
            else:
                median = float(numeric.median())
                X_frame[column] = numeric.fillna(median).astype(float)
        else:
            X_frame[column] = _deterministic_encode_column(stripped)

    X = X_frame.to_numpy(dtype=float)
    metadata = {
        "dataset_id": dataset_entry["dataset_id"],
        "relative_path": relative_path,
        "target_column": target_col,
        "positive_class": positive_class,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "source_group": dataset_entry.get("source_group"),
    }
    return X, y, metadata
