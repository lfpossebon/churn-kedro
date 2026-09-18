"""Nodes for the 'data_engineering' pipeline.

The pipeline follows the fit/transform split: ``fit_*`` nodes learn only from
the splits listed in ``split_to_fit`` (normally just ``train``) and return
artefacts, while ``transform_*`` nodes apply those artefacts to every row.
Keeping the two apart is what prevents label and statistics leakage, and it is
what lets the refit and inference pipelines reuse these same functions.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def _expand_column_groups(
    columns: dict[str, Any],
    group_keys: list[str],
) -> list[str]:
    """Flatten the requested column groups into a single list of column names.

    Args:
        columns: Column groups dictionary mapping group keys to either a single
            column name or a list of column names.
        group_keys: Group keys to expand, e.g. ``["target", "categorical"]``.

    Returns:
        Flat list of column names.
    """
    return [
        item
        for key in group_keys
        for item in (columns[key] if isinstance(columns[key], list) else [columns[key]])
    ]


def clean_data(
    raw_churn_data: pd.DataFrame,
    columns: dict[str, Any],
) -> pd.DataFrame:
    """Select, coerce, and impute columns from a raw DataFrame.

    Columns listed in ``columns`` that are absent from the DataFrame are
    silently skipped, so the same function works for both training data
    (which includes a target column) and inference data (which does not).
    Numerical columns are coerced to float with NaNs filled by 0;
    categorical columns are cast to stripped strings with NaNs filled by
    ``"Unknown"``.

    Args:
        raw_churn_data: Raw input DataFrame.
        columns: Column groups to process. Recognised keys:
            - ``target`` (str, optional): Target column name.
            - ``numerical`` (list[str]): Numerical feature column names.
            - ``categorical`` (list[str]): Categorical feature column names.

    Returns:
        Cleaned DataFrame containing only the columns that were both
        specified and present in the input.
    """
    df_raw = raw_churn_data.copy()

    all_columns = [
        item
        for v in columns.values()
        for item in (v if isinstance(v, list) else [v])
        if item in df_raw.columns
    ]

    df_flt = df_raw[all_columns].copy()

    for c in columns["numerical"]:
        df_flt[c] = pd.to_numeric(df_flt[c], errors="coerce").fillna(0)

    for c in columns["categorical"]:
        df_flt[c] = df_flt[c].astype(str).str.strip().fillna("Unknown")

    logger.info("Cleaned data: %d rows, %d columns", len(df_flt), len(df_flt.columns))

    return df_flt


def add_split_column(
    df_in: pd.DataFrame,
    split: dict[str, Any],
) -> pd.DataFrame:
    """Randomly assign each row to a train, test, or validate split.

    Args:
        df_in: Cleaned input DataFrame.
        split: Split configuration with keys:
            - ``train`` (float): Proportion of rows for training.
            - ``test`` (float): Proportion of rows for testing.
            - ``validate`` (float): Proportion of rows for validation.
            - ``random_state`` (int): Seed for reproducible assignment.

    Returns:
        Input DataFrame with an added ``split`` column whose values are
        ``"train"``, ``"test"``, or ``"validate"``.

    Raises:
        ValueError: If ``train + test + validate`` does not sum to 1.0.
    """
    total = split["train"] + split["test"] + split["validate"]
    probs = [split["train"], split["test"], split["validate"]]

    if not np.isclose(total, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    rng = np.random.default_rng(split["random_state"])

    labels = rng.choice(
        a=["train", "test", "validate"],
        size=len(df_in),
        p=probs,
        replace=True,
    )

    df_out = df_in.assign(split=labels)

    logger.info(
        "Split counts — train: %d, test: %d, validate: %d",
        (labels == "train").sum(),
        (labels == "test").sum(),
        (labels == "validate").sum(),
    )

    return df_out


def fit_encoders(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, LabelEncoder]:
    """Fit a LabelEncoder for each categorical column on the specified splits.

    Only rows belonging to the splits in ``params["split_to_fit"]`` are used,
    preventing label leakage from test or validation data.

    Args:
        df_in: DataFrame with a ``split`` column and all feature columns.
        columns: Column groups dictionary mapping group keys to column names.
        params: Encoder configuration with keys:
            - ``columns`` (list[str]): Group keys from ``columns`` to encode.
            - ``split_to_fit`` (list[str]): Split labels used for fitting.

    Returns:
        Mapping of column names to their fitted ``LabelEncoder`` instances.
    """
    encoders: dict[str, LabelEncoder] = {}

    cols_to_encode = _expand_column_groups(columns, params["columns"])

    for col in cols_to_encode:
        le = LabelEncoder()
        le.fit(df_in.loc[df_in["split"].isin(params["split_to_fit"]), col].astype(str))
        encoders[col] = le

    logger.info("Fitted label encoders for %d columns", len(encoders))
    return encoders


def transform_encoders(
    df_in: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """Apply fitted LabelEncoders to replace categorical columns with integer labels.

    Columns missing from the DataFrame are skipped, which is what allows the
    inference pipeline — where the target column is absent — to reuse the
    encoders fitted during training.

    Args:
        df_in: DataFrame containing the columns to encode.
        encoders: Mapping of column names to fitted ``LabelEncoder`` instances.

    Returns:
        DataFrame with each encoded column replaced by its integer labels.
    """
    df_out = df_in.copy()

    for col, en in encoders.items():
        if col in df_out.columns:
            df_out[col] = en.transform(df_out[col].astype(str))

    return df_out


def fit_scalers(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, StandardScaler]:
    """Fit a StandardScaler for each numerical column on the specified splits.

    Only rows belonging to the splits in ``params["split_to_fit"]`` are used,
    preventing statistics leakage from test or validation data.

    Args:
        df_in: DataFrame with a ``split`` column and all feature columns.
        columns: Column groups dictionary mapping group keys to column names.
        params: Scaler configuration with keys:
            - ``columns`` (list[str]): Group keys from ``columns`` to scale.
            - ``split_to_fit`` (list[str]): Split labels used for fitting.

    Returns:
        Mapping of column names to their fitted ``StandardScaler`` instances.
    """
    scalers: dict[str, StandardScaler] = {}

    cols_to_scale = _expand_column_groups(columns, params["columns"])

    for col in cols_to_scale:
        scaler = StandardScaler()
        scaler.fit(df_in.loc[df_in["split"].isin(params["split_to_fit"]), [col]])
        scalers[col] = scaler

    logger.info("Fitted scalers for %d columns", len(scalers))
    return scalers


def transform_scalers(
    df_in: pd.DataFrame,
    scalers: dict[str, StandardScaler],
) -> pd.DataFrame:
    """Apply fitted StandardScalers to standardize numerical columns.

    Args:
        df_in: DataFrame containing the columns to scale.
        scalers: Mapping of column names to fitted ``StandardScaler`` instances.

    Returns:
        DataFrame with each scaled column replaced by its standardized values.
    """
    df_out = df_in.copy()

    for col, scaler in scalers.items():
        if col in df_out.columns:
            df_out[col] = scaler.transform(df_out[[col]])

    return df_out
