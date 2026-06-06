"""Utilities for loading, cleaning, splitting, and grouping fraud data."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_merge(transaction_path: str, identity_path: str) -> pd.DataFrame:
    """Load transaction and identity CSV files and merge them on ``TransactionID``.

    Parameters
    ----------
    transaction_path:
        Path to ``train_transaction.csv``.
    identity_path:
        Path to ``train_identity.csv``.

    Returns
    -------
    pd.DataFrame
        A left-merged dataframe using transaction data as the left table.
    """

    transactions = pd.read_csv(transaction_path)
    identities = pd.read_csv(identity_path)
    return transactions.merge(identities, on="TransactionID", how="left")


def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Clean a dataframe and label-encode categorical columns.

    Cleaning steps:
    1. Drop columns with more than 50% missing values.
    2. Fill numeric nulls with the column median.
    3. Fill categorical nulls with ``"missing"``.
    4. Label-encode categorical columns and return the fitted encoders.

    Parameters
    ----------
    df:
        Input dataframe.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, LabelEncoder]]
        The cleaned dataframe and a dictionary of fitted encoders keyed by column.
    """

    df_clean = df.copy()
    min_non_null = len(df_clean) * 0.5
    df_clean = df_clean.dropna(axis=1, thresh=min_non_null)

    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

    for column in numeric_columns:
        df_clean[column] = df_clean[column].fillna(df_clean[column].median())

    encoders: Dict[str, LabelEncoder] = {}
    for column in categorical_columns:
        df_clean[column] = df_clean[column].fillna("missing").astype(str)
        encoder = LabelEncoder()
        df_clean[column] = encoder.fit_transform(df_clean[column])
        encoders[column] = encoder

    return df_clean, encoders


def split(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split the dataset into train, validation, calibration, and test partitions.

    The split ratios are 60% train, 15% validation, 15% calibration, and 10% test.
    Splits are stratified on ``isFraud`` and use ``random_state=42``.

    Parameters
    ----------
    df:
        Cleaned dataframe containing the target column ``isFraud``.

    Returns
    -------
    tuple
        ``X_train, X_val, X_cal, X_test, y_train, y_val, y_cal, y_test``.
    """

    if "isFraud" not in df.columns:
        raise KeyError("The dataframe must contain an 'isFraud' column.")

    X = df.drop(columns="isFraud")
    y = df["isFraud"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )

    X_val, X_remaining, y_val, y_remaining = train_test_split(
        X_temp,
        y_temp,
        test_size=0.625,
        stratify=y_temp,
        random_state=42,
    )

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.4,
        stratify=y_remaining,
        random_state=42,
    )

    return X_train, X_val, X_cal, X_test, y_train, y_val, y_cal, y_test


def get_proxy_groups(X_test: pd.DataFrame) -> pd.DataFrame:
    """Create proxy demographic group columns for fairness analysis.

    Proxy groups are defined as:
    - ``card_type`` from ``card4``
    - ``amount_bracket`` from quartiles of ``TransactionAmt``
    - ``product_type`` from ``ProductCD``

    Parameters
    ----------
    X_test:
        Test feature dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe with columns ``card_type``, ``amount_bracket``, and
        ``product_type`` aligned to ``X_test``.
    """

    required_columns = ["card4", "TransactionAmt", "ProductCD"]
    missing_columns = [column for column in required_columns if column not in X_test.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"X_test is missing required columns: {missing}")

    amount_labels = ["low", "medium", "high", "very_high"]
    amount_bracket = pd.qcut(
        X_test["TransactionAmt"],
        q=4,
        labels=amount_labels,
        duplicates="drop",
    )

    if amount_bracket.isna().any():
        amount_bracket = amount_bracket.astype("object").fillna("low")

    proxy_groups = pd.DataFrame(
        {
            "card_type": X_test["card4"].astype(str),
            "amount_bracket": amount_bracket.astype(str),
            "product_type": X_test["ProductCD"].astype(str),
        },
        index=X_test.index,
    )

    return proxy_groups
