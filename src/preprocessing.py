"""
src/preprocessing.py

Data preprocessing pipeline for the Stroke Prediction dataset.
Refactored from the Phase 1 notebook into reusable functions.

Typical usage:
    from src.preprocessing import load_data, prepare_pipeline

    X_train, X_test, y_train, y_test = prepare_pipeline("data/healthcare-dataset-stroke-data.csv")
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


CSV_NAME = "healthcare-dataset-stroke-data.csv"


def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load the stroke dataset from a CSV file.

    If csv_path is not provided, looks for the CSV in the data/ directory
    relative to the project root.

    Returns a raw DataFrame (no transformations applied).
    """
    if csv_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "data", CSV_NAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Run: python data/download_data.py"
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning steps:
    - Drop the 'id' column (not predictive)
    - Remove rows where gender == 'Other' (very few, unstable class)

    Returns a cleaned DataFrame.
    """
    df = df.copy()
    df = df.drop(columns=["id"], errors="ignore")
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply One-Hot Encoding to categorical columns using pd.get_dummies.
    drop_first=True to avoid multicollinearity.

    Returns a fully numeric DataFrame.
    """
    df = pd.get_dummies(df, drop_first=True)
    return df


def split_and_balance(
    df: pd.DataFrame,
    target_col: str = "stroke",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split into train/test (stratified) and apply SMOTE to the training set.

    Steps:
    - Fill NaN in 'bmi' with the training set mean (before SMOTE)
    - Stratified 80/20 split
    - SMOTE applied only on training data (no data leakage)

    Returns:
        X_train_res, X_test, y_train_res, y_test
        (X_train_res and y_train_res are the SMOTE-resampled training data)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fill NaN in bmi using training mean only (avoid leakage)
    bmi_mean = X_train["bmi"].mean()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_mean)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_mean)

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(
        f"Train size (after SMOTE): {X_train_res.shape[0]} "
        f"(positive: {y_train_res.sum()}, negative: {(y_train_res == 0).sum()})"
    )
    print(f"Test size: {X_test.shape[0]} (positive: {y_test.sum()})")

    return X_train_res, X_test, y_train_res, y_test


def prepare_pipeline(csv_path: str = None):
    """
    Convenience function: runs the full preprocessing pipeline end-to-end.

    Equivalent to:
        df = load_data(csv_path)
        df = clean_data(df)
        df = encode_features(df)
        return split_and_balance(df)

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_data(csv_path)
    df = clean_data(df)
    df = encode_features(df)
    return split_and_balance(df)
