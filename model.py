import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def infer_problem_type(target: pd.Series) -> str:
    """Infer whether the task is classification or regression."""
    if target.dtype == object:
        return "classification"

    unique_values = target.dropna().unique()
    if np.issubdtype(target.dtype, np.integer) and len(unique_values) <= 20:
        return "classification"

    return "regression"


def build_pipeline(
    problem_type: str,
    numeric_features: pd.Index,
    categorical_features: pd.Index,
) -> Tuple[Pipeline, dict, str]:
    """Build the preprocessing + model pipeline and associated search grid."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )

    if problem_type == "classification":
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 16, 32],
            "model__min_samples_split": [2, 4],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2", 0.7],
            "model__class_weight": [None, "balanced_subsample"],
        }
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 16, 32],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.7],
        }
        scoring = "neg_root_mean_squared_error"

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline, param_grid, scoring


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight derived features that help tree-based models."""

    result = df.copy()

    if "funding_total_usd" in result.columns:
        result["funding_total_log"] = np.log1p(result["funding_total_usd"])

    if {"funding_total_usd", "funding_rounds"}.issubset(result.columns):
        result["funding_per_round"] = result["funding_total_usd"] / (
            result["funding_rounds"].replace(0, np.nan)
        )

    if {"age_last_funding_year", "age_first_funding_year"}.issubset(result.columns):
        result["funding_age_span"] = (
            result["age_last_funding_year"] - result["age_first_funding_year"]
        )

    if {"age_last_milestone_year", "age_first_milestone_year"}.issubset(result.columns):
        result["milestone_age_span"] = (
            result["age_last_milestone_year"] - result["age_first_milestone_year"]
        )

    if {"milestones", "funding_rounds"}.issubset(result.columns):
        result["milestones_per_round"] = result["milestones"] / (
            result["funding_rounds"].replace(0, np.nan)
        )

    return result


def train_and_predict(train_path: pathlib.Path, test_path: pathlib.Path, output_path: pathlib.Path) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "labels" not in train_df.columns:
        raise ValueError("Training data must contain a 'labels' column as the target.")

    target = train_df["labels"]
    features = train_df.drop(columns=["labels"])

    test_ids = test_df["id"]

    # Keep the ID column only for submission
    if "id" in features.columns:
        features = features.drop(columns=["id"])
    if "id" in test_df.columns:
        test_features = test_df.drop(columns=["id"])
    else:
        test_features = test_df.copy()

    features = engineer_features(features)
    test_features = engineer_features(test_features)

    numeric_features = features.select_dtypes(include=["number", "bool"]).columns
    categorical_features = features.select_dtypes(include=["object", "category"]).columns

    problem_type = infer_problem_type(target)

    pipeline, param_grid, scoring = build_pipeline(
        problem_type=problem_type,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    search.fit(features, target)

    print("Best params:", search.best_params_)
    print("Best CV score:", search.best_score_)

    best_pipeline = search.best_estimator_
    predictions = best_pipeline.predict(test_features)

    submission = pd.DataFrame({"id": test_ids, "labels": predictions})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


if __name__ == "__main__":
    project_root = pathlib.Path(__file__).resolve().parent
    train_csv = project_root / "train.csv"
    test_csv = project_root / "test.csv"
    submission_csv = project_root / "submission.csv"

    train_and_predict(train_csv, test_csv, submission_csv)
