"""Unit tests for the data_engineering nodes, using synthetic DataFrames."""

import numpy as np
import pandas as pd
import pytest

from churn.pipelines.data_engineering.nodes import (
    add_split_column,
    clean_data,
    fit_encoders,
    fit_scalers,
    transform_encoders,
    transform_scalers,
)

COLUMNS = {
    "target": "Churn",
    "numerical": ["tenure", "TotalCharges"],
    "categorical": ["Contract", "gender"],
}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["a", "b", "c", "d"],
            "tenure": [1, 24, 0, 12],
            "TotalCharges": ["100.5", "2000", " ", "500"],
            "Contract": ["Month-to-month", "Two year", " One year", "Two year"],
            "gender": ["Male", "Female", "Male", "Female"],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )


class TestCleanData:
    def test_keeps_only_configured_columns(self, raw_df):
        out = clean_data(raw_df, COLUMNS)

        assert "customerID" not in out.columns
        assert set(out.columns) == {"Churn", "tenure", "TotalCharges", "Contract", "gender"}

    def test_blank_total_charges_becomes_zero(self, raw_df):
        """The notebook's classic gotcha: TotalCharges holds ' ' for tenure == 0."""
        out = clean_data(raw_df, COLUMNS)

        assert out["TotalCharges"].iloc[2] == 0.0
        assert out["TotalCharges"].dtype.kind == "f"

    def test_categorical_values_are_stripped(self, raw_df):
        out = clean_data(raw_df, COLUMNS)

        assert out["Contract"].iloc[2] == "One year"

    def test_missing_target_is_tolerated(self, raw_df):
        """Inference data has no Churn column; the same node must still work."""
        out = clean_data(raw_df.drop(columns=["Churn"]), COLUMNS)

        assert "Churn" not in out.columns
        assert len(out) == len(raw_df)

    def test_input_is_not_mutated(self, raw_df):
        before = raw_df.copy()
        clean_data(raw_df, COLUMNS)

        pd.testing.assert_frame_equal(raw_df, before)


class TestAddSplitColumn:
    def test_proportions_must_sum_to_one(self):
        df = pd.DataFrame({"x": range(10)})

        with pytest.raises(ValueError, match="must sum to 1.0"):
            add_split_column(df, {"train": 0.7, "test": 0.2, "validate": 0.2, "random_state": 42})

    def test_same_seed_is_reproducible(self):
        df = pd.DataFrame({"x": range(500)})
        split = {"train": 0.7, "test": 0.15, "validate": 0.15, "random_state": 42}

        first = add_split_column(df, split)
        second = add_split_column(df, split)

        pd.testing.assert_series_equal(first["split"], second["split"])

    def test_labels_are_only_the_three_expected(self):
        df = pd.DataFrame({"x": range(500)})
        out = add_split_column(
            df, {"train": 0.7, "test": 0.15, "validate": 0.15, "random_state": 42}
        )

        assert set(out["split"].unique()) <= {"train", "test", "validate"}


@pytest.fixture
def split_df() -> pd.DataFrame:
    """A frame where 'Fiber' appears only in the test split — the leakage probe."""
    return pd.DataFrame(
        {
            "tenure": [1.0, 2.0, 3.0, 40.0],
            "TotalCharges": [10.0, 20.0, 30.0, 400.0],
            "Contract": ["Month-to-month", "Two year", "Month-to-month", "One year"],
            "gender": ["Male", "Female", "Male", "Female"],
            "Churn": ["No", "Yes", "No", "Yes"],
            "split": ["train", "train", "train", "test"],
        }
    )


class TestFitTransformSeparation:
    def test_encoders_ignore_categories_outside_the_fitting_split(self, split_df):
        """'One year' exists only in the test split, so the encoder must not know it."""
        encoders = fit_encoders(split_df, COLUMNS, {"columns": ["target", "categorical"], "split_to_fit": ["train"]})

        assert "One year" not in set(encoders["Contract"].classes_)

    def test_widening_split_to_fit_includes_them(self, split_df):
        encoders = fit_encoders(
            split_df, COLUMNS, {"columns": ["target", "categorical"], "split_to_fit": ["train", "test"]}
        )

        assert "One year" in set(encoders["Contract"].classes_)

    def test_scaler_statistics_come_from_the_train_split_only(self, split_df):
        scalers = fit_scalers(split_df, COLUMNS, {"columns": ["numerical"], "split_to_fit": ["train"]})

        # tenure on train is [1, 2, 3] -> mean 2.0; the 40.0 in test must not shift it.
        assert scalers["tenure"].mean_[0] == pytest.approx(2.0)

    def test_transform_skips_columns_that_are_absent(self, split_df):
        """This guard is what lets inference reuse encoders fitted with a target."""
        encoders = fit_encoders(
            split_df, COLUMNS, {"columns": ["target", "categorical"], "split_to_fit": ["train", "test"]}
        )
        inference_like = split_df.drop(columns=["Churn", "split"])

        out = transform_encoders(inference_like, encoders)

        assert "Churn" not in out.columns
        assert out["Contract"].dtype.kind in "iu"

    def test_scaled_train_values_are_standardized(self, split_df):
        scalers = fit_scalers(split_df, COLUMNS, {"columns": ["numerical"], "split_to_fit": ["train"]})

        out = transform_scalers(split_df, scalers)

        train_tenure = out.loc[out["split"] == "train", "tenure"]
        assert train_tenure.mean() == pytest.approx(0.0, abs=1e-9)
        assert np.std(train_tenure) == pytest.approx(1.0)
