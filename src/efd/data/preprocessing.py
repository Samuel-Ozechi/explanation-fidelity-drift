from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class FeaturePolicy:
    """
    Defines the Baseline 1 feature policy.
    """

    target: str = "isFraud"

    temporal: str = "TransactionDT"

    identifier: str = "TransactionID"

    max_missing_rate: float = 0.95


class NumericTransactionPreprocessor:
    """
    Leakage-safe numeric feature preprocessing.

    Baseline 1:
        - numeric transaction features only
        - exclude target
        - exclude temporal variable
        - exclude identifier
        - remove high-missingness columns
        - remove training-period constants
        - median imputation

    Every decision is fitted on the chronological training period.
    """

    def __init__(
        self,
        policy: FeaturePolicy | None = None,
    ):

        self.policy = (
            policy
            if policy is not None
            else FeaturePolicy()
        )

        self.features_: list[str] | None = None

        self.imputer_: SimpleImputer | None = None

        self.feature_missing_rates_: dict[str, float] = {}

        self.removed_high_missing_: list[str] = []

        self.removed_constant_: list[str] = []

    def fit(
        self,
        train: pd.DataFrame,
    ) -> "NumericTransactionPreprocessor":

        excluded = {
            self.policy.target,
            self.policy.temporal,
            self.policy.identifier,
        }

        # Only numeric columns.
        numeric = train.select_dtypes(
            include=["number"]
        )

        candidates = [
            column
            for column in numeric.columns
            if column not in excluded
        ]

        if not candidates:
            raise ValueError(
                "No numeric candidate features found."
            )

        # Step 1: Missingness filtering
        missing_rates = (
            train[candidates]
            .isna()
            .mean()
        )

        self.feature_missing_rates_ = (
            missing_rates
            .to_dict()
        )

        self.removed_high_missing_ = [
            column
            for column in candidates
            if missing_rates[column]
            > self.policy.max_missing_rate
        ]

        candidates = [
            column
            for column in candidates
            if column not in self.removed_high_missing_
        ]

        # Step 2: Remove training-period constants
        cardinalities = (
            train[candidates]
            .nunique(
                dropna=True
            )
        )

        self.removed_constant_ = [
            column
            for column in candidates
            if cardinalities[column] <= 1
        ]

        candidates = [
            column
            for column in candidates
            if column not in self.removed_constant_
        ]

        if not candidates:
            raise ValueError(
                "No usable features remain after filtering."
            )

        self.features_ = candidates


        # Step 3: Fit imputer (ONLY on training period)
        self.imputer_ = SimpleImputer(
            strategy="median"
        )

        self.imputer_.fit(
            train[self.features_]
        )

        return self

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.features_ is None:
            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        if self.imputer_ is None:
            raise RuntimeError(
                "Imputer has not been fitted."
            )

        # Explicitly recreate the frozen training schema.
        x = frame.reindex(
            columns=self.features_
        )

        transformed = self.imputer_.transform(
            x
        )

        return pd.DataFrame(
            transformed,
            columns=self.features_,
            index=frame.index,
        )

    def fit_transform(
        self,
        train: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(train)

        return self.transform(train)

    @property
    def feature_names_(self) -> list[str]:

        if self.features_ is None:
            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        return list(self.features_)