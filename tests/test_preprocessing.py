import pandas as pd

from efd.data.preprocessing import (
    NumericTransactionPreprocessor,
)


def test_preprocessing_uses_training_schema_only():

    train = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [1, 2, 3, 4],
            "isFraud": [0, 0, 1, 0],

            "feature_a": [
                1.0,
                None,
                3.0,
                4.0,
            ],

            "constant": [
                1,
                1,
                1,
                1,
            ],

            "text_feature": [
                "a",
                "b",
                "a",
                "b",
            ],
        }
    )

    future = pd.DataFrame(
        {
            "TransactionID": [5],
            "TransactionDT": [5],
            "isFraud": [1],

            "feature_a": [None],

            "constant": [1],

            "text_feature": ["future"],
        }
    )

    processor = (
        NumericTransactionPreprocessor()
    )

    x_train = (
        processor
        .fit_transform(train)
    )

    x_future = (
        processor
        .transform(future)
    )

    assert (
        processor.feature_names_
        == ["feature_a"]
    )

    assert x_train.shape == (4, 1)

    assert x_future.shape == (1, 1)

    # Future missing value must use
    # the training median.
    assert (
        x_future.iloc[0, 0]
        == x_train["feature_a"].median()
    )