import pandas as pd

from efd.data.temporal import chronological_windows


def test_chronological_windows():

    df = pd.DataFrame(
        {
            "TransactionDT": [
                50,
                10,
                40,
                20,
                30,
                60,
                70,
                80,
                90,
                100,
            ],
            "isFraud": [
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
            ],
        }
    )

    train, windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=0.6,
        n_future_windows=2,
    )

    assert (
        train["TransactionDT"]
        .tolist()
        == [10, 20, 30, 40, 50, 60]
    )

    assert (
        windows[0]["TransactionDT"]
        .tolist()
        == [70, 80]
    )

    assert (
        windows[1]["TransactionDT"]
        .tolist()
        == [90, 100]
    )