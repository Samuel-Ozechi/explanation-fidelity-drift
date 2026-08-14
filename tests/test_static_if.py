from efd.data.temporal import transaction_date


def test_transaction_date_returns_date_only():
    assert transaction_date(8_745_798, "2017-12-01") == "2018-03-12"
