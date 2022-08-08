import logging

import pandas as pd
import pytest
from mock import call, patch

from churn_library import (
    CATEGORY_COLUMNS,
    encoder_helper,
    import_data,
    perform_eda,
    perform_feature_engineering,
)

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    test data import - this example is completed for you to
    assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Test import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Test import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        err_msg = "Test import_data: File doesn't  have rows and columns"
        logging.error(err_msg)
        raise err


def arrange_testdata():
    dataframe = pd.read_csv("./data/bank_data.csv")
    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return dataframe


@patch("churn_library.plt.savefig")
def test_eda(save_fig_mock):
    """
    test perform eda function
    """
    df = pd.read_csv("./data/bank_data.csv")
    expected_save_calls = [
        call(fname="./images/eda/churn_histogram.png"),
        call(fname="./images/eda/customer_age_histogram.png"),
        call(fname="./images/eda/marital_status_histogram.png"),
        call(fname="./images/eda/total_transaction_distribution.png"),
        call(fname="./images/eda/heatmap.png"),
    ]
    try:
        perform_eda(df)
        logging.info("Test perform_eda: SUCCESS")
        assert save_fig_mock.mock_calls == expected_save_calls
    except AssertionError as err:
        logging.error(f"Test perform_eda: Not all plots saved to disk {err}")
        raise err


def test_encoder_helper():
    """
    test encoder helper
    """

    # Arrange testdata
    dataframe = arrange_testdata()

    # Test that encoder does not manipulate df if category_lst is empty
    try:
        encoded_df = encoder_helper(dataframe, [])
        assert encoded_df.equals(dataframe)
    except AssertionError as err:
        logging.error(
            "Test encoder_helper: Should not manipulate df if category_lst is empty"
        )
        raise err

    # Test if encoder turns each categorical column into a new column
    try:
        encoded_df = encoder_helper(dataframe, CATEGORY_COLUMNS)
        expected_cols = [f"{cat}_Churn" for cat in CATEGORY_COLUMNS]
        common_cols = set(encoded_df.columns).intersection(set(expected_cols))
        assert len(common_cols) == len(expected_cols)
    except AssertionError as err:
        logging.error(
            "Test encoder_helper: Should turn each categorical column into a new column"
        )
        raise err

    logging.info("Test encoder_helper: SUCCESS")


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """

    data = arrange_testdata()

    # Test if datasets X are corresponding to targets y
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(data)
        assert len(X_test) == len(y_test) and len(X_train) == len(y_train)
    except AssertionError as err:
        logging.error("Test perform_feature_engineering: Wrong dataset sizes")
        raise err

    # Test if X_test is 30% of whole dataset
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(data)
        assert len(X_test) == pytest.approx(len(data) * 0.3, 2)
    except AssertionError as err:
        logging.error(
            "Test perform_feature_engineering: Test set should be 30 percent of whole dataset"
        )
        raise err

    logging.info("Test perform_feature_engineering: SUCCESS")


@pytest.mark.skip(reason="Not implemented")
def test_train_models():
    """
    test train_models
    """
    pass


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
