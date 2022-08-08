import logging

import pandas as pd
import pytest
from mock import call, patch

from churn_library import CATEGORY_COLUMNS, encoder_helper, import_data, perform_eda

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
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        err_msg = "Test import_data: File doesn't  have rows and columns"
        logging.error(err_msg)
        raise err


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
        logging.info("Testing perform_eda: SUCCESS")
        assert save_fig_mock.mock_calls == expected_save_calls
    except AssertionError as err:
        logging.error(f"Test perform_eda: Not all plots saved to disk {err}")
        raise err


def test_encoder_helper():
    """
    test encoder helper
    """

    # Arrange testdata
    dataframe = pd.read_csv("./data/bank_data.csv")
    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    try:
        encoded_df = encoder_helper(dataframe, [])
        assert encoded_df.equals(dataframe)
        logging.info("Test encoder_helper: Not manipulates df if category_lst is empty")
    except AssertionError as err:
        logging.error(
            "Test encoder_helper: Should not manipulate df if category_lst is empty"
        )
        raise err

    try:
        encoded_df = encoder_helper(dataframe, CATEGORY_COLUMNS)
        expected_cols = [f"{cat}_Churn" for cat in CATEGORY_COLUMNS]
        common_cols = set(encoded_df.columns).intersection(set(expected_cols))
        assert len(common_cols) == len(expected_cols)
        logging.info(
            "Test encoder_helper: turns each categorical column into a new column"
        )
    except AssertionError as err:
        logging.error(
            "Test encoder_helper: Should turn each categorical column into a new column"
        )
        raise err

    logging.info("Testing encoder_helper: SUCCESS")


@pytest.mark.skip(reason="Not implemented")
def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """
    pass


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
