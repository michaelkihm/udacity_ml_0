""" Unit tests of churn library functions """
import logging

import pandas as pd
import pytest
from mock import ANY, call, patch

from churn_library import (
    CATEGORY_COLUMNS,
    encoder_helper,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_models,
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
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Test import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Test import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        err_msg = "Test import_data: File doesn't  have rows and columns"
        logging.error(err_msg)
        raise err


def arrange_testdata():
    """Load testdata so that it can be used in the following unit tests"""
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
    dataframe = pd.read_csv("./data/bank_data.csv")
    expected_save_calls = [
        call(fname="./images/eda/churn_histogram.png"),
        call(fname="./images/eda/customer_age_histogram.png"),
        call(fname="./images/eda/marital_status_histogram.png"),
        call(fname="./images/eda/total_transaction_distribution.png"),
        call(fname="./images/eda/heatmap.png"),
    ]
    try:
        perform_eda(dataframe)
        logging.info("Test perform_eda: SUCCESS")
        assert save_fig_mock.mock_calls == expected_save_calls
    except AssertionError as err:
        logging.error("Test perform_eda: Not all plots saved to disk %s", err)
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
        x_train, x_test, y_train, y_test = perform_feature_engineering(data)
        assert len(x_test) == len(y_test) and len(x_train) == len(y_train)
    except AssertionError as err:
        logging.error("Test perform_feature_engineering: Wrong dataset sizes")
        raise err

    # Test if X_test is 30% of whole dataset
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(data)
        assert len(x_test) == pytest.approx(len(data) * 0.3, 2)
    except AssertionError as err:
        logging.error(
            "Test perform_feature_engineering: Test set should be 30 percent of whole dataset"
        )
        raise err

    logging.info("Test perform_feature_engineering: SUCCESS")


@patch("churn_library.plt.savefig")
@patch("churn_library.joblib.dump")
def test_train_models(save_model_mock, save_fig_mock):
    """
    test train_models
    """
    # Arrange testdata and only use first 100 data samples to
    # reduce test time
    dataframe = arrange_testdata()
    dataframe = dataframe.head(100)
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe)

    # Act
    train_models(x_train, x_test, y_train, y_test)

    # Test if roc curve was saved
    try:
        expected_call = call(fname="./images/results/roc_curve.png")
        assert expected_call in save_fig_mock.call_args_list
    except AssertionError as err:
        logging.error("Test train_models: Did not save roc_curve")
        raise err

    # Test if feature importance plot was saved
    try:
        expected_call = call(fname="./images/results/feature_importance_plot.png")
        assert expected_call in save_fig_mock.call_args_list
    except AssertionError as err:
        logging.error("Test train_models: Did not save feature_importance_plot")
        raise err

    # Test if random forest result was saved
    try:
        expected_call = call(fname="./images/results/RandomForest_results.png")
        assert expected_call in save_fig_mock.call_args_list
    except AssertionError as err:
        logging.error("Test train_models: Did not save RandomForest_results.png")
        raise err

    # Test if logistic regression result was saved
    try:
        expected_call = call(fname="./images/results/LogisticRegression_results.png")
        assert expected_call in save_fig_mock.call_args_list
    except AssertionError as err:
        logging.error("Test train_models: Did not save LogisticRegression_results.png")
        raise err

    # Test if random forest model was saved
    try:
        expected_call = call(ANY, "./models/rfc_model.pkl")
        assert expected_call in save_model_mock.call_args_list
    except AssertionError as err:
        logging.error("Test train_models: Did not save random forest model")
        raise err

    # Test if regression model was saved
    try:
        expected_call = call(ANY, "./models/logistic_model.pkl")
        assert expected_call in save_model_mock.call_args_list
    except AssertionError as err:
        logging.error("Test train_models: Did not save logistic regression model")
        raise err

    logging.info("Test train_models: SUCCESS")


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
