# library doc string


import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
CATEGORY_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

QUANT_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

FIGURE_SIZE = (20, 10)
CHURN_COL_NAME = "Churn"

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    return pd.read_csv(pth)


def perform_eda(df):
    """
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                df: processed dataframe with appended Churn column
        """
    required_cols = QUANT_COLUMNS + CATEGORY_COLUMNS
    assert len(set(required_cols).intersection(set(df.columns))) == len(
        required_cols
    ), "Dataset misses one or more required columns"

    # Compute churn column
    df[CHURN_COL_NAME] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Save churn histogram
    plt.figure(figsize=FIGURE_SIZE)
    df[CHURN_COL_NAME].hist()
    plt.savefig(fname="./images/eda/churn_histogram.png")

    # Save Cutomer_Age histogram
    plt.figure(figsize=FIGURE_SIZE)
    df["Customer_Age"].hist()
    plt.savefig(fname="./images/eda/customer_age_histogram.png")

    # Save Marital_Status histogram
    plt.figure(figsize=FIGURE_SIZE)
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig(fname="./images/eda/marital_status_histogram.png")

    # save histogram and density distribution of Total_Transaction
    plt.figure(figsize=FIGURE_SIZE)
    sns.histplot(df["Total_Trans_Ct"], kde=True)
    plt.savefig(fname="./images/eda/total_transaction_distribution.png")

    # Save heatmap
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(fname="./images/eda/heatmap.png")

    return df


def encoder_helper(df, category_lst, response="Churn"):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            encoded_df: pandas dataframe with new columns for
            categorical columns from param category_lst
    """
    encoded_df = df.copy(deep=True)

    for category in category_lst:

        column_groups = df.groupby(category).mean()[CHURN_COL_NAME]
        column_list = [column_groups.loc[val] for val in df[category]]

        column_name = f"{category}_{response}" if response else category
        encoded_df[column_name] = column_list

    return encoded_df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results
    and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass


if __name__ == "__main__":
    DATASET = import_data("./data/bank_data.csv")
    EDA_DF = perform_eda(DATASET)
