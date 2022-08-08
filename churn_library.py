"""
 Customer Churn library.
 Includes all function required for customer churn classification.
 Implements code from notebook churn_notebook in a more modular form.

 Author: Michael Kihm
 Creation Date: 8 August 2022
"""


import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import CATEGORY_COLUMNS, CHURN_COL_NAME, FIGURE_SIZE, QUANT_COLUMNS

sns.set()


os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    """
    return pd.read_csv(pth)


def perform_eda(dataframe):
    """
        perform eda on dataframe and save figures to images folder
        input:
                dataframe: pandas dataframe

        output:
                dataframe: processed dataframe with appended Churn column
        """
    required_cols = QUANT_COLUMNS + CATEGORY_COLUMNS
    assert len(set(required_cols).intersection(set(dataframe.columns))) == len(
        required_cols
    ), "Dataset misses one or more required columns"

    # Compute churn column
    dataframe[CHURN_COL_NAME] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Save churn histogram
    plt.figure(figsize=FIGURE_SIZE)
    dataframe[CHURN_COL_NAME].hist()
    plt.savefig(fname="./images/eda/churn_histogram.png")

    # Save Cutomer_Age histogram
    plt.figure(figsize=FIGURE_SIZE)
    dataframe["Customer_Age"].hist()
    plt.savefig(fname="./images/eda/customer_age_histogram.png")

    # Save Marital_Status histogram
    plt.figure(figsize=FIGURE_SIZE)
    dataframe.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig(fname="./images/eda/marital_status_histogram.png")

    # save histogram and density distribution of Total_Transaction
    plt.figure(figsize=FIGURE_SIZE)
    sns.histplot(dataframe["Total_Trans_Ct"], kde=True)
    plt.savefig(fname="./images/eda/total_transaction_distribution.png")

    # Save heatmap
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(dataframe.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(fname="./images/eda/heatmap.png")

    return dataframe


def encoder_helper(dataframe, category_lst, response="Churn"):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            encoded_dataframe: pandas dataframe with new columns for
            categorical columns from param category_lst
    """
    encoded_dataframe = dataframe.copy(deep=True)

    for category in category_lst:

        column_groups = dataframe.groupby(category).mean()[CHURN_COL_NAME]
        column_list = [column_groups.loc[val] for val in dataframe[category]]

        column_name = f"{category}_{response}" if response else category
        encoded_dataframe[column_name] = column_list

    return encoded_dataframe


def perform_feature_engineering(dataframe, response="Churn"):
    """
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    test_set_size = 0.3
    encoded_dataframe = encoder_helper(dataframe, CATEGORY_COLUMNS, response)

    # get target variable
    target_y = encoded_dataframe["Churn"]

    # create dataset X
    dataset = pd.DataFrame()
    keep_cols = [
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
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    dataset[keep_cols] = encoded_dataframe[keep_cols]

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, target_y, test_size=test_set_size, random_state=42
    )

    return x_train, x_test, y_train, y_test


def save_classification_report(
    y_train, y_test, y_train_preds, y_test_preds, clf_name,
):
    """Helper function to save classification report"""
    x_pos = 0.01
    fontdict = {"fontsize": 10, "fontproperties": "monospace"}
    test_report = str(classification_report(y_test, y_test_preds))
    train_report = str(classification_report(y_train, y_train_preds))

    # create plot
    plt.figure(figsize=(5, 5))
    plt.text(x_pos, 1.1, f"{clf_name} Train", fontdict)
    plt.text(x_pos, 0.05, train_report, fontdict)
    plt.text(x_pos, 0.6, f"{clf_name} Test", fontdict)
    plt.text(x_pos, 0.7, test_report, fontdict)
    plt.axis("off")

    # save plot
    plt.savefig(fname=f"./images/results/{clf_name}_results.png")


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
    # Random Forest report
    save_classification_report(
        y_train, y_test, y_train_preds_rf, y_test_preds_rf, "RandomForest",
    )

    # Logistic Regression report
    save_classification_report(
        y_train, y_test, y_train_preds_lr, y_test_preds_lr, "LogisticRegression",
    )


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.best_estimator_.feature_importances_

    # sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # create plot
    plt.figure(figsize=FIGURE_SIZE)
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # save plot
    plt.savefig(fname=os.path.join(output_pth, "feature_importance_plot.png"))


def train_models(x_train, x_test, y_train, y_test):
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
    # define RandomForest and LogisticRegression models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    # define grid search parameters
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    # grid search for training of RandomForest
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # validate models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # plot and save roc curves
    alpha = 0.8
    plt.figure(figsize=FIGURE_SIZE)
    axis = plt.gca()
    plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=alpha)
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=axis, alpha=alpha)
    plt.savefig(fname="./images/results/roc_curve.png")

    # save best models and
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")

    # save classification report and feature importance plots
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
    feature_importance_plot(cv_rfc, x_test, "./images/results")


if __name__ == "__main__":
    DATASET = import_data("./data/bank_data.csv")
    EDA_DF = perform_eda(DATASET)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(EDA_DF)
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
