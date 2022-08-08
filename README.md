# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
ML pipeline to identify credit card customers that are most likely to churn. The completed project contains a Python package for a machine learning project that follows coding (PEP8) and engineering best practices. The package can also be run interactively or from the command-line interface (CLI). See section Running Files for more details.

The data was pulled from the [kaggle Credit Card Customers dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code). The implementation of the pipeline can be found in [churn_library](churn_library.py).

The worklfow of the pipeline can be broken into the following steps.
1. Import Data: Loads [bank_data.csv](data/bank_data.csv) as pandas dataframe.
2. Exploratory Data Analysis: Export feature distribution plots and analyse feature correlations by plotting a heatmap. The exported plots can be found in directory [eda](images/eda/).
3. Feature Engineering: Turn each categorical column into a new column with propotion of churn for each category. Split dataset into test and train datasets.
4. Train Models: Training logistic regression and random forest models. The trained models will be saved in [models directory](models/).
## Files and data description
### File structure
* images
    * eda:
        - churn_histogram.png
        - customer_age_histogram.png
        - heatmap.png
        - marital_status_histogram.png
        - total_transaction_distribution.png
    * results
        - feature_importance_plot.png
        - LogisticRegression_results.png
        - RandomForest_results.png
        - roc_curve.png
* logs
    - churn_library.log
* models
    - logistic_model.pkl
    - rfc_model.pkl
* churn_library.py
* churn_script_logging_and_tests.py
* churn_notebook.ipynb
* constants.py
* environment.yaml
* Guide.ipynb
* README.md
* requirements.txt

### Most important files and directories
- [images/eda](images/eda): directory with exported plots from Exploratory Data Analysis.
- [logs/churn_library.log](logs/churn_library.log): logs of churn_script_logging_and_tests run. Should only have SUCCESS logs.
- [churn_library.py](churn_library.py): includes all functions required to run Customer Churn pipeline.
- [churn_script_logging_and_tests.py](churn_script_logging_and_tests.py): includes all unit tests of ML pipeline.
- [churn_notebook.ipynb](churn_notebook.ipynb): origninal notebook which was refacotored in [churn_library](churn_library.py).
- [constants.py](constants.py): includes constants shared between library functions and library tests.
- [requirements.txt](requirements.txt): contains the libraries used in this project.
- [environment.yaml](environment.yaml): conda environment file. Used to create conda environment used in this project


## Running Files
1. Create conda environment from [env file](environment.yml)
```
conda env create -f environment.yml
conda activate udacity_project_one
```
2. Run unit tests to assert that pipeline can be executed
```
pytest churn_script_logging_and_tests.py 
```
3. Run ML pipeline
```
ipython churn_library.py
```

