# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
ML pipeline to identify credit card customers that are most likely to churn. Uses [kaggle Credit Card Customers dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code) to train a [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). The implementation of the pipeline can be found in [churn_library](churn_library.py).

## Files and data description
- [churn_library](churn_library.py): includes all function required to run Customer Churn pipeline.
- [churn_script_logging_and_tests](churn_script_logging_and_tests.py): includes all unit tests of ML pipeline.
- [logs](logs/churn_library.log): logs of churn_script_logging_and_tests run. Should only have SUCCESS logs. 

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

