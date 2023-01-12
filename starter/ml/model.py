# Helper functions to train a model, compute model metrics and run inference
import random
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    
    return GradientBoostingClassifier(random_state=42).fit(X_train, y_train)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return {"precision":precision, "recall": recall, "fbeta":fbeta}


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : GradientBoostingClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def slice_metrics(test_data, preds, actual, slice_col):
    """ compute model metrics on slices of data

    Inputs
    ------
    test_data : pandas.DataFrame
        data for which the metrics are desired
    preds : np.array, pd.Series, list
        predictions for df as output by `inference`
    actual : np.array, pd.Series, list
        actual outcomes from the raw test_data
    slice_col : str
        columns to use for slicing
    Returns
    -------
    pandas.DataFrame
        dataframe where columns are the single model slices and rows,
        respectivels, precision, recall, fbeta scores for each slice
    """
    slices = test_data[slice_col].unique()
    
    slice_dict = {}
    for val in slices:
        
        idx = test_data[slice_col]==val
        slice_dict[val] = compute_model_metrics(preds[idx], actual[idx])
        
    return pd.DataFrame(slice_dict)


