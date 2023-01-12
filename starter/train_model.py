# Script to train machine learning model.

import pandas as pd
import joblib
import sys
from numpy import savetxt

from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics, inference, slice_metrics
from ml.data import process_data


# Add the necessary imports for the starter code.

# Add code to load in the data.
sys.path.append('../data')
df = pd.read_csv("data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, stratify=df['salary'], test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# save test data to csv file for later
savetxt('data/X_test.csv', X_test, delimiter=',')
savetxt('data/y_test.csv', y_test, delimiter=',')
joblib.dump(encoder, "model/OneHotEncoder.pkl")
joblib.dump(lb, "model/LabelBinarizer.pkl")


# Train and save a model.
gbm_model = train_model(X_train, y_train)

sys.path.append('../model')
filename = 'gbm_model.pkl'
joblib.dump(gbm_model, "model/"+filename)

# test model performance
preds = inference(gbm_model, X_test)

# compute global and slice model metrics
model_metrics = pd.DataFrame(compute_model_metrics(y_test, preds), index=[0])
model_metrics.to_csv('model/model_metrics.csv')

slice_metrics = {col: slice_metrics(test, preds, y_test, col) for col in cat_features}

[slice_df.to_csv(f"model/{key}_sliced_metrics.csv") for key, slice_df in slice_metrics.items()]




