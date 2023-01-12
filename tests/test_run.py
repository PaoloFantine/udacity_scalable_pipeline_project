# file to test helper functions in starter/ml/model.py

import pytest
import sys
import joblib

# setting path
sys.path.append('../starter/')
sys.path.append('../model/')

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def gbm_model():
    return joblib.load("model/gbm_model.pkl")

def test_model_output_classes(gbm_model):
    assert gbm_model._n_classes==2
    assert gbm_model.classes_[0]==0
    assert gbm_model.classes_[1]==1
    
def test_model_random_state(gbm_model):
    assert gbm_model.random_state==42

def test_model_metrics():
    pass

def test_inference_pipeline():
    pass