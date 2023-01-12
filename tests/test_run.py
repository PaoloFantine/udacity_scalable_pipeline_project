# file to test helper functions in starter/ml/model.py

import pytest
import sys
import joblib

from numpy import loadtxt

# setting path
sys.path.append('../starter/')
sys.path.append('../model/')
sys.path.append('../data/')

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def gbm_model():
    return joblib.load("model/gbm_model.pkl")

@pytest.fixture
def test_data():
    return loadtxt('data/X_test.csv', delimiter=',')

# testing ml.model.train_model function
def test_model_output_classes(gbm_model):
    """
    test the correct classes are modelled
    """
    assert gbm_model._n_classes==2
    assert gbm_model.classes_[0]==0
    assert gbm_model.classes_[1]==1
    
def test_model_random_state(gbm_model):
    """
    test the set random state is stored in the model
    """
    assert gbm_model.random_state==42
    
# testing ml.model.inference
@pytest.fixture
def preds(gbm_model, test_data):
    return inference(gbm_model, test_data)

def test_inference_classes(preds):
    """
    test all predictions are 0,1 as should be for classification
    """
    assert all(preds) in [0,1]
    
def test_preds_data_size(preds, test_data):
    """
    test preds is as long as there are rows in the data used for inference
    """
    assert len(preds)==test_data.shape[0]

# testing ml.model.model.metrics
@pytest.fixture
def y_test():
    return loadtxt('data/y_test.csv', delimiter=',')

@pytest.fixture
def model_metrics_list(y_test, preds):
    return compute_model_metrics(y_test, preds).values()

def test_model_metrics(model_metrics_list):
    '''
    test that 3 model metrics are computed and they fall between the expected ranges
    '''
    assert all(model_metrics_list)>=0
    assert all(model_metrics_list)<=1
    assert len(model_metrics_list)==3
    
def test_precision_calculation(preds, y_test, model_metrics_list):
    '''
    test calculation of precision
    '''
    precision, recall, fbeta = model_metrics_list
    
    tpositive_idx = y_test==1
    negative_idx = y_test==0
    
    tp = sum(preds[tpositive_idx]==y_test[tpositive_idx])
    fp = sum(preds[negative_idx]!=y_test[negative_idx])
    
    assert precision==tp/(tp+fp)
    
def test_recall_calculation(preds, y_test, model_metrics_list):
    '''
    test calculation of recall
    '''
    precision, recall, fbeta = model_metrics_list
    
    tpositive_idx = y_test==1
    
    tp = sum(preds[tpositive_idx]==y_test[tpositive_idx])
    fn = sum(preds[tpositive_idx]!=y_test[tpositive_idx])
    
    assert recall==tp/(tp+fn)

def test_f1_score_calculation(model_metrics_list):
    '''
    test that fbeta is computed correctly
    '''
    precision, recall, fbeta = model_metrics_list
    assert fbeta==2*precision*recall/(precision+recall)

