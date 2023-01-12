# code for the API
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union

import pandas as pd
import numpy as np
import joblib

from starter.ml.model import inference
from starter.ml.data import process_data

# Instantiate the app.
app = FastAPI(
    title="Exercise API",
    description="An API that demonstrates checking the values of your inputs.",
    version="1.0.0",
    )

class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        
        schema_extra = {
        "example": {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 83311,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 13,
                "native-country": "United-States",
            }
        }

@app.post("/data/")
async def model_inference(data: Data):

    
    df = pd.DataFrame.from_dict(data.dict(), orient='index').transpose(copy=True)

    model = joblib.load("model/gbm_model.pkl")
    encoder = joblib.load("model/OneHotEncoder.pkl")
    lb = joblib.load("model/LabelBinarizer.pkl")

    cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    ]

    X_categorical = df[cat_features].values
    X_continuous = df.drop(*[cat_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    prediction = ">$50k" if inference(model, X)==1 else "<=$50k"
    
    return prediction

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the API for the census data modelling pipeline!"}