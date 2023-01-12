from fastapi.testclient import TestClient
import json

from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the API for the census data modelling pipeline!"}

def test_post_status_code_0():
    data = {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 83311,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 13,
                "native_country": "United-States",
            }
    r = client.post("/data/", data=json.dumps(data))
    print(r.json())
    assert r.content==b'"<=$50k"'
    assert r.status_code==200

def test_post_status_code_1():
    data = {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 83311,
                "education": "Doctorate",
                "education_num": 16,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 1000,
                "capital_loss": 0,
                "hours_per_week": 60,
                "native_country": "United-States",
            }
    r = client.post("/data/", data=json.dumps(data))
    print(r.json())
    assert r.content==b'">$50k"'
    assert r.status_code==200