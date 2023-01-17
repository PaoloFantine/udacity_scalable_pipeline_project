import requests
import json

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

welcome = requests.get("https://udacity-scalable-pl-deployment.onrender.com")#127.0.0.1:8000
r = requests.post("https://udacity-scalable-pl-deployment.onrender.com/data", data=json.dumps(data))

print(welcome.json())
print(r.json())
print("status_code:", r.status_code)