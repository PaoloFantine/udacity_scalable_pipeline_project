Project building a full pipeline to deploy an app doing inference on the census data (https://archive.ics.uci.edu/ml/datasets/census+income)

# GitHub repository
The repository with the full code can be found at https://github.com/PaoloFantine/udacity_scalable_pipeline_project

# Environment Set up
The conda environment can be created and installed by running `conda env create -f environment.yml`

## Relevant folders

Code for training the model is in folder `starter`
Folder  `model` contains the trained model, label binarizer and OneHotEncoder needed for inference on new data as well as .csv files giving model performance for all data slices and a jupyter notebook called `Model_Card.ipynb` describing the model, its intended use, performance and faults

Unit and API tests are contained in folder `tests`

The main API file is in the root folder and called `main.py`

A sample request file called `sample_request.py` is also included

# API Creation

The API created for this project can is available at https://udacity-scalable-pl-deployment.onrender.com
It will stay open for the time necessary for evaluation
