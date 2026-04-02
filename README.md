# Student Exam Performance Indicator

<p align="center">
  <strong>An end-to-end machine learning project that predicts a student's math score from demographic and academic inputs.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask&logoColor=white">
  <img alt="Scikit-Learn" src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-Boosting-EC6B23?style=for-the-badge">
  <img alt="CatBoost" src="https://img.shields.io/badge/CatBoost-Gradient_Boosting-FFCC00?style=for-the-badge">
</p>

---

## Overview

This project builds a complete machine learning workflow for **student performance prediction** and serves it through a simple **Flask web application**.

The model predicts a student's **math score** using:

- Gender
- Race / ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Reading score
- Writing score

The repository includes:

- data ingestion and train-test split
- preprocessing with categorical and numerical pipelines
- model training with multiple regression algorithms
- serialized model artifacts for inference
- a web UI for collecting inputs and showing predictions

---

## Demo Flow

1. Open the landing page.
2. Navigate to the prediction form.
3. Enter student details and academic scores.
4. Submit the form.
5. Receive a predicted math score instantly.

---

## Features

- End-to-end ML pipeline from raw CSV to deployed prediction flow
- Clean separation of ingestion, transformation, training, and prediction logic
- Multiple regression models compared during training
- Saved `preprocessor.pkl` and `model.pkl` artifacts for reuse
- Flask app with a styled landing page and prediction form
- Notebook assets for EDA and model experimentation

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Web App | Flask |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn, XGBoost, CatBoost |
| Serialization | Dill |

---

## Project Structure

```text
mlproject/
|-- app.py
|-- requirements.txt
|-- README.md
|-- artifacts/
|   |-- model.pkl
|   |-- preprocessor.pkl
|   |-- raw.csv
|   |-- train.csv
|   `-- test.csv
|-- templates/
|   |-- index.html
|   `-- home.html
`-- src/
    |-- components/
    |   |-- data_ingestion.py
    |   |-- data_transformation.py
    |   `-- model_trainer.py
    |-- pipeline/
    |   |-- predict_pipeline.py
    |   `-- train_pipeline.py
    |-- notebook/
    |   |-- EDA STUDENT PERFORMANCE.ipynb
    |   |-- MODEL TRAINING.ipynb
    |   `-- data/
    |       `-- stud.csv
    |-- exception.py
    |-- logger.py
    `-- utils.py
```

---

## Machine Learning Pipeline

### 1. Data Ingestion

The ingestion step:

- reads the dataset from `src/notebook/data/stud.csv`
- stores a raw copy in `artifacts/raw.csv`
- splits data into training and test sets
- writes them to `artifacts/train.csv` and `artifacts/test.csv`

### 2. Data Transformation

The preprocessing pipeline handles two feature groups:

- Numerical features:
  - `reading_score`
  - `writing_score`
- Categorical features:
  - `gender`
  - `race_ethnicity`
  - `parental_level_of_education`
  - `lunch`
  - `test_preparation_course`

Transformations used:

- `SimpleImputer(strategy="median")` for numerical data
- `StandardScaler()` for numerical scaling
- `SimpleImputer(strategy="most_frequent")` for categorical data
- `OneHotEncoder()` for categorical encoding

The fitted preprocessor is saved as:

```bash
artifacts/preprocessor.pkl
```

### 3. Model Training

The trainer evaluates several regression models, including:

- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Linear Regression
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

After evaluation, the best-performing model is saved as:

```bash
artifacts/model.pkl
```

### 4. Prediction Pipeline

The prediction pipeline:

- loads the trained model and preprocessor
- transforms incoming user input
- predicts the student's math score
- returns the result to the Flask template

---

## Web Application

The Flask app exposes two main routes:

| Route | Method | Purpose |
|---|---|---|
| `/` | `GET` | Landing page |
| `/predictdata` | `GET`, `POST` | Prediction form and result page |

The interface includes:

- a modern landing page
- an input form for student details
- instant score prediction after submission

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd mlproject
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

On macOS / Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How To Run

### Run the Flask app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

### Run the training pipeline

You can trigger training from:

```bash
python src/components/data_ingestion.py
```

This will:

- ingest the dataset
- create training and test splits
- build preprocessing artifacts
- train and save the best model

---

## Input Features Used For Prediction

| Feature | Type |
|---|---|
| gender | categorical |
| race_ethnicity | categorical |
| parental_level_of_education | categorical |
| lunch | categorical |
| test_preparation_course | categorical |
| reading_score | numerical |
| writing_score | numerical |

### Target Variable

```text
math_score
```

---

## Artifacts

The `artifacts/` directory stores generated outputs used during training and inference:

- `raw.csv` - raw copied dataset
- `train.csv` - training split
- `test.csv` - test split
- `preprocessor.pkl` - fitted preprocessing pipeline
- `model.pkl` - trained regression model

---

## Notebooks

The notebooks inside `src/notebook/` are useful for:

- exploratory data analysis
- feature understanding
- testing model ideas before moving them into the production pipeline

---

## Why This Project Stands Out

- It is not just a notebook experiment; it includes a usable web app.
- It separates ML stages into reusable Python modules.
- It demonstrates model comparison, serialization, and deployment basics in one project.
- It is a strong beginner-to-intermediate portfolio project for data science and ML engineering.

---

## Future Improvements

- add model performance metrics to the README
- containerize the project with Docker
- add automated tests for pipeline components
- deploy the Flask app to a cloud platform
- improve validation and error handling in the form
- add experiment tracking for model selection

---

## Author

Built as an end-to-end machine learning project for predicting student exam performance using Flask and regression models.
