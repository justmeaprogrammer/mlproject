<h1 align="center">Student Exam Performance Indicator</h1>

<p align="center">
  <img alt="Student Exam Performance Indicator Banner" src="https://quickchart.io/chart?width=1200&height=360&backgroundColor=%230b1020&c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22Ingestion%22%2C%22Transform%22%2C%22Training%22%2C%22Artifacts%22%2C%22Predict%22%2C%22Flask%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Pipeline%20Strength%22%2C%22data%22%3A%5B48%2C66%2C82%2C91%2C96%2C100%5D%2C%22borderColor%22%3A%22%2322c55e%22%2C%22backgroundColor%22%3A%22rgba%2834%2C197%2C94%2C0.18%29%22%2C%22fill%22%3Atrue%2C%22tension%22%3A0.35%7D%2C%7B%22label%22%3A%22Modular%20Design%22%2C%22data%22%3A%5B42%2C58%2C76%2C86%2C92%2C98%5D%2C%22borderColor%22%3A%22%2338bdf8%22%2C%22backgroundColor%22%3A%22rgba%2856%2C189%2C248%2C0.14%29%22%2C%22fill%22%3Atrue%2C%22tension%22%3A0.35%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22legend%22%3A%7B%22labels%22%3A%7B%22color%22%3A%22%23ffffff%22%7D%7D%2C%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Student%20Exam%20Performance%20Indicator%22%2C%22color%22%3A%22%23ffffff%22%2C%22font%22%3A%7B%22size%22%3A24%2C%22weight%22%3A%22bold%22%7D%7D%2C%22subtitle%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22End-to-End%20ML%20Pipeline%20%7C%20Flask%20Integrated%20%7C%20Modular%20Codebase%22%2C%22color%22%3A%22%23cbd5e1%22%2C%22font%22%3A%7B%22size%22%3A14%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22ticks%22%3A%7B%22color%22%3A%22%23cbd5e1%22%7D%2C%22grid%22%3A%7B%22color%22%3A%22rgba%28255%2C255%2C255%2C0.08%29%22%7D%7D%2C%22y%22%3A%7B%22ticks%22%3A%7B%22color%22%3A%22%23cbd5e1%22%7D%2C%22grid%22%3A%7B%22color%22%3A%22rgba%28255%2C255%2C255%2C0.08%29%22%7D%2C%22beginAtZero%22%3Atrue%2C%22max%22%3A100%7D%7D%2C%22layout%22%3A%7B%22padding%22%3A16%7D%7D%7D" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-Web_Integrated-111827?style=for-the-badge&logo=flask&logoColor=white">
  <img alt="Scikit-Learn" src="https://img.shields.io/badge/Scikit--Learn-Pipeline-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-Model_Selection-EC6B23?style=for-the-badge">
  <img alt="CatBoost" src="https://img.shields.io/badge/CatBoost-Regression-FFCC00?style=for-the-badge&logoColor=black">
  <img alt="End To End" src="https://img.shields.io/badge/ML-End_to_End-0F766E?style=for-the-badge">
</p>

<p align="center">
  <strong>A polished end-to-end machine learning project that takes raw student data, trains a regression pipeline, serializes production-ready artifacts, and serves predictions through a Flask web application.</strong>
</p>

<p align="center">
  This repository is built to present more than just model training. It shows <strong>pipeline thinking</strong>, <strong>modular Python engineering</strong>, and <strong>web integration</strong> in one complete portfolio project.
</p>

<p align="center">
  <img alt="Pipeline First" src="https://img.shields.io/badge/Pipeline-First-0F766E?style=flat-square">
  <img alt="Modular Codebase" src="https://img.shields.io/badge/Modular-Codebase-1D4ED8?style=flat-square">
  <img alt="Artifacts Ready" src="https://img.shields.io/badge/Artifacts-Ready-F59E0B?style=flat-square">
  <img alt="Flask UI" src="https://img.shields.io/badge/Flask-Integrated-7C3AED?style=flat-square">
</p>

---

## Why This Project Stands Out

- It is an end-to-end ML system, not only a notebook experiment.
- It separates ingestion, preprocessing, training, inference, utilities, templates, logging, and exception handling into clean modules.
- It persists reusable artifacts like `model.pkl` and `preprocessor.pkl` for real inference workflows.
- It exposes the model through a Flask UI, making the project web-integrated and recruiter-friendly.
- It demonstrates how to move from dataset to prediction screen with a structure that is easier to scale and maintain.

---

## Visual Architecture

```text
Raw Dataset
    |
    v
Data Ingestion
    |
    v
Train / Test Split
    |
    v
Data Transformation
    |
    v
Model Training + Selection
    |
    v
Saved Artifacts
    |
    v
Prediction Pipeline
    |
    v
Flask Web App
```

### End-to-End Flow

1. Raw student data is loaded from `src/notebook/data/stud.csv`.
2. The ingestion component creates `raw.csv`, `train.csv`, and `test.csv` inside `artifacts/`.
3. The transformation component builds preprocessing pipelines for numerical and categorical features.
4. The trainer evaluates multiple regression models with hyperparameter search.
5. The best model and fitted preprocessor are serialized for reuse.
6. The prediction pipeline loads those artifacts during inference.
7. Flask collects user input from the web form and returns a predicted math score.

---

## Pipeline Structure

This project is strongest when viewed as a layered ML pipeline:

| Pipeline Stage | What It Does | Code Area |
|---|---|---|
| Data Ingestion | Reads the dataset, stores the raw copy, performs train-test split | `src/components/data_ingestion.py` |
| Data Transformation | Imputes missing values, scales numerical features, encodes categorical features | `src/components/data_transformation.py` |
| Model Training | Compares multiple regressors with `GridSearchCV` and selects the best model | `src/components/model_trainer.py` |
| Artifact Persistence | Saves trained objects with `dill` for downstream inference | `src/utils.py` |
| Prediction Pipeline | Loads artifacts, transforms incoming data, predicts output | `src/pipeline/predict_pipeline.py` |
| Web Integration | Accepts form input and serves predictions through Flask templates | `app.py`, `templates/` |

### Preprocessing Design

Numerical features:

- `reading_score`
- `writing_score`

Categorical features:

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`

Transformations applied:

- `SimpleImputer(strategy="median")` for numerical columns
- `StandardScaler()` for numerical normalization
- `SimpleImputer(strategy="most_frequent")` for categorical columns
- `OneHotEncoder()` for categorical encoding

### Model Selection Strategy

The trainer compares multiple regression models, including:

- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Linear Regression
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

This makes the project stronger than a single-model demo because it shows a real evaluation mindset rather than hardcoding one algorithm from the start.

---

## Modular Coding Style

One of the best parts of this codebase is the separation of concerns:

| Module | Responsibility |
|---|---|
| `src/components/` | Core ML building blocks for ingestion, transformation, and training |
| `src/pipeline/` | Inference-facing workflow for prediction |
| `src/utils.py` | Shared serialization and model evaluation helpers |
| `src/logger.py` | Centralized logging setup |
| `src/exception.py` | Custom exception formatting for easier debugging |
| `templates/` | Flask frontend pages for landing and prediction form |
| `artifacts/` | Saved datasets and trained pipeline objects |

This modular structure signals good engineering habits:

- easier maintenance
- cleaner debugging
- better reusability
- clearer ownership of logic
- smoother transition from experimentation to application

For hiring managers and technical reviewers, that matters because it shows you can organize ML code like a real software project instead of a single long script.

---

## Web-Integrated Machine Learning

This is not just a backend training workflow. The project is connected to a Flask application so the user can interact with the model through a browser.

### Flask Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | `GET` | Landing page |
| `/predictdata` | `GET`, `POST` | Input form and prediction result |

### What The Web Layer Adds

- real user input collection
- immediate prediction response
- a clean demonstration of model serving
- an application-style portfolio presentation instead of notebook-only output

That is what makes this a true end-to-end project: the pipeline is trained in Python and consumed in a working web interface.

---

## Project Structure

```text
mlproject/
|-- app.py
|-- application.py
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
    |   `-- predict_pipeline.py
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

## Recruiter-Facing Highlights

- Built as a complete ML application from raw data to browser-based prediction.
- Uses modular Python design instead of monolithic scripts.
- Demonstrates artifact management for reusable inference.
- Includes model comparison and tuning rather than relying on a single baseline.
- Shows awareness of engineering concerns like logging, custom exceptions, and project structure.
- Bridges data science and software development through Flask integration.

If you want a project that communicates, "I can build machine learning systems that users can actually interact with," this is the right kind of portfolio piece.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, CatBoost |
| Serialization | Dill |
| Experimentation | Jupyter Notebook |

---

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd mlproject
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

macOS / Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the pipeline

```bash
python src/components/data_ingestion.py
```

This runs ingestion, preprocessing, and model training, then saves the artifacts required for inference.

### 5. Launch the Flask app

```bash
python app.py
```

Open the app in your browser and use the prediction form to get a math score estimate.

---

## Final Takeaway

This project is a strong showcase of:

- end-to-end machine learning execution
- clean pipeline architecture
- modular coding practices
- model training plus inference readiness
- Flask-based web integration

It presents you as someone who can build not only models, but complete ML applications.
