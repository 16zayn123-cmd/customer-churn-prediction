# Customer Churn Prediction — End-to-End Machine Learning Project

A portfolio-ready machine learning project that predicts telecom customer churn and turns the trained model into a usable application.

## What this project demonstrates

- Data cleaning
- Exploratory data analysis
- Train/test splitting
- Imputation
- One-hot encoding
- Feature scaling
- Logistic Regression
- Random Forest
- Model comparison
- ROC-AUC, precision, recall and F1 evaluation
- An sklearn `Pipeline`
- Saving a trained model with `joblib`
- A professional Streamlit interface
- Batch customer scoring
- A Flask prediction API
- GitHub-ready project structure
- Render deployment configuration

## Architecture

```text
Raw customer data
       |
       v
Cleaning / validation
       |
       v
sklearn preprocessing pipeline
  |                     |
  | numeric             | categorical
  v                     v
impute + scale      impute + one-hot encode
       \                /
        \              /
         v            v
        ML classifier
             |
             v
      churn probability
          /       \
         v         v
   Streamlit UI   Flask API
         |
         v
      deployment
```

## Project structure

```text
customer_churn_end_to_end/
|
├── app.py
├── api.py
├── requirements.txt
├── render.yaml
├── README.md
├── MODEL_CARD.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── notebooks/
│   └── 01_churn_analysis.ipynb
└── src/
    ├── __init__.py
    ├── download_data.py
    └── train.py
```

## 1. Install the project

Open a terminal in this folder.

### Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 2. Get the dataset

If you already have `WA_Fn-UseC_-Telco-Customer-Churn.csv`, place it inside the `data/` folder.

Otherwise run:

```bash
python src/download_data.py
```

The training script will also try to download it automatically if it is missing.

## 3. Train the model

```bash
python src/train.py
```

This will:

1. Load and clean the data.
2. Split it into training and test sets.
3. Build preprocessing pipelines.
4. Train Logistic Regression and Random Forest.
5. Evaluate both.
6. Select the model with the highest ROC-AUC.
7. Save the winning pipeline to `models/churn_pipeline.joblib`.
8. Save metrics and metadata.
9. Create a confusion-matrix image.

Generated files:

```text
models/churn_pipeline.joblib
models/metadata.json
reports/metrics.json
reports/confusion_matrix.png
```

## 4. Run the professional interface

```bash
streamlit run app.py
```

Streamlit will show a local address, normally:

```text
http://localhost:8501
```

The app contains three tabs:

### Single customer
Enter one customer's details and receive:

- churn probability
- low/medium/high risk band
- predicted class
- suggested retention action
- descriptive profile signals

### Batch scoring
Upload a CSV of customers and receive:

- churn probability for every customer
- predicted class
- risk band
- downloadable scored CSV

### Model performance
View:

- selected model
- ROC-AUC
- recall
- precision
- F1
- comparison between candidate models
- leading transformed features

## 5. Optional Flask API

The project also includes a simple API to demonstrate how another system could call the model.

Run:

```bash
python api.py
```

Health check:

```text
GET http://localhost:5000/health
```

Prediction endpoint:

```text
POST http://localhost:5000/predict
```

Example JSON body:

```json
{
  "SeniorCitizen": 0,
  "tenure": 5,
  "MonthlyCharges": 89.5,
  "TotalCharges": 447.5,
  "gender": "Male",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

Example response:

```json
{
  "churn_probability": 0.81,
  "predicted_churn": 1
}
```

The actual probability depends on the model trained on your data.

## 6. Put it on GitHub

Create a new GitHub repository called something like:

```text
customer-churn-prediction
```

Then from this project folder:

```bash
git init
git add .
git commit -m "Build end-to-end customer churn prediction project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

You can also use GitHub Desktop instead of these commands.

### What should be visible to an employer?

Your repository should make it easy to find:

- the business problem
- your notebook
- the preprocessing approach
- models compared
- evaluation metrics
- the Streamlit application
- the optional Flask API
- deployment instructions
- responsible-use limitations

## 7. Deploy on Render

This repository already contains `render.yaml`.

After pushing the project to GitHub:

1. Sign in to Render.
2. Create a new Blueprint or Web Service from your GitHub repository.
3. Render can read the included `render.yaml`.
4. During the build it will install the packages and run the training script.
5. It will then start the Streamlit app.

Equivalent settings are:

Build command:

```bash
pip install -r requirements.txt && python src/train.py
```

Start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Once deployed, you will have a public link that a recruiter can open.

## Portfolio explanation

A concise interview explanation:

> I built an end-to-end customer churn prediction system rather than stopping at a notebook. I cleaned the Telco churn data, created a reusable sklearn preprocessing and modelling pipeline, compared Logistic Regression with Random Forest using ROC-AUC and other classification metrics, and saved the selected model. I then exposed the model through a Streamlit application that supports both individual predictions and batch customer scoring. I also added a Flask API to show how the model could integrate with another application and prepared the repository for cloud deployment.

## Important

Do not present model probability as certainty. A churn score is a model estimate based on patterns in historical data. In a real organisation, model performance, fairness, drift, privacy and security would need to be monitored continuously.
