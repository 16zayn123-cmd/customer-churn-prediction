# Customer Churn Prediction — End-to-End Machine Learning Project

## Project overview

I built this project to take a machine-learning problem beyond a Jupyter notebook and turn it into a complete, usable application.

The goal is to predict whether a telecom customer is likely to churn using information such as their tenure, contract type, internet service, monthly charges, payment method and support services.

The project covers the full workflow from **data exploration and preprocessing through model training, evaluation, model persistence, a Streamlit interface and an optional Flask API**.

> **Business question:** Can customer account and service information be used to identify customers who are at higher risk of leaving, so that a retention team can intervene earlier?

---

## Application

The trained model is connected to a Streamlit application where a user can enter customer information and receive a churn probability and risk classification.

![Customer Churn Streamlit application](docs/streamlit-app.png)

The application contains three sections:

- **Single customer** — enter one customer's details and calculate their churn risk.
- **Batch scoring** — upload a CSV and score multiple customers at once.
- **Model performance** — review the selected model and its evaluation metrics.

---

## End-to-end workflow

```text
Raw Telco customer data
        |
        v
Jupyter notebook
EDA + data understanding
        |
        v
Final preprocessing decisions
        |
        v
src/train.py
        |
        +--> Missing-value handling
        +--> One-hot encoding
        +--> Feature scaling
        +--> Logistic Regression
        +--> Random Forest
        |
        v
Model evaluation
Accuracy / Precision / Recall / F1 / ROC-AUC
        |
        v
Best model selected
        |
        v
models/churn_pipeline.joblib
        |
        +-------------------+
        |                   |
        v                   v
Streamlit app           Flask API
Human-facing            Software-facing
interface               prediction endpoint
```

A key design choice was to save the **entire scikit-learn pipeline**, rather than saving only the classifier. This means new raw customer data goes through the same preprocessing used during training before a prediction is made.

---

## What I did

### 1. Explored the data in Jupyter

I used the notebook to understand the dataset before building the final training script. This included:

- inspecting the columns and data types;
- checking missing values and duplicates;
- converting `TotalCharges` from text to numeric data;
- examining the churn class distribution;
- investigating churn against contract type, tenure, internet service and monthly charges;
- deciding which variables required numerical or categorical preprocessing.

The notebook is intentionally kept separate from the final production-style training code. I used it as the experimentation and analysis environment, while `train.py` contains the repeatable final process.

### 2. Built a reusable preprocessing pipeline

The dataset contains both numerical and categorical variables, so I used a `ColumnTransformer` to apply different preprocessing steps to each group.

**Numerical features**

- Median imputation for missing values
- Standard scaling

**Categorical features**

- Most-frequent-value imputation
- One-hot encoding
- `handle_unknown="ignore"` so unseen categories do not cause prediction failures

This preprocessing is contained inside the same scikit-learn `Pipeline` as the classifier.

### 3. Compared classification models

The training script compares:

- Logistic Regression
- Random Forest

Both models use class balancing because churn is not evenly distributed between the two target classes.

The script evaluates each model using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

I use ROC-AUC as the model-selection metric because the application is primarily interested in ranking customers by churn risk and producing a probability score rather than only returning a yes/no label.

### 4. Saved the trained pipeline

The selected model is saved using `joblib`:

```text
models/churn_pipeline.joblib
```

This saved file contains both the preprocessing steps and trained classifier. The Streamlit application can therefore load one object and make predictions from raw customer inputs.

### 5. Built a user-facing Streamlit application

The Streamlit application converts the machine-learning model into something a non-technical user can interact with.

For an individual customer, the app:

1. collects the customer's information;
2. converts the inputs into a Pandas DataFrame;
3. passes the DataFrame into the saved pipeline;
4. calculates the probability of churn;
5. displays a risk level and suggested retention action.

The batch-scoring section also allows a CSV containing multiple customers to be scored in one operation.

### 6. Added a Flask API

I also included a Flask API to demonstrate how the same model could be integrated into another software system.

For example, a company's CRM could send a customer's existing account data to the API and receive a churn probability without an employee manually entering the information into the Streamlit interface.

---

## Project structure

```text
customer_churn_end_to_end/
|
├── app.py                         # Streamlit application
├── api.py                         # Optional Flask prediction API
├── README.md
├── MODEL_CARD.md
├── requirements.txt
├── render.yaml
|
├── data/
│   └── README.md
|
├── notebooks/
│   └── 01_churn_analysis.ipynb    # EDA and experimentation
|
├── src/
│   ├── download_data.py           # Makes sure the dataset is available
│   └── train.py                   # Final repeatable training pipeline
|
├── models/                        # Generated trained-model files
├── reports/                       # Generated metrics and confusion matrix
└── docs/
    └── streamlit-app.png
```

---

## Dataset

The project uses the public Telco Customer Churn dataset. The target variable is:

```text
Churn
```

where:

```text
No  = customer stayed
Yes = customer churned
```

The model uses customer demographics, account information, services, contract information and charges. `customerID` is not used as a predictive feature.

The project includes `src/download_data.py`, which checks whether the dataset is already available and attempts to download it when necessary.

---

## Running the project locally

### 1. Clone or download the repository

Open a terminal in the project folder.

### 2. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` contains the Python libraries needed by the project, including Pandas, NumPy, scikit-learn, Streamlit, Flask, Matplotlib and joblib.

### 3. Train the models

```bash
python src/train.py
```

The script will:

- load and clean the data;
- create the train/test split;
- build the preprocessing pipeline;
- train the candidate models;
- evaluate them;
- select the model with the highest ROC-AUC;
- save the winning pipeline;
- save model metadata and evaluation outputs.

### 4. Start the Streamlit application

```bash
python -m streamlit run app.py
```

Streamlit starts a local web server and normally opens the application at:

```text
http://localhost:8501
```

---

## Optional Flask API

Run the API with:

```bash
python api.py
```

Health endpoint:

```text
GET http://localhost:5000/health
```

Prediction endpoint:

```text
POST http://localhost:5000/predict
```

A calling application sends customer data as JSON. For example:

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

The API returns the model's churn probability and predicted class.

---

## What I learned

This project helped me understand the difference between **developing a model** and **turning a model into a usable system**.

The main lessons I took from the project were:

- A Jupyter notebook is useful for exploration and experimentation, while a `.py` training script makes the final process reproducible.
- Preprocessing should be fitted on the training data and kept together with the model in a pipeline.
- Categorical data needs to be converted into a numerical representation before most machine-learning algorithms can use it.
- Accuracy alone can be misleading for an imbalanced classification problem, so precision, recall, F1 and ROC-AUC also matter.
- `predict_proba()` is useful in a churn problem because a business can rank customers by risk instead of relying only on a binary prediction.
- Saving the complete pipeline with `joblib` prevents the application from having to manually reproduce the training transformations.
- Streamlit provides a quick way to expose a model through a human-facing interface.
- A Flask API serves a different purpose: it allows other software systems to send data to the model and receive predictions programmatically.
- Organising an ML project into separate data, notebook, source-code, model, report and application components makes the work easier to reproduce and explain.

---

## Improvements I would make next

If I developed this project further, I would explore:

- cross-validation and more systematic hyperparameter tuning;
- threshold optimisation based on the business cost of false negatives and false positives;
- probability calibration;
- SHAP or another local explanation method for individual predictions;
- experiment tracking;
- model and data-drift monitoring;
- automated tests for preprocessing and API inputs;
- containerisation with Docker;
- deployment to a cloud service and CI/CD from GitHub.

---

## Responsible use

This is a portfolio and learning project built using a public historical dataset. The predictions demonstrate patterns learned from that dataset and should not be treated as causal explanations or used as an automated decision system for real customers without additional validation.

A production churn system would require appropriate data governance, privacy and security controls, fairness testing, monitoring, drift detection, human oversight and periodic retraining.

---

## Technologies used

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `joblib` · `Streamlit` · `Flask` · `Jupyter` · `Git/GitHub`

