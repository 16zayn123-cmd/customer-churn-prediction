# Model Card — Customer Churn Classifier

## Purpose

This model is a portfolio demonstration that estimates the probability that a telecom customer will churn.

## Data

The project uses the public IBM-style Telco Customer Churn dataset. It includes customer demographics, services, account information, charges and a binary `Churn` target.

## Modelling approach

The training script compares:

- Logistic Regression with class balancing
- Random Forest with class balancing

Both models use the same preprocessing pipeline:

- Median imputation and standard scaling for numeric features
- Most-frequent imputation and one-hot encoding for categorical features

The model with the highest test ROC-AUC is saved.

## Intended use

- Learning end-to-end machine learning
- Portfolio demonstrations
- Showing how model predictions can be exposed through a UI or API
- Demonstrating batch scoring

## Not intended for

- Live commercial customer decisions
- Automatic denial of service or benefits
- Causal conclusions about why a customer churned

## Limitations

This dataset is historical and simplified. A real churn system would require data governance, privacy controls, fairness testing, model monitoring, drift detection, security, retraining and validation against current business data.
