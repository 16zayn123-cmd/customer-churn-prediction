# Data

The project expects the IBM Telco Customer Churn CSV at:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

You have three options:

1. Put your existing CSV in this folder.
2. Run `python src/download_data.py`.
3. Run `python src/train.py`; the training script will try to download the data automatically if it is missing.

The original dataset contains 7,043 customer records and the target column is `Churn`.
