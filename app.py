from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
        .app-title {font-size: 2.2rem; font-weight: 750; margin-bottom: 0;}
        .subtitle {opacity: 0.75; margin-top: 0.25rem; margin-bottom: 1.5rem;}
        .risk-box {
            border: 1px solid rgba(120,120,120,0.25);
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            margin-top: 0.5rem;
        }
        .risk-number {font-size: 2.6rem; font-weight: 800; margin: 0.1rem 0;}
        .small-note {font-size: 0.86rem; opacity: 0.72;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def build_customer_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values])


def risk_label(probability: float) -> str:
    if probability >= 0.70:
        return "High risk"
    if probability >= 0.40:
        return "Medium risk"
    return "Low risk"


def customer_profile_signals(row: dict) -> list[str]:
    """
    Human-readable profile signals.
    These are descriptive heuristics, not local SHAP/model explanations.
    """
    signals = []

    if row["Contract"] == "Month-to-month":
        signals.append("Month-to-month contract")
    if row["tenure"] <= 12:
        signals.append("Customer has been with the company for 12 months or less")
    if row["PaymentMethod"] == "Electronic check":
        signals.append("Electronic-check payment method")
    if row["TechSupport"] in {"No", "No internet service"}:
        signals.append("No active tech-support service")
    if row["OnlineSecurity"] in {"No", "No internet service"}:
        signals.append("No active online-security service")
    if row["MonthlyCharges"] >= 85:
        signals.append("Relatively high monthly charge")

    return signals[:5]


model, metadata = load_artifacts()

st.markdown('<p class="app-title">Customer Churn Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predict churn risk, understand the customer profile, and score customer files in bulk.</p>',
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "The trained model is not present yet. From the project folder run: "
        "`python src/train.py`, then restart Streamlit."
    )
    st.stop()

threshold = float(metadata.get("decision_threshold", 0.5))

tab_predict, tab_batch, tab_model = st.tabs(
    ["Single customer", "Batch scoring", "Model performance"]
)

with tab_predict:
    st.subheader("Customer profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"],
        )
        internet_service = st.selectbox(
            "Internet service",
            ["Fiber optic", "DSL", "No"],
        )
        online_security = st.selectbox(
            "Online security",
            ["No", "Yes", "No internet service"],
        )
        online_backup = st.selectbox(
            "Online backup",
            ["No", "Yes", "No internet service"],
        )
        device_protection = st.selectbox(
            "Device protection",
            ["No", "Yes", "No internet service"],
        )
        tech_support = st.selectbox(
            "Tech support",
            ["No", "Yes", "No internet service"],
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )
        streaming_movies = st.selectbox(
            "Streaming movies",
            ["No", "Yes", "No internet service"],
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=200.0,
            value=75.0,
            step=0.5,
        )
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(max(0, tenure * monthly_charges)),
            step=10.0,
        )

    customer = {
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
    }

    if st.button("Analyse churn risk", type="primary", use_container_width=True):
        customer_df = build_customer_frame(customer)
        probability = float(model.predict_proba(customer_df)[0, 1])
        prediction = int(probability >= threshold)
        label = risk_label(probability)

        result_col, action_col = st.columns([1, 1.2])

        with result_col:
            st.markdown(
                f"""
                <div class="risk-box">
                    <div class="small-note">Predicted churn probability</div>
                    <div class="risk-number">{probability:.1%}</div>
                    <strong>{label}</strong>
                    <div class="small-note">Model decision: {"Likely to churn" if prediction else "Likely to stay"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(probability, 0.0), 1.0))

        with action_col:
            st.markdown("#### Suggested business action")
            if probability >= 0.70:
                st.write(
                    "Prioritise this customer for proactive retention outreach. "
                    "Review contract options, service issues and relevant retention offers."
                )
            elif probability >= 0.40:
                st.write(
                    "Consider a lower-cost retention intervention, such as a service review "
                    "or targeted communication."
                )
            else:
                st.write(
                    "No urgent retention intervention is indicated by this model. "
                    "Continue normal customer engagement."
                )

        signals = customer_profile_signals(customer)
        st.markdown("#### Customer profile signals")
        if signals:
            for signal in signals:
                st.write(f"• {signal}")
        else:
            st.write("No simple high-risk profile signals were triggered.")

        st.caption(
            "Profile signals are descriptive rules for presentation purposes and are not "
            "a causal explanation of the model prediction."
        )

with tab_batch:
    st.subheader("Score a customer CSV")
    st.write(
        "Upload a CSV containing the same feature columns used by the model. "
        "The app will append churn probability, predicted class and risk band."
    )

    template_columns = metadata["features"]
    template_df = pd.DataFrame(columns=template_columns)
    st.download_button(
        "Download input template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_scoring_template.csv",
        mime="text/csv",
    )

    upload = st.file_uploader("Upload customer CSV", type=["csv"])

    if upload is not None:
        batch_df = pd.read_csv(upload)
        missing = [col for col in template_columns if col not in batch_df.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            score_input = batch_df[template_columns].copy()
            probabilities = model.predict_proba(score_input)[:, 1]

            output = batch_df.copy()
            output["churn_probability"] = probabilities
            output["predicted_churn"] = (probabilities >= threshold).astype(int)
            output["risk_band"] = [
                risk_label(float(prob)) for prob in probabilities
            ]

            st.success(f"Scored {len(output):,} customers.")
            st.dataframe(output.head(50), use_container_width=True)

            st.download_button(
                "Download scored customers",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="scored_customers.csv",
                mime="text/csv",
                type="primary",
            )

with tab_model:
    st.subheader("Model performance")

    selected = metadata["selected_model"]
    selected_metrics = metadata["metrics"][selected]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selected model", selected)
    c2.metric("ROC-AUC", f"{selected_metrics['roc_auc']:.3f}")
    c3.metric("Recall", f"{selected_metrics['recall']:.3f}")
    c4.metric("Precision", f"{selected_metrics['precision']:.3f}")
    c5.metric("F1", f"{selected_metrics['f1']:.3f}")

    st.markdown("#### Model comparison")
    metrics_df = (
        pd.DataFrame(metadata["metrics"])
        .T.reset_index()
        .rename(columns={"index": "model"})
    )
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("#### Most influential transformed features")
    importance = pd.DataFrame(metadata.get("global_feature_importance", []))
    if not importance.empty:
        display_cols = ["feature", "importance"]
        st.dataframe(importance[display_cols], use_container_width=True)
    else:
        st.info("Feature importance is not available for this estimator.")

    st.markdown("#### Responsible-use note")
    st.write(
        "This is a portfolio demonstration trained on a historical public dataset. "
        "It should not be used as an automated decision-maker for real customers without "
        "validation, monitoring, privacy controls, fairness review and human oversight."
    )
