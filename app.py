
import joblib
import pandas as pd
import streamlit as st

bundle = joblib.load("best_bean_model.joblib")
model = bundle["model"]
le = bundle["label_encoder"]
features = bundle["features"]
stats = bundle["feature_stats"]

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘")
st.title("🫘 Dry Bean Type Classifier")
st.write("Enter the physical measurements of the bean and click **Predict**.")

with st.form("bean_form"):
    inputs = {}
    col1, col2 = st.columns(2)
    for i, feat in enumerate(features):
        is_large = stats.loc[feat, "max"] > 100
        with (col1 if i % 2 == 0 else col2):
            inputs[feat] = st.number_input(
                feat,
                min_value=0.0,
                value=float(stats.loc[feat, "50%"]),
                step=1.0 if is_large else 0.0001,
                format="%.2f" if is_large else "%.6f",
            )
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([inputs])[features]
    predicted_class = le.inverse_transform(model.predict(row))[0]
    st.success(f"Predicted bean class: **{predicted_class}**")
