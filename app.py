import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# --- Load model and encoders ---
@st.cache_resource
def load_artifacts():
    model = load_model("model.h5")
    with open("label_emcoder_gender.pkl", "rb") as f:
        label_encoder_gen = pickle.load(f)
    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, label_encoder_gen, onehot_encoder_geo, scaler

model, label_encoder_gen, onehot_encoder_geo, scaler = load_artifacts()

st.title("Customer Churn Prediction")

# --- Input Form ---
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Geography", [""] + list(onehot_encoder_geo.categories_[0]))
        gender = st.selectbox("Gender", [""] + list(label_encoder_gen.classes_))

        age = st.text_input("Age (18–92)")
        credit_score = st.text_input("Credit Score (300–850)")
        balance = st.text_input("Balance (0.0–300000.0)")

    with col2:
        estimated_salary = st.text_input("Estimated Salary (0.0–300000.0)")
        tenure = st.text_input("Tenure (years) (0–10)")
        num_products = st.selectbox("Number of Products", ["", 1, 2, 3, 4])
        has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.radio("Is Active Member?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# --- Prediction ---
if submitted:
    # Check for any missing required input
    if "" in [geography, gender, age, credit_score, balance, estimated_salary, tenure, num_products]:
        st.warning("Please fill in all fields.")
    else:
        try:
            # Convert inputs to correct types
            input_dict = {
                "CreditScore": int(credit_score),
                "Gender": label_encoder_gen.transform([gender])[0],
                "Age": int(age),
                "Tenure": int(tenure),
                "Balance": float(balance),
                "NumOfProducts": int(num_products),
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active_member == "Yes" else 0,
                "EstimatedSalary": float(estimated_salary),
            }

            df = pd.DataFrame([input_dict])
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))
            df = pd.concat([df, geo_df], axis=1)

            df = df[scaler.feature_names_in_]
            scaled = scaler.transform(df)

            prediction = model.predict(scaled)[0][0]

            st.metric("Churn Probability", f"{prediction:.1%}")
            if prediction > 0.5:
                st.error("High risk of churn ⚠️")
            else:
                st.success("Low churn risk ✅")

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
