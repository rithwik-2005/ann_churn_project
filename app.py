import streamlit as st
from src.predict import load_artifacts, predict

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🏦", layout="centered")

@st.cache_resource
def get_artifacts():
    return load_artifacts()

model, label_encoder, onehot_encoder, scaler = get_artifacts()

st.title("🏦 Customer Churn Prediction")
st.markdown("Enter customer details to predict churn probability using an ANN model.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Geography", list(onehot_encoder.categories_[0]))
        gender = st.selectbox("Gender", list(label_encoder.classes_))
        age = st.slider("Age", 18, 92, 35)
        credit_score = st.slider("Credit Score", 300, 850, 600)
        balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0)

    with col2:
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=50000.0)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.radio("Is Active Member?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

if submitted:
    input_data = {
        "geography": geography,
        "gender": gender,
        "age": age,
        "credit_score": credit_score,
        "balance": balance,
        "estimated_salary": estimated_salary,
        "tenure": tenure,
        "num_products": num_products,
        "has_cr_card": 1 if has_cr_card == "Yes" else 0,
        "is_active_member": 1 if is_active_member == "Yes" else 0,
    }

    try:
        probability = predict(model, label_encoder, onehot_encoder, scaler, input_data)
        st.divider()
        st.metric("Churn Probability", f"{probability:.1%}")

        if probability > 0.5:
            st.error(f"⚠️ High risk of churn ({probability:.1%})")
        else:
            st.success(f"✅ Low churn risk ({probability:.1%})")

        st.progress(probability)

    except Exception as e:
        st.error(f"Prediction error: {e}")
