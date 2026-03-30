import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model


def load_artifacts(path="artifacts"):
    model = load_model(f"{path}/model")
    with open(f"{path}/label_encoder_gender.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(f"{path}/onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder = pickle.load(f)
    with open(f"{path}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, label_encoder, onehot_encoder, scaler


def predict(model, label_encoder, onehot_encoder, scaler, input_data: dict):
    df = pd.DataFrame([{
        "CreditScore": input_data["credit_score"],
        "Gender": label_encoder.transform([input_data["gender"]])[0],
        "Age": input_data["age"],
        "Tenure": input_data["tenure"],
        "Balance": input_data["balance"],
        "NumOfProducts": input_data["num_products"],
        "HasCrCard": input_data["has_cr_card"],
        "IsActiveMember": input_data["is_active_member"],
        "EstimatedSalary": input_data["estimated_salary"],
    }])

    geo_encoded = onehot_encoder.transform([[input_data["geography"]]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(["Geography"]))
    df = pd.concat([df, geo_df], axis=1)
    df = df[scaler.feature_names_in_]

    scaled = scaler.transform(df)
    probability = model.predict(scaled)[0][0]
    return float(probability)
