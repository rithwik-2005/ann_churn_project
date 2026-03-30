import os
from src.preprocess import load_data, encode_and_split, save_artifacts
from src.model import build_model, train_model, save_model


def run():
    os.makedirs("artifacts", exist_ok=True)

    df = load_data("Churn_Modelling.csv")
    X_train, X_test, y_train, y_test, label_encoder, onehot_encoder, scaler = encode_and_split(df)

    model = build_model(input_dim=X_train.shape[1])
    train_model(model, X_train, y_train, X_test, y_test)

    save_model(model)
    save_artifacts(label_encoder, onehot_encoder, scaler)
    print("Training complete. Artifacts saved to artifacts/")


if __name__ == "__main__":
    run()