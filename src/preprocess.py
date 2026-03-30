import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
    return df


def encode_and_split(df):
    label_encoder = LabelEncoder()
    df["Gender"] = label_encoder.fit_transform(df["Gender"])

    onehot_encoder = OneHotEncoder(drop="first")
    geo_encoded = onehot_encoder.fit_transform(df[["Geography"]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(["Geography"]))

    df = pd.concat([df.drop("Geography", axis=1), geo_df], axis=1)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, label_encoder, onehot_encoder, scaler


def save_artifacts(label_encoder, onehot_encoder, scaler, path="artifacts"):
    with open(f"{path}/label_encoder_gender.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(f"{path}/onehot_encoder_geo.pkl", "wb") as f:
        pickle.dump(onehot_encoder, f)
    with open(f"{path}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
