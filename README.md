#  Customer Churn Prediction — ANN

A production-ready customer churn prediction app built with TensorFlow/Keras and Streamlit. The model is an Artificial Neural Network (ANN) trained on bank customer data to predict whether a customer will leave.

---

## Project Structure

```
ann_churn/
├── src/
│   ├── preprocess.py        # Data loading, encoding, scaling
│   ├── model.py             # ANN architecture and training
│   └── predict.py           # Load artifacts and run inference
├── artifacts/               # Saved model + encoders (gitignored)
├── .github/
│   └── workflows/
│       └── ci_cd.yml        # GitHub Actions CI/CD pipeline
├── app.py                   # Streamlit web app
├── train.py                 # Training entry point
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

---

##  Quickstart

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/ann-churn-prediction.git
cd ann-churn-prediction
pip install -r requirements.txt
```

### 2. Train the model

Place `Churn_Modelling.csv` in the project root, then run:

```bash
python train.py
```

This saves `model.h5`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, and `scaler.pkl` to the `artifacts/` folder.

### 3. Run the web app

```bash
streamlit run app.py
```

---

##  Docker

### Build and run locally

```bash
docker build -t <your_dockerhub_username>/ann-churn-prediction:latest .
docker run -p 8501:8501 <your_dockerhub_username>/ann-churn-prediction:latest
```

Open `http://localhost:8501`

### Push to Docker Hub

```bash
docker login
docker push <your_dockerhub_username>/ann-churn-prediction:latest
```

---

##  CI/CD (GitHub Actions)

The pipeline in `.github/workflows/ci_cd.yml` automatically:

1. Installs dependencies on every push to `main`
2. Builds the Docker image
3. Pushes it to Docker Hub
4. Triggers a redeploy on Render

### Required GitHub Secrets

Go to **Settings → Secrets and Variables → Actions** and add:

| Secret | Value |
|--------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub password or access token |
| `RENDER_DEPLOY_HOOK_URL` | Your Render deploy hook URL |

---

##  Deploy on Render

1. Push your code (with the `artifacts/` folder included or use Render environment) to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects the `render.yaml` and deploys using Docker
5. Or set **Deploy Hook URL** in Render dashboard and add it as a GitHub secret

---

##  Model Details

| Component | Detail |
|-----------|--------|
| Architecture | ANN — 64 → 32 → 1 (sigmoid) |
| Optimizer | Adam (lr=0.01) |
| Loss | Binary Cross-Entropy |
| Dataset | [Churn Modelling (Kaggle)](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| Accuracy | ~86% on test set |

### Features Used

`CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

---

##  Dataset

Download `Churn_Modelling.csv` from [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) and place it in the project root before training.

---

##  Tech Stack

- **Model**: TensorFlow / Keras
- **App**: Streamlit
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: Render
