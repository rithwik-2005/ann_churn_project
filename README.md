# Customer Churn Prediction — ANN

A production-ready customer churn prediction app built with TensorFlow/Keras and Streamlit. The model is an Artificial Neural Network (ANN) trained on bank customer data to predict whether a customer will leave.

## Live Demo

[https://ann-churn-prediction-ovlb.onrender.com](https://ann-churn-prediction-ovlb.onrender.com)

---

## Project Structure

```
ann_churn/
├── src/
│   ├── preprocess.py        
│   ├── model.py             
│   └── predict.py           
├── artifacts/               
├── .github/
│   └── workflows/
│       └── ci_cd.yml        
├── app.py                   
├── train.py                 
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/rithwik-2005/ann-churn-prediction.git
cd ann-churn-prediction
pip install -r requirements.txt
```

### 2. Train the model

Place `Churn_Modelling.csv` in the project root, then run:

```bash
python train.py
```

This saves the model and encoders to the `artifacts/` folder.

### 3. Run the web app

```bash
streamlit run app.py
```

---

## Docker

### Build and run locally

```bash
docker build -t rithwik_2005/ann-churn-prediction:latest .
docker run -p 8501:8501 rithwik_2005/ann-churn-prediction:latest
```

Open `http://localhost:8501`

### Push to Docker Hub

```bash
docker login
docker push rithwik_2005/ann-churn-prediction:latest
```

---

## CI/CD (GitHub Actions)

The pipeline in `.github/workflows/ci_cd.yml` automatically:

1. Installs dependencies on every push to `main`
2. Builds the Docker image
3. Pushes it to Docker Hub
4. Triggers a redeploy on Render

### Required GitHub Secrets

| Secret | Value |
|--------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub access token |
| `RENDER_DEPLOY_HOOK_URL` | Your Render deploy hook URL |

---

## Deploy on Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo and select **Docker** as environment
4. Render auto-detects `render.yaml` and deploys

---

## Model Details

| Component | Detail |
|-----------|--------|
| Architecture | ANN — 128 → 64 → 32 → 1 (sigmoid) |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Cross-Entropy |
| Dataset | [Churn Modelling (Kaggle)](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| Accuracy | ~86% on test set |

### Features Used

`CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

---

## Tech Stack

- **Model**: TensorFlow / Keras
- **App**: Streamlit
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: Render
