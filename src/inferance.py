import torch
import joblib
import numpy as np
import pandas as pd

from src.model import MyModel

# ---------- Load artifacts ----------
preprocessor = joblib.load("preprocessor.pkl")
model = MyModel(input_dim=15)
model.load_state_dict(torch.load("model.pth"))

model.eval()

# ---------- Predict function ----------
def predict(data_dict):
    df = pd.DataFrame([data_dict])
    X = preprocessor.transform(df)
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        prob = model(X)
    return float(prob.item())

