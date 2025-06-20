import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Fix random seed
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_SAMPLES = 20
EPOCHS = 100
BATCH_SIZE = 32
DROPOUT_RATE = 0.1
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

#Dataset Loaders 
def load_dataset(name):
    from sklearn.datasets import fetch_openml

    if name == "boston":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        return data.data, data.target

    elif name == "concrete":
        data = fetch_openml(data_id=165, as_frame=True)  # ID for concrete
        return data.data.to_numpy(), data.target.to_numpy()

    elif name == "energy":
        data = fetch_openml(data_id=44074, as_frame=True)  # ID for energy efficiency
        return data.data.iloc[:, :8].to_numpy(), data.data.iloc[:, 8].to_numpy()

    else:
        raise ValueError("Unknown dataset")


#MC Dropout Network
class MCDropoutRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(50, 50)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.out = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

#Training
def train_model(model, X_train, y_train):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#MC Predictions
def predict_mc(model, X, T=MC_SAMPLES):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(T):
            out = model(torch.tensor(X).float().to(DEVICE)).cpu().numpy()
            preds.append(out)
    preds = np.stack(preds, axis=0)  # (T, N, 1)
    mean = preds.mean(axis=0).squeeze()
    var = preds.var(axis=0).squeeze()
    return mean, var

#Evaluation
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def predictive_log_likelihood(y_true, y_pred, var):
    tau = 1.0  # assuming unit noise precision
    return -0.5 * np.mean(np.log(2 * np.pi * var) + ((y_true - y_pred) ** 2) / var)

#Main Experiment
def evaluate_dataset(name):
    print(f"\nDataset: {name}")
    X, y = load_dataset(name)
    X, y = shuffle(X, y, random_state=42)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).squeeze()

    rmses, lls = [], []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        model = MCDropoutRegressor(X.shape[1]).to(DEVICE)
        train_model(model, X_train, y_train)
        y_mean, y_var = predict_mc(model, X_test)

        rmses.append(rmse(y_test, y_mean))
        lls.append(predictive_log_likelihood(y_test, y_mean, y_var + 1e-6))  # add epsilon to prevent log(0)

    print(f"Avg RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"Avg Log Likelihood: {np.mean(lls):.4f} ± {np.std(lls):.4f}")

if __name__ == "__main__":
    for dataset in ["boston", "energy"]:
        evaluate_dataset(dataset)

