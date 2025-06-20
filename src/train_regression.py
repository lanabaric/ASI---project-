import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

# Ensure results folder is created relative to script location
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

DROPOUT_RATE = 0.1
NUM_UNITS = 1024
NUM_LAYERS = 5
MC_SAMPLES = 100
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. Load CO₂ Dataset
def load_co2_data():
    import statsmodels.api as sm
    dta = sm.datasets.co2.load_pandas().data
    dta = dta.fillna(method='ffill')  # Fill missing values
    dta.reset_index(inplace=True)
    dta["Index"] = np.arange(len(dta))
    dta = dta[["Index", "co2"]].dropna().values
    return dta


#2. Preprocessing
def preprocess(data):
    X = data[:, 0:1]
    y = data[:, 1:2]
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    return X, y, scaler_x, scaler_y

#3. Model with MC Dropout
class MCDropoutModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(NUM_LAYERS):
            layers.append(nn.Linear(in_dim, NUM_UNITS))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT_RATE))
            in_dim = NUM_UNITS
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#4. Train
def train(model, X_train, y_train):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

#5. Predict with MC Dropout
def predict_mc(model, X_test, T=MC_SAMPLES):
    model.train()  # Important: keep dropout ON
    preds = []
    with torch.no_grad():
        for _ in range(T):
            pred = model(torch.tensor(X_test).float().to(DEVICE)).cpu().numpy()
            preds.append(pred)
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

#6. Plot
def plot_predictions(X_test, mean, std, y_true, scaler_x, scaler_y):
    x_plot = scaler_x.inverse_transform(X_test)
    y_true = scaler_y.inverse_transform(y_true)
    y_mean = scaler_y.inverse_transform(mean)
    y_std = std * scaler_y.scale_

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, label="True", color='black')
    plt.plot(x_plot, y_mean, label="Predictive Mean", color='blue')
    plt.fill_between(x_plot.squeeze(),
                     y_mean.squeeze() - 2*y_std.squeeze(),
                     y_mean.squeeze() + 2*y_std.squeeze(),
                     alpha=0.3, label="±2 Std Dev", color='blue')
    plt.title("MC Dropout Predictive Mean and Uncertainty")
    plt.xlabel("Time")
    plt.ylabel("CO₂ Level (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "co2_prediction.png"))
    plt.show()

#Main
def main():
    data = load_co2_data()
    X, y, scaler_x, scaler_y = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = MCDropoutModel().to(DEVICE)
    train(model, X_train, y_train)

    mean, std = predict_mc(model, X_test)
    plot_predictions(X_test, mean, std, y_test, scaler_x, scaler_y)

if __name__ == "__main__":
    main()
