import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Ensure results directory exists
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT_RATE = 0.5
MC_SAMPLES = 100


#1. Define LeNet with Dropout before FC layers
class LeNetDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # dropout only before final layer
        return self.fc2(x)


#2. Train the model
def train_model(model, train_loader, epochs=5, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


#3. Get rotated versions of an image
def rotate_image(image, angles):
    return [TF.rotate(image, angle) for angle in angles]


#4. Perform MC Dropout on rotated inputs
def mc_dropout_predictions(model, images, samples=MC_SAMPLES):
    model.train()  # dropout ON
    all_preds = []

    for img in tqdm(images, desc="Rotated MC Samples"):
        img = img.unsqueeze(0).to(DEVICE)
        preds = []
        with torch.no_grad():
            for _ in range(samples):
                logits = model(img)
                prob = F.softmax(logits, dim=1).cpu().numpy()
                preds.append(prob)
        all_preds.append(np.squeeze(np.array(preds)))  # (samples, 10)
    return all_preds  # list of (samples, 10)


#5. Plot softmax confidence and uncertainty
def plot_softmax_variation(mc_preds, angles):
    means = [np.mean(p, axis=0) for p in mc_preds]
    stds = [np.std(p, axis=0) for p in mc_preds]
    means = np.array(means)
    stds = np.array(stds)

    top_classes = np.argsort(means, axis=1)[:, -3:]  # Top 3 per rotation

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(3):
        class_idx = top_classes[:, 2 - i]  # get i-th top class
        mean_vals = means[np.arange(len(angles)), class_idx]
        std_vals = stds[np.arange(len(angles)), class_idx]
        axs[0].plot(angles, mean_vals, label=f"Class {class_idx[0]}")
        axs[1].plot(angles, std_vals, label=f"Class {class_idx[0]}")

    axs[0].set_title("Softmax Mean Confidence vs Rotation")
    axs[1].set_title("Softmax Uncertainty (Std Dev) vs Rotation")
    axs[1].set_xlabel("Rotation Angle (degrees)")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel("Softmax Mean")
    axs[1].set_ylabel("Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mnist_mc_dropout_rotation.png"))
    plt.show()


#Main
def main():
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Build and train model
    model = LeNetDropout().to(DEVICE)
    train_model(model, train_loader)

    # Choose a digit "1" from test set
    one_img = next(img for img, label in test_dataset if label == 1)

    #Rotate and predict
    angles = list(range(0, 185, 15)) 
    rotated_imgs = rotate_image(one_img, angles)
    mc_preds = mc_dropout_predictions(model, rotated_imgs)

    plot_softmax_variation(mc_preds, angles)


if __name__ == "__main__":
    main()
