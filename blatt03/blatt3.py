import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())  # Check for CUDA
print(torch.backends.mps.is_available())  # For Apple Silicon
print(torch.backends.cpu.get_cpu_capability())  # CPU info
print(torch.__config__.show())


def heaviside(x):
    return torch.where(x > 0, 1, 0)

def perceptron_loss(label, true_label):
    return torch.sum(label != true_label).item()

def perceptron_infer(x, params):
    weights, bias = params
    return heaviside(torch.matmul(x, weights) + bias)

def optimizer_step(x, y, params):
    weights, bias = params
    for i in range(len(x)):
        y_pred = perceptron_infer(x[i], params)
        error = y[i] - y_pred
        weights += error * x[i]  # Update der Gewichte
        bias += error  # Update des Bias
    return weights, bias

def train_perceptron(x, y, epochs):
    params = (torch.zeros(x.shape[1]), torch.tensor(0.0))
    for epoch in range(epochs):
        y_pred = perceptron_infer(x, params)
        loss = perceptron_loss(y_pred, y)
        print(f"Loss at {loss} in epoch {epoch}")
        params = optimizer_step(x, y, params)
    return params


X, y = make_moons(n_samples=100, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# In Torch-Tensoren umwandeln
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# created the following example with ChatGPT

def visualize_classification(X_test, y_test, params):
    weights, bias = params
    X_min, X_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    Y_min, Y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.1),
                         np.arange(Y_min, Y_max, 0.1))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    Z = perceptron_infer(grid, params).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40, edgecolor='k')
    plt.show()

# Trainiere das Perzeptron und visualisiere das Ergebnis
params = train_perceptron(X_train_tensor, y_train_tensor, epochs=10)
visualize_classification(X_test, y_test, params)