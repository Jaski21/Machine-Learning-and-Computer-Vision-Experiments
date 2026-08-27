import numpy as np
import matplotlib.pyplot as plt


def compute_error(x, y, w, b):
    """Compute the mean squared cost J(w, b) / 2."""
    m = len(x)
    total_error = 0.0

    for i in range(m):
        prediction = np.dot(w, x[i]) + b
        total_error += (prediction - y[i]) ** 2

    return total_error / (2 * m)


def compute_gradient(x, y, w, b):
    """Compute gradients for all weights and the bias."""
    m = len(x)
    dw = np.zeros_like(w)
    db = 0.0

    for i in range(m):
        prediction = np.dot(w, x[i]) + b
        error = prediction - y[i]
        dw += error * x[i]
        db += error

    return dw / m, db / m


def standardize_features(x):
    """Standardize each feature to approximately zero mean and unit variance."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    if np.any(std == 0):
        raise ValueError("A feature has zero variance and cannot be standardized.")

    return (x - mean) / std, mean, std


x = np.array(
    [
        [1200, 2, 20],
        [1500, 3, 15],
        [1800, 3, 10],
        [2000, 4, 8],
        [2200, 4, 5],
        [2500, 5, 4],
        [2800, 5, 2],
        [3000, 6, 1],
    ],
    dtype=float,
)

y = np.array([230, 290, 340, 395, 430, 495, 550, 590], dtype=float)
feature_names = ["Size", "Bedrooms", "Age"]

n_features = x.shape[1]
w = np.zeros(n_features)
b = 0.0
learning_rate = 0.01
iterations = 100_000

original_x = x.copy()
x, mean, std = standardize_features(x)

for i in range(iterations):
    dw, db = compute_gradient(x, y, w, b)
    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 10_000 == 0:
        error = compute_error(x, y, w, b)
        print(
            f"Iteration {i:6d}: Error = {error:.6f}, "
            f"w = {w}, b = {b:.6f}"
        )

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
fig.subplots_adjust(hspace=0.35)

for j, name in enumerate(feature_names):
    axes[0, j].plot(original_x[:, j], marker="o")
    axes[0, j].set_title(f"Original {name}")
    axes[0, j].set_xlabel("Sample")
    axes[0, j].set_ylabel(name)

    axes[1, j].plot(x[:, j], marker="o")
    axes[1, j].set_title(f"Standardized {name}")
    axes[1, j].set_xlabel("Sample")
    axes[1, j].set_ylabel("z-score")

plt.tight_layout()
plt.show()
