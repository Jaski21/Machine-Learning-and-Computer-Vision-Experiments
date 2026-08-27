import numpy as np


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

n_features = x.shape[1]
w = np.zeros(n_features)
b = 0.0

# The raw features have very different scales, so this learning rate is small.
learning_rate = 1e-8
iterations = 100_000

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
