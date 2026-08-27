import numpy as np
import matplotlib.pyplot as plt


def compute_error(x, y, w, b):
    """Compute the mean squared cost J(w, b) / 2."""
    m = len(x)
    total_error = 0.0

    for i in range(m):
        prediction = w * x[i] + b
        total_error += (prediction - y[i]) ** 2

    return total_error / (2 * m)


def gradient(x, y, w, b):
    """Compute partial derivatives of the cost with respect to w and b."""
    m = len(x)
    dw = 0.0
    db = 0.0

    for i in range(m):
        prediction = w * x[i] + b
        error = prediction - y[i]
        dw += error * x[i]
        db += error

    return dw / m, db / m


x = np.arange(1, 11, dtype=float)
y = np.array([8.5, 14.7, 14.2, 21.8, 19.9, 26.5, 30.1, 28.7, 35.6, 38.2])

w = 0.0
b = 0.0
learning_rate = 0.01
iterations = 10_000

for i in range(iterations):
    dw, db = gradient(x, y, w, b)
    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 1_000 == 0:
        error = compute_error(x, y, w, b)
        print(f"Iteration {i:5d}: Error = {error:.6f}, w = {w:.6f}, b = {b:.6f}")

predictions = w * x + b

plt.plot(x, predictions, label="Our Prediction")
plt.scatter(x, y, marker="x", label="Actual Values")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.tight_layout()
plt.show()
