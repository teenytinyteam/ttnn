import numpy as np

feature = np.array([28.1, 58.0])
label = np.array([165])

weight = np.ones([1, 2]) / 2
bias = np.zeros(1)


def forward(x, w, b):
    return x @ w.T + b


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


def gradient(p, y):
    return (p - y) * 2


def backward(x, d, w, b, lr):
    return w - d * x * lr, b - np.sum(d) * lr


LEARNING_RATE = 0.00001

prediction = forward(feature, weight, bias)
error = mse_loss(prediction, label)

print(f'prediction: {prediction}')
print(f'error: {error}')

delta = gradient(prediction, label)
weight, bias = backward(feature, delta, weight, bias, LEARNING_RATE)

print(f"weight: {weight}")
print(f"bias: {bias}")
