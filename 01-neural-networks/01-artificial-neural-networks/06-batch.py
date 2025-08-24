import numpy as np

features = np.array([[28.1, 58.0],
                     [22.5, 72.0],
                     [31.4, 45.0],
                     [19.8, 85.0],
                     [27.6, 63]])
labels = np.array([[165],
                   [95],
                   [210],
                   [70],
                   [155]])

weight = np.ones([1, 2]) / 2
bias = np.zeros(1)


def forward(x, w, b):
    return x @ w.T + b


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


def gradient(p, y):
    return (p - y) * 2


def backward(x, d, w, b, lr):
    return w - d.T @ x * lr / len(x), b - np.sum(d, axis=0) * lr / len(x)


LEARNING_RATE = 0.00001
BATCHES = 2

for i in range(0, len(features), BATCHES):
    feature = features[i: i + BATCHES]
    label = labels[i: i + BATCHES]

    prediction = forward(feature, weight, bias)
    error = mse_loss(prediction, label)

    print(f'prediction: {prediction}')
    print(f'error: {error}')

    delta = gradient(prediction, label)
    weight, bias = backward(feature, delta, weight, bias, LEARNING_RATE)

    print(f"weight: {weight}")
    print(f"bias: {bias}")
