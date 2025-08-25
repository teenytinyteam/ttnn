import numpy as np

np.random.seed(99)

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

hidden_weight = np.random.rand(4, 2) / 2
hidden_bias = np.zeros(4)
output_weight = np.random.rand(1, 4) / 4
output_bias = np.zeros(1)


def forward(x, w, b):
    return x @ w.T + b


def relu(x):
    return np.maximum(0, x)


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


def gradient(p, y):
    return (p - y) * 2


def gradient_backward(d, w):
    return d @ w


def relu_backward(y, d):
    return (y > 0) * d


def backward(x, d, w, b, lr):
    return w - d.T @ x * lr / len(x), b - np.sum(d, axis=0) * lr / len(x)


LEARNING_RATE = 0.00001
BATCHES = 2
EPOCHS = 100

for epoch in range(EPOCHS):
    print(f"epoch: {epoch}")

    for i in range(0, len(features), BATCHES):
        feature = features[i: i + BATCHES]
        label = labels[i: i + BATCHES]

        hidden = relu(forward(feature, hidden_weight, hidden_bias))
        prediction = forward(hidden, output_weight, output_bias)
        error = mse_loss(prediction, label)

        print(f'prediction: {prediction}')
        print(f'error: {error}')

        output_delta = gradient(prediction, label)
        hidden_delta = relu_backward(hidden, gradient_backward(output_delta, output_weight))

        output_weight, output_bias = backward(hidden, output_delta, output_weight, output_bias, LEARNING_RATE)
        hidden_weight, hidden_bias = backward(feature, hidden_delta, hidden_weight, hidden_bias, LEARNING_RATE)

        print(f"hidden weight: {hidden_weight}")
        print(f"hidden bias: {hidden_bias}")
        print(f"output weight: {output_weight}")
        print(f"output bias: {output_bias}")
