import numpy as np


class DataLoader:

    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.features = [[28.1, 58.0],
                         [22.5, 72.0],
                         [31.4, 45.0],
                         [19.8, 85.0],
                         [27.6, 63]]
        self.labels = [[165],
                       [95],
                       [210],
                       [70],
                       [155]]

    def __len__(self):  # 3
        return len(self.features)

    def __getitem__(self, index):  # 4
        return (Tensor(self.features[index: index + self.batch_size]),
                Tensor(self.labels[index: index + self.batch_size]))


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)

        self.grad = None
        self.gradient_fn = lambda: None
        self.parents = set()

    def gradient(self):
        if self.gradient_fn:
            self.gradient_fn()

        for p in self.parents:
            p.gradient()

    def size(self):
        return self.data.shape[-1]


class Linear:

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size

        self.weight = Tensor(np.ones((out_size, in_size)) / in_size)
        self.bias = Tensor(np.zeros(out_size))

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(x.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            self.weight.grad = p.grad.T @ x.data / len(x.data)
            self.bias.grad = np.sum(p.grad, axis=0) / len(x.data)

        p.gradient_fn = gradient_fn
        p.parents = {self.weight, self.bias}
        return p

    def parameters(self):
        return [self.weight, self.bias]


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        mse = Tensor(((p.data - y.data) ** 2).mean())

        def gradient_fn():
            p.grad = (p.data - y.data) * 2

        mse.gradient_fn = gradient_fn
        mse.parents = {p}
        return mse


class SGD:

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def backward(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr


LEARNING_RATE = 0.00001
BATCHES = 2
EPOCHS = 100

dataset = DataLoader(BATCHES)

feature, label = dataset[0]
model = Linear(feature.size(), label.size())
loss = MSELoss()
sgd = SGD(model.parameters(), LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"epoch: {epoch}")

    for i in range(0, len(dataset), dataset.batch_size):
        feature, label = dataset[i]

        prediction = model(feature)
        error = loss(prediction, label)

        print(f'prediction: {prediction.data}')
        print(f'error: {error.data}')

        error.gradient()
        sgd.backward()

        print(f"weight: {model.weight.data}")
        print(f"bias: {model.bias.data}")
