import csv
import re
from abc import ABC, abstractmethod

import numpy as np

np.random.seed(99)


class DataLoader:

    def __init__(self, sequence_batch_size):
        self.sequence_batch_size = sequence_batch_size

        self.reviews = []
        self.sentiments = []
        with open('reviews.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for _, row in enumerate(reader):
                self.reviews.append(row[0])
                self.sentiments.append(row[1])

        split_reviews = []
        for r in self.reviews:
            split_reviews.append(self.clean_text(r.lower()).split())

        self.vocabulary = set(w for r in split_reviews for w in r)
        self.word2index = {w: idx for idx, w in enumerate(self.vocabulary)}
        self.index2word = {idx: w for idx, w in enumerate(self.vocabulary)}
        self.tokens = [[self.word2index[w] for w in r if w in self.word2index] for r in split_reviews]

        self.train()

    @staticmethod
    def clean_text(text):
        txt = re.sub(r'<[^>]+>', '', text)
        txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
        return txt

    def encode(self, text):
        words = self.clean_text(text.lower()).split()
        return [self.word2index[word] for word in words]

    def decode(self, tokens):
        return " ".join([self.index2word[index] for index in tokens])

    def train(self):
        self.sequences = self.tokens[:-10]

    def eval(self):
        self.sequences = self.tokens[-10:]

    def __len__(self):  # 3
        return len(self.sequences)

    def __getitem__(self, index):  # 4
        return Sequence(self.sequences[index], len(self.vocabulary), self.sequence_batch_size)


class Sequence:

    def __init__(self, tokens, vocabulary_size, batch_size):
        self.tokens = tokens
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size

    def __len__(self):  # 3
        return len(self.tokens) - self.batch_size

    def __getitem__(self, index):  # 4
        return (Tensor([self.tokens[index:index + self.batch_size]]),
                Tensor([self.embedding(self.tokens[index + self.batch_size:
                                                   index + self.batch_size + 1])]))

    def embedding(self, index):
        ebd = np.zeros(self.vocabulary_size)
        ebd[index] = 1
        return ebd


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

    def shape(self):
        return self.data.shape

    def size(self):
        return np.prod(self.data.shape[1:])

    def concat(self, other, axis):
        p = Tensor(np.concatenate([self.data, other.data], axis=axis))

        def gradient_fn():
            self.grad, other.grad = np.split(p.grad, [self.data.shape[axis]], axis=axis)

        p.gradient_fn = gradient_fn
        p.parents = {self, other}
        return p


class Layer(ABC):

    def __init__(self):
        self.training = True

    def __call__(self, *args):
        return self.forward(*args)

    @abstractmethod
    def forward(self, *args):
        pass

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Sequential(Layer):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()


class Linear(Layer):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.weight = Tensor(np.random.rand(out_size, in_size) / in_size)
        self.bias = Tensor(np.zeros(out_size))

    def forward(self, x: Tensor):
        p = Tensor(x.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            self.weight.grad = p.grad.T @ x.data
            self.bias.grad = np.sum(p.grad, axis=0)
            x.grad = p.grad @ self.weight.data

        p.gradient_fn = gradient_fn
        p.parents = {self.weight, self.bias, x}
        return p

    def parameters(self):
        return [self.weight, self.bias]


class Embedding(Layer):

    def __init__(self, vocabulary_size, embedding_size, axis=1):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.axis = axis

        self.weight = Tensor(np.random.rand(embedding_size, vocabulary_size) / vocabulary_size)

    def forward(self, x: Tensor):
        p = Tensor(np.sum(self.weight.data.T[x.data], axis=self.axis))

        def gradient_fn():
            if self.weight.grad is None:
                self.weight.grad = np.zeros_like(self.weight.data)
            self.weight.grad.T[x.data] += p.grad

        p.gradient_fn = gradient_fn
        p.parents = {self.weight}
        return p

    def parameters(self):
        return [self.weight]


class ReLU(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.maximum(0, x.data))

        def gradient_fn():
            x.grad = (p.data > 0) * p.grad

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class Tanh(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.tanh(x.data))

        def gradient_fn():
            x.grad = p.grad * (1 - p.data ** 2)

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class Sigmoid(Layer):

    def __init__(self, clip_range=(-100, 100)):
        super().__init__()
        self.clip_range = clip_range

    def forward(self, x: Tensor):
        z = np.clip(x.data, self.clip_range[0], self.clip_range[1])
        p = Tensor(1 / (1 + np.exp(-z)))

        def gradient_fn():
            x.grad = p.grad * p.data * (1 - p.data)

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class Softmax(Layer):

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def gradient_fn():
            x.grad = np.zeros_like(x.data)
            for idx in range(x.data.shape[0]):
                itm = p.data[idx].reshape(-1, 1)
                x.grad[idx] = (np.diagflat(itm) - itm @ itm.T) @ p.grad[idx]

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        mse = Tensor(((p.data - y.data) ** 2).mean())

        def gradient_fn():
            p.grad = (p.data - y.data) * 2

        mse.gradient_fn = gradient_fn
        mse.parents = {p}
        return mse


class BCELoss:

    def __call__(self, p: Tensor, y: Tensor):
        clipped = np.clip(p.data, 1e-7, 1 - 1e-7)
        bce = Tensor(-np.mean(y.data * np.log(clipped)
                              + (1 - y.data) * np.log(1 - clipped)))

        def gradient_fn():
            p.grad = (clipped - y.data) / (clipped * (1 - clipped) * len(p.data))

        bce.gradient_fn = gradient_fn
        bce.parents = {p}
        return bce


class CELoss:

    def __call__(self, p: Tensor, y: Tensor):
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)

        log = np.log(softmax + 1e-10)
        ce = Tensor(0 - np.sum(y.data * log) / len(p.data))

        def gradient_fn():
            p.grad = (softmax - y.data) / len(p.data)

        ce.gradient_fn = gradient_fn
        ce.parents = {p}
        return ce


class SGD:

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def backward(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr


class RNN(Layer):

    def __init__(self, vocabulary_size, embedding_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.embedding = Embedding(vocabulary_size, embedding_size)
        self.update = Linear(embedding_size * 2, embedding_size)
        self.hidden = Linear(embedding_size, embedding_size)
        self.output = Linear(embedding_size, vocabulary_size)
        self.tanh = Tanh()

        self.layers = [self.embedding,
                       self.update,
                       self.hidden,
                       self.output,
                       self.tanh]

    def __call__(self, x: Tensor, h: Tensor):
        return self.forward(x, h)

    def forward(self, x: Tensor, h: Tensor):
        if not h:
            h = Tensor(np.zeros((1, self.embedding_size)))

        embedding_feature = self.embedding(x)
        concat_feature = embedding_feature.concat(h, axis=1)
        cell_hidden = self.tanh(self.update(concat_feature))
        hidden_feature = self.tanh(self.hidden(cell_hidden))

        return (self.output(hidden_feature),
                Tensor(hidden_feature.data))

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


LEARNING_RATE = 0.02
BATCHES = 2
EPOCHS = 100

dataset = DataLoader(BATCHES)

model = RNN(len(dataset.vocabulary), 64)

loss = CELoss()
sgd = SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")

    for i in range(len(dataset)):
        sequence = dataset[i]

        hidden = None
        for i in range(len(sequence)):
            feature, label = sequence[i]

            prediction, hidden = model(feature, hidden)
            error = loss(prediction, label)

            print(f'Prediction: {prediction.data}')
            print(f'Error: {error.data}')

            error.gradient()
            sgd.backward()

dataset.eval()

for i in range(len(dataset)):
    sequence = dataset[i]

    feature, label = sequence[0]
    original = sequence.tokens[:BATCHES]
    generated = original.copy()

    hidden = None
    for j in range(len(sequence)):
        feature, label = sequence[j]

        prediction, hidden = model(feature, hidden)
        original.append(sequence.tokens[j + BATCHES])
        generated.append(prediction.data.argmax())

    print(f'original: {dataset.decode(original)}')
    print(f'generated: {dataset.decode(generated)}')
