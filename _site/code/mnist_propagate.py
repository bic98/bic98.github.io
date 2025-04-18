from keras.src.ops import normalize, one_hot
import numpy as np
from collections import OrderedDict

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  # W.T is correct, but need to ensure W is not None
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # If t is one-hot encoded
    if t.size == y.size:
        return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]
    # If t is label encoded
    else:
        return -np.sum(np.log(y[np.arange(y.shape[0]), t] + 1e-7)) / y.shape[0]

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if batch_size == 0:
            raise ValueError("Batch size cannot be zero")
        dx = (self.y - self.t) / batch_size
        return dx

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights and biases
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Create layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Set gradients
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

import tensorflow as tf
import keras
from tqdm import tqdm  # tqdm 임포트

# MNIST 데이터 불러오기
(x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("훈련 데이터 크기:", x_train.shape)  # (60000, 28, 28)
print("훈련 라벨 크기:", t_train.shape)  # (60000,)
print("테스트 데이터 크기:", x_test.shape)  # (10000, 28, 28)
print("테스트 라벨 크기:", t_test.shape)  # (10000,)

x_train = x_train.reshape(-1, 28*28)  # (60000, 784)
x_test = x_test.reshape(-1, 28*28)    # (10000, 784)
print("Flatten 후 훈련 데이터 크기:", x_train.shape)  # (60000, 784)

# One-hot encode the labels
t_train = keras.utils.to_categorical(t_train, num_classes=10)
t_test = keras.utils.to_categorical(t_test, num_classes=10)
print("One-hot 변환 후 레이블 크기:", t_train.shape)  # (60000, 10)


iters_num = 1e5
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=128, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in tqdm(range(int(iters_num)), ncols=80, mininterval=1.0):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")

import pickle

# 모델 저장 경로 설정
model_path = 'mnist_model.pkl'

# 모델 저장
with open(model_path, 'wb') as f:
    pickle.dump(network, f)
    
print(f"모델이 {model_path}에 저장되었습니다.")
