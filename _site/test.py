import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # Prevent overflow
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights and biases
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y

    def _cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        batch_size = y.shape[0]
        
        # Add small value to prevent log(0)
        delta = 1e-7
        
        # Handle both one-hot and label encoded targets
        if t.size == y.size:  # one-hot
            return -np.sum(t * np.log(y + delta)) / batch_size
        else:  # label encoded
            return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

    def loss(self, x, t):
        y = self.predict(x)
        return self._cross_entropy_error(y, t)

    def _numerical_gradient(self, f, x):
        h = 1e-4  # Small value for numerical differentiation
        grad = np.zeros_like(x)
        
        # Vectorized operations where possible
        it = np.nditer(x, flags=['multi_index'], op_flags=[['readwrite']])
        while not it.finished:
            idx = it.multi_index
            orig_val = x[idx]
            
            # Calculate f(x+h)
            x[idx] = orig_val + h
            fxh1 = f(x)
            
            # Calculate f(x-h)
            x[idx] = orig_val - h
            fxh2 = f(x)
            
            # Gradient at this point
            grad[idx] = (fxh1 - fxh2) / (2*h)
            
            # Restore original value
            x[idx] = orig_val
            it.iternext()
        
        return grad

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1) if t.ndim != 1 else t
        
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = self._numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self._numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self._numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self._numerical_gradient(loss_W, self.params['b2'])
        
        return grads


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# MNIST 데이터 불러오기
(x_train, t_train), (x_test, t_test) = mnist.load_data()

print("훈련 데이터 크기:", x_train.shape)  # (60000, 28, 28)
print("훈련 라벨 크기:", t_train.shape)  # (60000,)
print("테스트 데이터 크기:", x_test.shape)  # (10000, 28, 28)
print("테스트 라벨 크기:", t_test.shape)  # (10000,)

x_train = x_train.reshape(-1, 28*28)  # (60000, 784)
x_test = x_test.reshape(-1, 28*28)    # (10000, 784)
print("Flatten 후 훈련 데이터 크기:", x_train.shape)  # (60000, 784)

t_train = to_categorical(t_train, num_classes=10)
t_test = to_categorical(t_test, num_classes=10)
print("One-hot 변환 후 레이블 크기:", t_train.shape)  # (60000, 10)


iters_num = 1e4
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []



# Training loop with tqdm
for i in tqdm(range(int(iters_num)), desc="Training Progress", ncols=80, mininterval=0.1):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # Calculate gradient
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # Update parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
