import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
X_train = pd.read_csv('../../data/X_train.csv').drop(columns=['Loan_ID', 'Property_Area', 'Gender'])
medians = X_train.median().to_numpy(dtype=float)
medians[medians == 0] = 1
means = X_train.mean().to_numpy(dtype=float)
stds = X_train.std().to_numpy(dtype=float)
X_train_scaled = (X_train - means) / (stds + 1e-15)
X_train_scaled = X_train_scaled.to_numpy(dtype=float)
y_train = pd.read_csv('../../data/y_train.csv').to_numpy(dtype=float)
y_train = y_train.reshape(-1, 1)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def forward(X, w, b):
    
    z = np.dot(X, w) + b
    A = sigmoid(z)
    
    return A

def compute_cost(A, y):
    m  = y.shape[0]
    cost = -1/m * np.sum(y * np.log(A + 1e-15) + (1-y) * np.log(1-A + 1e-15))
    return cost

def gradient_descent(X, A, y):
    m = y.shape[0]
    dw = 1/m * np.dot(X.T, (A - y))
    db = 1/m * np.sum(A - y)
    return dw, db

def train(X, y, learning_rate = 0.01, num_iterations = 100000):
    n_features = X.shape[1]
    w = np.random.rand(n_features, 1)*0.01
    b = 0.0
    for i in range(num_iterations):
        A = forward(X, w, b)
        cost = compute_cost(A, y)
        
        if(i % 10000 == 0):
            print(f"Iteration {i}, cost={cost:.6f}")
        dw, db = gradient_descent(X, A, y)
        w -= learning_rate * dw
        b -= learning_rate * db
    
    return w, b

w, b = train(X_train_scaled, y_train, learning_rate=0.0001, num_iterations=1000000)
np.save('w.npy', w)
np.save('b.npy', np.array([b]))
np.save('medians.npy', medians)
np.save('mean.npy', X_train.mean().to_numpy(dtype=float))
np.save('std.npy', X_train.std().to_numpy(dtype=float))
print("Training finished. Weights saved.")

    