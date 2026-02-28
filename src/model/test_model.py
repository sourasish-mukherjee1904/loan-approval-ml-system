import pandas as pd
import numpy as np
w = np.load('w.npy')
b = np.load('b.npy')[0]
X_test = pd.read_csv('../../data/X_test.csv').drop(columns=['Loan_ID', 'Property_Area', 'Gender']).to_numpy(dtype=float)
y_test = pd.read_csv('../../data/y_test.csv').to_numpy(dtype=float)
y_test = y_test.reshape(-1, 1)
medians = np.load('medians.npy')
means = np.load('mean.npy')
stds = np.load('std.npy')
X_test_scaled = (X_test - means) / (stds + 1e-15)
no_of_times_correct = 0
for i in range(X_test.shape[0]):
    z = np.dot(X_test_scaled[i], w) + b
    A = 1/(1+np.exp(-z))
    predicted_label = 1 if A >= 0.5 else 0
    if predicted_label == y_test[i][0]:
        no_of_times_correct += 1
accuracy = no_of_times_correct / X_test_scaled.shape[0]
print(f"No. of times correct: {no_of_times_correct}")
print(f"No of test sets: {X_test_scaled.shape[0]} samples")
print(f"Accuracy: {100*accuracy:.2f}%")
