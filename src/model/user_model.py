import numpy as np

# Load trained model
w = np.load('w.npy')
b = np.load('b.npy')[0]

# Load scaling parameters
means = np.load('mean.npy')
stds = np.load('std.npy')

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

print("\nEnter loan applicant details:")
print("⚠️ Enter encoded values exactly as used in training.\n")

# Order MUST match X_train.columns EXACTLY
feature_names = [
    'Married (0/1)',
    'Dependents (0/1/2/3)',
    'Education (0=Not Graduate, 1=Graduate)',
    'Self_Employed (0/1)',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History (0/1)'
]

user_input = []

for feature in feature_names:
    while True:
        try:
            value = float(input(f"{feature}: "))
            user_input.append(value)
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

# Convert to numpy
X_user = np.array(user_input).reshape(1, -1)

# Apply SAME standardization
X_user_scaled = (X_user - means) / (stds + 1e-15)

# Predict
z = np.dot(X_user_scaled, w) + b
prob = sigmoid(z)[0][0]

prediction = 1 if prob >= 0.5 else 0

print("\n==============================")
print(f"Loan Approval Probability: {prob:.4f}")

if prediction == 1:
    print("Decision: ✅ LOAN APPROVED")
else:
    print("Decision: ❌ LOAN REJECTED")
print("==============================\n")