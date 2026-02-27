import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv("../data/loan_data.csv.xls")
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['Credit_History'].fillna(0, inplace=True)
df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})
le = LabelEncoder()

cols = ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']

for col in cols:
    df[col] = le.fit_transform(df[col])
    
scaler = StandardScaler()

num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']

df[num_cols] = scaler.fit_transform(df[num_cols])
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.to_csv("../data/X_train.csv",index=False)
X_test.to_csv("../data/X_test.csv",index=False)
y_train.to_csv("../data/y_train.csv",index=False)
y_test.to_csv("../data/y_test.csv",index=False)