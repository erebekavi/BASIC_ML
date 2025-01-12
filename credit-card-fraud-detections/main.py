"""\
PROBLEM STATEMENT
---------------------------------------------------------------------
CREDIT CARD FRAUD DETECTION
---------------------------------------------------------------------
Build a model to detect fraudulent credit card transactions. Use a
dataset containing information about credit card transactions, and
experiment with algorithms like Logistic Regression, Decision Trees,
or Random Forests to classify transactions as fraudulent or
legitimate.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


data = pd.read_csv("credit-card-fraud-detections/fraudTrain.csv")
test_data = pd.read_csv("credit-card-fraud-detections/fraudTest.csv")


data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['trans_day'] = data['trans_date_trans_time'].dt.day
data['trans_month'] = data['trans_date_trans_time'].dt.month
data['trans_year'] = data['trans_date_trans_time'].dt.year
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_minute'] = data['trans_date_trans_time'].dt.minute
data.drop(columns=['trans_date_trans_time'], inplace=True)


label_cols = ['merchant', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
encoder = LabelEncoder()

for col in label_cols:
    data[col] = encoder.fit_transform(data[col].astype(str)) 



x_train = data.drop(["is_fraud", "trans_num"], axis=1)  
y_train = data["is_fraud"]


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
