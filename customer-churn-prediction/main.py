"""
PROBLEM STATEMENT
---------------------------------------------------------------------
CUSTOMER CHURN PREDICTION
---------------------------------------------------------------------
Develop a model to predict customer churn for a subscription-
based service or business. Use historical customer data, including
features like usage behavior and customer demographics, and try
algorithms like Logistic Regression, Random For

"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("customer-churn-prediction/Churn_Modelling.xls")
data.drop(columns=["Surname","Geography","Gender"])
x_train = data.drop(["Exited"],axis=1)
y_train = data["Exited"]

X_train , X_test, Y_train , Y_test = train_test_split(x_train , y_train, test_size=0.2, random_state= 42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test,y_pred)
print(f"accuracy:{accuracy:.4f}")