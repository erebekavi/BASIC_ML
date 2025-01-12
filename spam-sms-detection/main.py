"""
PROBLEM STATEMENT
---------------------------------------------------------------------
SPAM SMS DETECTION
---------------------------------------------------------------------
Build an AI model that can classify SMS messages as spam or
legitimate. Use techniques like TF-IDF or word embeddings with
classifiers like Naive Bayes, Logistic Regression, or Support Vector

Machines to identify spam messages

"""
#`BernoulliNB` is a classifier used for binary features (0 or 1). 
# It predicts classes by calculating probabilities, assuming features are independent. 
# It's useful for tasks like spam detection.

#import package required
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

#load dataset
data = pd.read_csv("spam-sms-detection/spam.csv",sep=",",encoding="latin")

#using TF_IDF for V2 column 
TF_IDF = TfidfVectorizer()
x_train = TF_IDF.fit_transform(data["v2"])

#spliting data
x = x_train
y = data["v1"]

X_train,X_test,Y_train,Y_test = train_test_split( x, y, test_size=0.2,random_state=42)

#training model
model = BernoulliNB()
model.fit(X_train,Y_train)

#predicting the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test,y_pred)
print(f"\n\nAccuracy:{accuracy}\n\n")

#loading the predicted value on CSV file
x_pred = model.predict(X_test)
test_data = data.iloc[Y_test.index].copy()     # integer-location based indexing (.iloc)
test_data["predicted"] = y_pred      
test_data.drop(columns=["v1","Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace =True)
test_data.to_csv("spam-sms-detection/predicted-spam.csv")