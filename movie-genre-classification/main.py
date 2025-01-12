"""
PROBLEM STATEMENT
------------------------------------------------------------------
MOVIE GENRE CLASSIFICATION
------------------------------------------------------------------
Create a machine learning model that can predict the genre of a
movie based on its plot summary or other textual information. You
can use techniques like TF-IDF or word embeddings with classifiers
such as Naive Bayes, Logistic Regression, or Support Vector
Machines.

"""
#Multinomial Naïve Bayes is great for classifying text or documents based on word counts.
#TF-IDF (Term Frequency-Inverse Document Frequency) shows how important a word is in a document compared to others.

#import package required
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


#load dataset
train_data = pd.read_csv("movie-genre-classification/dataset/train_data.txt",sep=":::",names=["title","genre","des"],engine="python")
test_data = pd.read_csv("movie-genre-classification/dataset/test_data.txt",sep=":::",names=["id","title","des"],engine="python")


# using TF-IDF method on dataset
TF_IDF = TfidfVectorizer()

x_train = TF_IDF.fit_transform(train_data["des"])
x_test = TF_IDF.transform(test_data["des"])

#spliting dataset
x = x_train
y = train_data["genre"]

X_train,X_test,Y_train,Y_test = train_test_split( x, y, test_size=0.2, random_state=42) #we splited dataset train - 70% and test - 20%

#training the model
model = MultinomialNB() 
model.fit(X_train,Y_train)

#predicting the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test,y_pred)
print(f"\n\nAccuracy:{accuracy}\n\n")

#loading the predicted value on CSV file
x_pred = model.predict(x_test)
test_data["predicted-Genre"] = x_pred
test_data.to_csv("movie-genre-classification/predicted-genre.csv")






