# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:32:10 2021
@author: pradeep
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset/news.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset['text'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y=dataset['label'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# using Naive_bayes 
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating accuracy 
from sklearn.metrics import confusion_matrix,accuracy_score
score=accuracy_score(y_test,y_pred)
print('using Naive Bayes :')
print('accuracy :',score*100,'%')

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#using PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train, y_train)

# Predicting the Test set results
y_pred=pac.predict(X_test)

# Calculating accuracy 
from sklearn.metrics import confusion_matrix,accuracy_score
score=accuracy_score(y_test,y_pred)
print('using PassiveAggressiveClassifier :')
print('accuracy :',score*100,'%')

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
