Fake_News_Detector

The model detects whether the given news is REAL or FAKE.
Algorithm
In this model we used two different algorithms :
Naive Bayes - 80.26%
PassiveAggressiveClassifier - 88.32%
Dataset

you can download the dataset from the link given (https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view) ::

=>Download the dataset and put it in a folder named dataset
The dataset has a shape of 7795×4. The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE.

To run the model

    python Fake_news_detector.py
The accuracy of the model is 80.26% using Naive Bayes and 88.32% using PassiveAggressiveClassifier.




