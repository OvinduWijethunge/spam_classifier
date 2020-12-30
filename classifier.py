# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 18:52:45 2020

@author: Ovindu Wijethunge
"""

import pandas as pd
import re
import nltk
#nltk.download("stopwords")
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
ps = PorterStemmer()
corpus = []

messages = pd.read_csv('SMSSpamCollection', sep ='\t',names=["label","message"])


for i in range(0,len(messages)):
   
    review = re.sub('[^a-zA-Z]',' ', messages['message'][i])
    review = review.lower()
    review = review.split()
   
    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    
    review = ' '.join(review)
    
    corpus.append(review)
    


cv = CountVectorizer(max_features=5000) 
X = cv.fit_transform(corpus).toarray()  

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred =spam_detect_model.predict(X_test)


confusion_m  = confusion_matrix(y_test,y_pred) 
accuracy = accuracy_score(y_test,y_pred) 




    
 


