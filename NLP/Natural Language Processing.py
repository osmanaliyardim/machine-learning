# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:22:53 2019

@author: Osman Ali Yardım

Machine Learning - NLP Implementation

Prediction of Genders from Profile Description
"""

import pandas as pd

data = pd.read_csv(r'gender-classifier.csv', encoding='latin1')

#data manipulating
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis=0, inplace=True)
data.gender = [1 if each == 'female' else 0 for each in data.gender]

# %% cleaning and preparing data (regular expression)
import re

first_description = data.description[4] #a sample
description = re.sub("[^a-zA-Z]", " ", first_description) #change words dont exist in alphabet with a space
description = description.lower() #change all words to lowercase

# %% stopwords (irrelevant words)
import nltk #natural language tool kit
nltk.download('stopwords')
from nltk.corpus import stopwords

#description = description.split()
nltk.download('punkt')
description = nltk.word_tokenize(description) # to seperate shouldn't, don't etc.

# %% extract irrelevant words (the, and, or, to..)
description = [word for word in description if not word in set(stopwords.words("english"))]

# %% lemmazation (loved -> love)
import nltk as nlp
nltk.download('wordnet')

lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]

description = " ".join(description)

# %% apply it for whole profiles
description_list = []

for description in data.description:
    description = re.sub("[^a-zA-Z]", " ", description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [word for word in description if not word in set(stopwords.words("english"))] #it can take too much time
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
    
# %% bag of words
from sklearn.feature_extraction.text import CountVectorizer #to create bag of words

max_features = 500

count_vectorizer = CountVectorizer(max_features=max_features) #stop_words='english' can be used here

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() #x

print('{} most used words: {}'.format(max_features, count_vectorizer.get_feature_names()))

# %% text classification
y = data.iloc[:,0].values #male or female
x = sparce_matrix

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# naive bayes
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()

nb.fit(x_train, y_train)

#prediction
#y_pred = nb.predict(x_test)
print('accuracy: ', nb.score(x_test, y_test))