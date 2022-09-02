import numpy as np
import pandas as pd
import os
import pyprind
import re

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Create the movie dataset
# basepath = 'aclImdb'
# labels = {'pos': 1, 'neg': 0}
# pbar = pyprind.ProgBar(50000)
# df = pd.DataFrame()

# for s in ('test', 'train'):
#     for l in ('pos', 'neg'):
#         path = os.path.join(basepath, s, l)
        
#         for file in sorted(os.listdir(path)):
#             with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
#                 review = infile.read()
#             df = df.append([[review, labels[l]]], ignore_index=True)
            
#             pbar.update()

# df.columns = ['review', 'sentiment']

# Shuffle the dataframe
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))

# Save the dataframe to a CSV file
# df.to_csv('movie_data.csv', index=False, encoding='utf-8')

# Load the dataframe from the CSV file
df = pd.read_csv('movie_data.csv', encoding='utf-8')

# Preprocessing the movie dataset
# Clean the text data
def preprocessor(text):
      text = re.sub('<[^>]*>', '', text)
      emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
      # Lowercase the text and put the emoticons back
      text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
      return text

df['review'] = df['review'].apply(preprocessor)

# Split the dataset into training and test sets
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# Create tokenizer
def tokenizer(text):
      return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
      return [porter.stem(word) for word in text.split()]

# Set stop words
nltk.download('stopwords')
stop = stopwords.words('english')

# Model (logistic regression) evaluation
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [
   {
      'vect__ngram_range': [(1, 1)],
      'vect__stop_words': [stop, None],
      'vect__tokenizer': [tokenizer, tokenizer_porter],
      'clf__penalty': ['l1', 'l2'],
      'clf__C': [1.0, 10.0, 100.0]
   },
   {
      'vect__ngram_range': [(1, 1)],
      'vect__stop_words': [stop, None],
      'vect__tokenizer': [tokenizer, tokenizer_porter],
      'vect__use_idf':[False],
      'vect__norm':[None],
      'clf__penalty': ['l1', 'l2'],
      'clf__C': [1.0, 10.0, 100.0]
   }
]

lr_tfidf = Pipeline([
   ('vect', tfidf),
   ('clf', LogisticRegression(random_state=0, solver='liblinear'))
])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

# Fit the model
gs_lr_tfidf.fit(X_train, y_train)

# Best parameters
print(gs_lr_tfidf.best_params_)