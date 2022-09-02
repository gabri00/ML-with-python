import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyprind

df = pd.read_csv('movie_data.csv', encoding='utf-8')

pbar = pyprind.ProgBar(13)
pbar.update()

# Create a bag-of-words model
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
X = count.fit_transform(df['review'].values)
pbar.update()

# Fit a LDA estimator to the bag-of-words matrix
lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)
pbar.update()

# Show the topics
n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
   pbar.update()
   print("Topic %d:" % (topic_idx))
   print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))