import sys
sys.path.insert(1, '/Gabri/atom/ML-with-python/resources/')

import pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from functions.plot_decision_regions import plot_decision_regions


# Import dataframe
df_wine = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)


# Separate train and test datasets
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# Standardize the features
stds = StandardScaler()
X_train_std = stds.fit_transform(X_train)
X_test_std = stds.transform(X_test)


pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')


# Dimensionality reduction
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


# Fit the logistic regression model on the reduced dataset
lr.fit(X_train_pca, y_train)


# Plot decision regions
plot_decision_regions(X_train_pca, y_train, classifier=lr)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()