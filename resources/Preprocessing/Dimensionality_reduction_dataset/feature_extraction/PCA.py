# Reduce dimensionality of a dataset via feature extraction
# with the Principal Component Analysis (PCA) algorithm -> for unsupervised data compression

import pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Import dataframe
df_wine = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)


# Separate train and test datasets
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# Standardize the features
stds = StandardScaler()
X_train_std = stds.fit_transform(X_train)
X_test_std = stds.transform(X_test)


# Calculate the covariance matrix and calculate the eigenvectors and eigenvalues
cov_M = np.cov(X_train_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_M)


# Compute the Explained Variance Ratio
var_exp = [(i / sum(eig_vals)) for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Plot the individual explained variance and the cumulative explained variance
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative explained variance')

plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')

plt.tight_layout()
plt.show()


# Select the k eigenvectors which correspond to the k largest eigenvalues
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda k: k[0], reverse=True)
# Build a projection matrix (k=2)
W = np.hstack((eig_pairs[0][1][:, np.newaxis], eig_pairs[1][1][:, np.newaxis]))
# Transform the 13-dimensional dataset, X, using the projection matrix, W, to obtain the new 2-dimensional feature subspace
X_train_pca = X_train_std.dot(W)
# print(X_train_pca)

# Plot the new subspace
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==1, 0], X_train_pca[y_train==1, 1], c=c, label=l, marker=m)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()