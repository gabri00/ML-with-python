# Reduce dimensionality of a dataset via feature extraction
# with the Linear Discriminant Analysis (LDA) algorithm -> for supervised data compression for maximizing class separability

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


# Calculate the mean vectors
np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
	mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))

# Compute the within-class scatter matrix (Sw)
d = 134	# Dimensions
Sw = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
	class_scatter = np.cov(X_train_std[y_train==label].T)
	Sw += class_scatter

# Compute the between-class scatter matrix (Sb)
mean_overall = np.mean(X_train_std, axis=0)
Sb = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
	n = X_train_std[y_train == i + 1, :].shape[0]
	mean_vec = mean_vec.reshape(d, 1)	# To make a column vector
	mean_overall = mean_overall.reshape(d, 1)
	Sb += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

# Calculate the eigenvectors and eigenvalues of (Sw^-1).dot(Sb)
eig_vals, eig_vecs = np.linalg.eig(no.linalg.inv(Sw).dot(Sb))

# Select the k eigenvectors which correspond to the k largest eigenvalues
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)





# # Plot the new subspace
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']

# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train==l, 0],
#                 X_train_pca[y_train==l, 1],
#                 c=c, label=l, marker=m)

# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')

# plt.tight_layout()
# plt.show()