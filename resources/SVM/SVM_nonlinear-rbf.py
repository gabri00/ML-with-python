import sys
sys.path.insert(1, '/Gabri/atom/ML-with-python/resources/')

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC

from functions.gen_xor_dataset import gen_xor_dataset
from functions.plot_decision_regions import plot_decision_regions

# Generate XOR gate dataset
X_xor, y_xor = gen_xor_dataset()

# Plot dataset
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b',
            marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            edgecolor='black',
            label='-1')

plt.legend(loc='best')

plt.tight_layout()
plt.show()


# SVM classifier with Gaussian kernel (RBF) for non-linearly separable data
svm = SVC(kernel='rbf', C=1.0, gamma=0.10, random_state=1)
svm.fit(X_xor, y_xor)

# Plot decision regions
plot_decision_regions(X=X_xor, y=y_xor, classifier=svm)

plt.title('SVM - linearly inseparable')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()