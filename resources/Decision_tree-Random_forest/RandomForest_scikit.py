import sys
sys.path.insert(1, '/Gabri/atom/ML-with-python/resources/')

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from functions.plot_decision_regions import plot_decision_regions
from functions.init_dataset import init_dataset

# Load iris data (load all 150 examples)
iris_data = datasets.load_iris()

# Separate train and test samples (train: 70%, test: 30% default) and apply feature standardization
X_train_std, X_test_std, y_train, y_test = init_dataset(dataset=iris_data)

# Random Forest classifier (is an ensemble of Decision Trees) using Gini impurity method for Information Gain
forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train_std, y_train)

# Put together train and test data
X_std = np.vstack((X_train_std, X_test_std))
y = np.hstack((y_train, y_test))

# Plot decision regions
plot_decision_regions(X=X_std,
                      y=y,
                      classifier=forest,
                      test_idx=range(105, 150))

plt.title('Random Forest - Scikit')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()