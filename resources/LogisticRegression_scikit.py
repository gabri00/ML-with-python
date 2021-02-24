import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from plotting.plot_decision_regions import plot_decision_regions

# Load iris data (load all 150 examples)
iris_data = datasets.load_iris()
X = iris_data.data[:, [2, 3]]
y = iris_data.target

# Separete data for tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Use StandardScaler class to standardize the feature
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Logistic Regression classifier
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

# Plot decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=lr,
                      test_idx=range(105, 150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Logistic Regression classifier - scikit')
plt.show()

# print(lr.predict_proba(X_test_std[:6, :]))
# print(lr.predict_proba(X_test_std[:6, :]).argmax(axis=1))