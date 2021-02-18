import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
# from sklearn.metrics import accuracy_score

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

# Perceptron learning
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

# Note the really low misclassification and accuracy of the training
print(f'Misclassified examples: {(y_test != y_pred).sum()}')
print(f'Accuracy: {ppn.score(X_test_std, y_test)}')   # or print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Plot decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt.figure(figsize=(12, 8))

plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=ppn,
                      test_idx=range(105, 150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Perceptron classifier - scikit')
plt.show()