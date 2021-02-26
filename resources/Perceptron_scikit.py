import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import Perceptron
# from sklearn.metrics import accuracy_score

from functions.plot_decision_regions import plot_decision_regions
from functions.init_dataset import init_dataset

# Load iris data (load all 150 examples)
iris_data = datasets.load_iris()

# Separate train and test samples (train: 70%, test: 30% default) and apply feature standardization
X_train_std, X_test_std, y_train, y_test = init_dataset(dataset=iris_data)

# Perceptron classifier
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

# Note the really low misclassification and accuracy of the training
print(f'Misclassified examples: {(y_test != y_pred).sum()}')
print(f'Accuracy: {ppn.score(X_test_std, y_test)}')   # or print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Put together train and test data
X_std = np.vstack((X_train_std, X_test_std))
y = np.hstack((y_train, y_test))

# Plot decision regions
plt.figure(figsize=(12, 8))
plot_decision_regions(X=X_std,
                      y=y,
                      classifier=ppn,
                      test_idx=range(105, 150))

plt.title('Perceptron classifier - scikit')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.show()