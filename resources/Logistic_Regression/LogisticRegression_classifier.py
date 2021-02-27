import sys
sys.path.insert(1, '/Gabri/atom/ML-with-python/resources/')

import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from functions.plot_decision_regions import plot_decision_regions

"""
Implementation of the Logistic Regression Classifier using Gradient Descent to minimize the cost. 
    X -> features
    y[] -> output
    activation function -> phi(z) = sigmoid(z)
    threshold function -> u(z)
"""

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier:
	    Parameters
            - eta : (float) is the learning rate (0.0 < eta < 1.0)
            - n_iter : (int) number of iterations over the training dataset
            - random_state_seed : (int) random number generator seed for random weights initialization
	    Attributes
	    	- w_ : (1D-array) weights after fitting
	    	- cost_ : (list) logistic cost function in each epoch
    """
    def __init__(self, eta=0.05, n_iter=100, random_state_seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state_seed = random_state_seed

    def fit(self, X, y):
        """
        Fit training data:
	        Parameters
	        	- X : (2D-array-like) training vectors of shape = [n_examples, n_features]
	        	- y : (1D-array-like) target values of shape = [n_examples]
	        Returns
	        	- self : (object)
        """
        rgen = np.random.RandomState(self.random_state_seed)
        # Apply a Gaussian Distribution to normalize weights
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            # Calculate new weights: w = w + delta_w,
            # where delta_w = -eta*∇J(w) = eta*sum(yi - phi(zi))*xi
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # Calculate the logistic cost, J(w) (wich is the log-likelihood l(w)) 
            cost = -y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        # Calculate network input as the dot product of w and X (no need to transpose here)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        # Compute logistic sigmoid activation, phi(z)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X):
        # Return class label after passing through sign function
        return np.where(self.net_input(X) >= 0.0, 1, 0) # or: return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# Iris dataset from https://archive.ics.uci.edu\ml\machine-learning-databases\iris\iris.data
df = pandas.read_csv('data/iris.data', header=None, encoding='utf-8')

y = df.iloc[0:100, 4].values            # Select only Setosa and Versicolor => select the first 100 lines
y = np.where(y == 'Iris-setosa', 0, 1)  # Set Setosa to 0 and Versicolor to 1
X = df.iloc[0:100, [2, 3]].values       # Extract sepal length (pos 0) and petal length (pos 2)

# Separete data for tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# Train model
lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state_seed=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

# Plot decision regions
plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)

plt.title('Logistic Regression (GD) - Decision regions')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()