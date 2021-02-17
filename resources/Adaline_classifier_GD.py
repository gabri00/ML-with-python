import numpy as np
import pandas
import matplotlib.pyplot as plt

from plotting.plot_decision_regions import plot_decision_regions

"""
Implementation of the adaptive linear neuron (aka ADALINE) single layer Neural Network
The weights are updated by minimizing the cost function via Gradient Descent
    X -> features
    y[] -> output
    activation function -> phi(z) = z
    threshold function -> sgn{z}
"""

class AdalineGD(object):
    """
    ADAptive LInear NEuron classifier:
	    Parameters
            - eta : (float) is the learning rate (0.0 < eta < 1.0)
            - n_iter : (int) number of iterations over the training dataset
            - random_state_seed : (int) random number generator seed for random weights initialization
	    Attributes
	    	- w_ : (1D-array) weights after fitting
	    	- cost_ : (list) sum of squares cost function value in each epoch
    """
    def __init__(self, eta=0.01, n_iter=50, random_state_seed=1):
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
            # Calculate the cost J(w) = 0.5 * sum(yi - phi(zi))^2
            errors = y - output
            cost = (errors**2).sum() / 2.0
            # Calculate new weights: w = w + delta_w,
            # where delta_w = -eta*∇J(w) = eta*sum(yi - phi(zi))*xi
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        # Calculate network input as the dot product of w and X (no need to transpose here)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        # Compute linear activation, phi(z)
        return X

    def predict(self, X):
        # Return class label after passing through sign function
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# Iris dataset from https://archive.ics.uci.edu\ml\machine-learning-databases\iris\iris.data
df = pandas.read_csv('data/iris.data', header=None, encoding='utf-8')

y = df.iloc[0:100, 4].values            # Select only Setosa and Versicolor => select the first 100 lines
y = np.where(y == 'Iris-setosa', -1, 1) # Set Setosa to -1 and Versicolor to 1
X = df.iloc[0:100, [0, 2]].values       # Extract sepal length (pos 0) and petal length (pos 2)

# Plot the cost against the number of epochs
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# With eta=0.01
ada1 = AdalineGD(n_iter=15, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(cost)')
ax[0].set_title('ADALINE - Learning rate: 0.01')
# With eta=0.0001
ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Cost')
ax[1].set_title('ADALINE - Learning rate: 0.0001')

# plt.savefig('images/plots/adaline_with_2_learning_rates.png', dpi=300)
plt.show()


# Improve the gradient descent through feature scaling with standardization method
# Standardize features
X_std = np.copy(X)
for j in range(X_std.ndim):
    X_std[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std()

# Train model
ada_gd = AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)

# Plot decision regions
plot_decision_regions(X_std, y, classifier=ada_gd)

plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.title('ADALINE - Gradient Descent regions')

# plt.savefig('images/plots/adaline_gd_with_standardization.png', dpi=300)
plt.show()

# Plot cost per epoch
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('ADALINE_GD - Cost per epoch')

# plt.savefig('images/plots/adaline_gd_with_standardization_cost.png', dpi=300)
plt.show()