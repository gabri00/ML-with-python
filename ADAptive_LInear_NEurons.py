import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Implementation of the adaptive linear neuron (aka ADALINE) single layer NN
# The weights are updated by minimizing the cost function via gradient descent

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
            # Calculate cost J(w) = 0.5 * sum(yi - phi(zi))^2
            errors = y - output
            cost = (errors**2).sum() / 2.0
            # Calculate new weights: w = w + delta_w,
            # where delta_w = -eta*∇J(w) = eta*sum(yi - phi(zi))*xi
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        # Calculate network input as the dot product of w and x (no need to transpose here)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        # Compute linear activation, phi(z). In Adaline phi(z) is the identity function, so phi(w^T*x) = w^T*x
        return X

    def predict(self, X):
        # Return class label after passing through sign function
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# Iris dataset from https://archive.ics.uci.edu\ml\machine-learning-databases\iris\iris.data
# Read-in the Iris data
df = pandas.read_csv('data/iris.data', header=None, encoding='utf-8')
# Plot the Iris data (select only Setosa and Versicolor => select only the first 100 lines)
y = df.iloc[0:100, 4].values
# Normalize Setosa to -1 and Versicolor to 1
y = np.where(y == 'Iris-setosa', -1, 1)
# Extract sepal length (pos 0) and petal length (pos 2)
X = df.iloc[0:100, [0, 2]].values


# Plot the cost against the number of epochs
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# With eta=0.01
ada1 = AdalineGD(n_iter=15, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(cost)')
ax[0].set_title('ADALINE - Learning rate: 0.01')
# with eta=0.0001
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

# ada_gd = AdalineGD(n_iter=15, eta=0.01)
# ada_gd.fit(X_std, y)

# plot_decision_regions(X_std, y, classifier=ada_gd)
# plt.title('Adaline - Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# # plt.savefig('images/02_14_1.png', dpi=300)
# plt.show()

# plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Sum-squared-error')

# plt.tight_layout()
# # plt.savefig('images/02_14_2.png', dpi=300)
# plt.show()



# # ## Large scale machine learning and stochastic gradient descent



# class AdalineSGD(object):
#     """ADAptive LInear NEuron classifier.

#     Parameters
#     ------------
#     eta : float
#       Learning rate (between 0.0 and 1.0)
#     n_iter : int
#       Passes over the training dataset.
#     shuffle : bool (default: True)
#       Shuffles training data every epoch if True to prevent cycles.
#     random_state : int
#       Random number generator seed for random weight
#       initialization.


#     Attributes
#     -----------
#     w_ : 1d-array
#       Weights after fitting.
#     cost_ : list
#       Sum-of-squares cost function value averaged over all
#       training examples in each epoch.


#     """
#     def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
#         self.eta = eta
#         self.n_iter = n_iter
#         self.w_initialized = False
#         self.shuffle = shuffle
#         self.random_state = random_state

#     def fit(self, X, y):
#         """ Fit training data.

#         Parameters
#         ----------
#         X : {array-like}, shape = [n_examples, n_features]
#           Training vectors, where n_examples is the number of examples and
#           n_features is the number of features.
#         y : array-like, shape = [n_examples]
#           Target values.

#         Returns
#         -------
#         self : object

#         """
#         self._initialize_weights(X.shape[1])
#         self.cost_ = []
#         for i in range(self.n_iter):
#             if self.shuffle:
#                 X, y = self._shuffle(X, y)
#             cost = []
#             for xi, target in zip(X, y):
#                 cost.append(self._update_weights(xi, target))
#             avg_cost = sum(cost) / len(y)
#             self.cost_.append(avg_cost)
#         return self

#     def partial_fit(self, X, y):
#         """Fit training data without reinitializing the weights"""
#         if not self.w_initialized:
#             self._initialize_weights(X.shape[1])
#         if y.ravel().shape[0] > 1:
#             for xi, target in zip(X, y):
#                 self._update_weights(xi, target)
#         else:
#             self._update_weights(X, y)
#         return self

#     def _shuffle(self, X, y):
#         """Shuffle training data"""
#         r = self.rgen.permutation(len(y))
#         return X[r], y[r]

#     def _initialize_weights(self, m):
#         """Initialize weights to small random numbers"""
#         self.rgen = np.random.RandomState(self.random_state)
#         self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
#         self.w_initialized = True

#     def _update_weights(self, xi, target):
#         """Apply Adaline learning rule to update the weights"""
#         output = self.activation(self.net_input(xi))
#         error = (target - output)
#         self.w_[1:] += self.eta * xi.dot(error)
#         self.w_[0] += self.eta * error
#         cost = 0.5 * error**2
#         return cost

#     def net_input(self, X):
#         """Calculate net input"""
#         return np.dot(X, self.w_[1:]) + self.w_[0]

#     def activation(self, X):
#         """Compute linear activation"""
#         return X

#     def predict(self, X):
#         """Return class label after unit step"""
#         return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)




# ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# ada_sgd.fit(X_std, y)

# plot_decision_regions(X_std, y, classifier=ada_sgd)
# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')

# plt.tight_layout()
# # plt.savefig('images/02_15_1.png', dpi=300)
# plt.show()

# plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Average Cost')

# plt.tight_layout()
# # plt.savefig('images/02_15_2.png', dpi=300)
# plt.show()




# ada_sgd.partial_fit(X_std[0, :], y[0])
