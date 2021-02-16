import numpy as np
import pandas
import matplotlib.pyplot as plt

from plotting.plot_decision_regions import plot_decision_regions

# Large scale machine learning and stochastic gradient descent

class AdalineSGD(object):
    """
    ADAptive LInear NEuron classifier:
	    Parameters
            - eta : (float) is the learning rate (0.0 < eta < 1.0)
            - n_iter : (int) number of iterations over the training dataset
            - shuffle : (bool) shuffles training data every epoch to prevent cycles
            - random_state_seed : (int) random number generator seed for random weights initialization
	    Attributes
	    	- w_ : (1D-array) weights after fitting
	    	- cost_ : (list) sum of squares cost function value averaged over all training examples in each epoch
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state_seed=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
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
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []

            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))

            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        # Fit training data without reinitializing weights, fits only a individual training examples
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X, y):
        # Shuffle data
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        # Initialize weights to small random numbers
        self.rgen = np.random.RandomState(self.random_state_seed)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        # Apply Adaline learning rule to update the weights
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        # Calculate network input as the dot product of w and X (no need to transpose here)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        # Compute linear activation, phi(z). In Adaline phi(z) is the identity function
        return X

    def predict(self, X):
        # Return class label after passing through sign function
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# Show plots related to Stochastic Gradient Descent Adaline
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_sgd)

plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
# plt.savefig('images/plots/adaline_sgd_with_standardization_decision_regions.png', dpi=300)

plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
plt.show()
# plt.savefig('images/plots/adaline_sgd_with_standardization_cost.png', dpi=300)
# ada_sgd.partial_fit(X_std[0, :], y[0])