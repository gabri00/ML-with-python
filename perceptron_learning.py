import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Implementation of the perceptron learning algorithm

# Perceptron model:
#   [1, x1, ..., xm] --> [w0, w1, ..., wm] --> z = w0 + x1w1 + ... + xmwm --> (threshold function) sgn{z} --> y: output & delta_w: update

class Perceptron(object):
    """
    Perceptron classifier:
        Parameters
            - eta : (float) is the learning rate (0.0 < eta < 1.0)
            - n_iter : (int) number of iterations over the training dataset
            - random_state_seed : (int) random number generator seed for random weights initialization
        Attributes
            - w_ : (1D-array) weights after fitting
            - errors_ : (list) number of updates in each epoch
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
        random_gen = np.random.RandomState(self.random_state_seed)
        # Apply a Gaussian Distribution to normalize weights
        self.w_ = random_gen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            curr_error = 0
            for xi, target in zip(X, y):
                # delta_wj (aka update) = eta * (yi - ypi) * xi
                update = self.eta * (target - self.predict(xi))
                # wj = wj + delta_wj
                self.w_[1:] += update * xi
                self.w_[0] += update
                curr_error += int(update != 0.0)
            self.errors_.append(curr_error)

        return self

    def net_input(self, X):
        # Calculate network input (ypi) as the dot product of w and x (no need to transpose here)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # Return class label (y) after passing through sign function
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Training the perceptron model on the Iris dataset from https://archive.ics.uci.edu\ml\machine-learning-databases\iris\iris.data

# Read-in the Iris data
df = pandas.read_csv('data/iris.data', header=None, encoding='utf-8')
# Plot the Iris data (select only Setosa and Versicolor => select only the first 100 lines)
y = df.iloc[0:100, 4].values
# Normalize Setosa to -1 and Versicolor to 1
y = np.where(y == 'Iris-setosa', -1, 1)
# Extract sepal length (pos 0) and petal length (pos 2)
X = df.iloc[0:100, [0, 2]].values

# Plot the iris data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', edgecolor='black', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')

plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

plt.savefig('images/plots/iris_data_plot.png', dpi=300)
plt.show()


# Train the perceptron model
perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(X, y)

# Plot the number of updates per iteration
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/plots/misclassification_errors_per_epoch_plot.png', dpi=300)
plt.show()

# Plotting decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Setup markers and colors
    markers = ('X', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)    # xx1.shape = 305x235

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap, antialiased=True)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class data
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

plot_decision_regions(X, y, classifier=perceptron)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/plots/decision_regions_plot.png', dpi=300)
plt.show()