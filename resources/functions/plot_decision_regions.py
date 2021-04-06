import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Plotting decision regions
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Setup markers and colors
    markers = ('X', 'x', 'v', 'o', '^')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap, antialiased=True)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class data
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    c=colors[idx],
                    alpha=0.8,
                    marker=markers[idx],
                    edgecolor='black',
                    label=cl)
        
        # Highlight test examples
        if test_idx:
            plt.scatter(x=X[test_idx, 0],
                        y=X[test_idx, 1],
                        c='',
                        alpha=1.0,
                        marker='o',
                        edgecolor='yellow',
                        s=120,
                        label='test set')