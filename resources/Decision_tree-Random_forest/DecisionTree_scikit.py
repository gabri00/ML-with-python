import sys
sys.path.insert(1, '/Gabri/atom/ML-with-python/resources/')

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree

from sklearn.tree import export_graphviz
# from pydotplus import graph_from_dot_data

from functions.plot_decision_regions import plot_decision_regions
from functions.init_dataset import init_dataset

# Load iris data (load all 150 examples)
iris_data = datasets.load_iris()

# Separate train and test samples (train: 70%, test: 30% default) and apply feature standardization
X_train_std, X_test_std, y_train, y_test = init_dataset(dataset=iris_data)

# Decision Tree using Gini impurity method for Information Gain
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train_std, y_train)

# Put together train and test data
X_std = np.vstack((X_train_std, X_test_std))
y = np.hstack((y_train, y_test))

# Plot decision regions
plot_decision_regions(X=X_std,
                      y=y,
                      classifier=tree_model,
                      test_idx=range(105, 150))

plt.title('Decision Tree - Scikit')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Display tree graph using sklearn
# tree.plot_tree(tree_model)
# plt.show()

# Display tree graph using graphviz
dot_data = export_graphviz(tree_model,
                           filled=True,
                           rounded=True,
                           label=all,
                           class_names=['Setosa', 'Versicolor', 'Virginica'],
                           feature_names=['Petal length', 'Petal width'],
                           out_file='decisionTree.dot')
# graph = graph_from_dot_data(dot_data)
# graph.write_png('images/tree.png')