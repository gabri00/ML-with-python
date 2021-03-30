import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create a dataframe of categorical data
df = pandas.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
df.columns = ['color', 'size', 'price', 'classLabel']


# Map class labels as integers
class_map = {label: index for index, label in enumerate(np.unique(df['classLabel']))}
df['classLabel'] = df['classLabel'].map(class_map)

print(df)

# Reverse class labels mapping
inv_class_map = {v: k for k, v in class_map.items()}
df['classLabel'] = df['classLabel'].map(inv_class_map)

print(df)

# Map class labels with scikit-learn
le = LabelEncoder()
y = le.fit_transform(df['classLabel'].values)

print(y)

# Reverse mapping with scikit-learn
le.inverse_transform(y)