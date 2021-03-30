import pandas
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Create a dataframe of categorical data
df = pandas.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
df.columns = ['color', 'size', 'price', 'classLabel']

X = df[['color', 'size', 'price']].values


# Perform one-hot encoding to map nominal features
color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

# May use ColumnTransformer to map columns in a multi-feature array


# one-hot encoding with pandas
print(pandas.get_dummies(df[['price', 'color', 'size']]))