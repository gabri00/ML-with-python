import pandas

# Create a dataframe of categorical data
df = pandas.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
df.columns = ['color', 'size', 'price', 'classLabel']

print(df)


# Mapping the categorical string values (size) into integers
size_map = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_map)

print(df)

# Inverse mapping
inv_size_map = {v: k for k, v in size_map.items()}
df['size'] = df['size'].map(inv_size_map)

print(df)