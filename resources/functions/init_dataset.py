from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def init_dataset(dataset, test_perc=0.3, split_seed=1):
    X = dataset.data[:, [2, 3]]
    y = dataset.target

    # Separete data for tests
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Use StandardScaler class to standardize the feature
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test