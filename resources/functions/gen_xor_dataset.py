import numpy as np

def gen_xor_dataset(n=200, m=2):
    np.random.seed(1)
    X_xor = np.random.randn(n, m)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    return X_xor, y_xor