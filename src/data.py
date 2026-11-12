# the spiral dataset, train/test split and mini-batching
import numpy as np


def make_spiral(n_per_class=300, K=3, noise=0.2, turns=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * K, 2)); y = np.zeros(n_per_class * K, dtype=int)
    for k in range(K):
        idx = slice(n_per_class * k, n_per_class * (k + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(k * 2*np.pi/K, k * 2*np.pi/K + turns * 2*np.pi, n_per_class) \
            + rng.normal(0, noise, n_per_class)
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = k
    return X, y


def split_train_test(X, Y, split_ratio=0.8, seed=42):
    rnd = np.random.default_rng(seed=seed)
    idx = np.arange(X.shape[0])
    idx = rnd.choice(idx, size=idx.shape[0], replace=False)
    train_len = int(idx.shape[0] * split_ratio)
    train_x, test_x = X[idx[:train_len]], X[idx[train_len:]]
    train_y, test_y = Y[idx[:train_len]], Y[idx[train_len:]]
    return train_x, train_y, test_x, test_y


def data_iterator(data, batch, shuffle=False, rng=None):
    x, y = data
    idx = np.arange(len(x))                 # len(x), not len(data) -> data is the (x, y) tuple
    if shuffle:
        rng = rng if rng is not None else np.random.default_rng(seed=42)
        rng.shuffle(idx)                    # in-place, returns None
    for i in range(0, len(x), batch):
        b_idx = idx[i : i+batch]
        yield x[b_idx], y[b_idx]
