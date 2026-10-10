"""Small, download-free task: scikit-learn's 8x8 digits (1797 samples, 10 classes)."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression


def digits(seed=0):
    d = load_digits()
    order = np.random.default_rng(seed).permutation(len(d.target))
    return d.images[order], d.target[order]


def shifted(images, dx=1, dy=1):
    """Deployment shift: the same digits, translated on the sensor."""
    return np.roll(images, (dy, dx), axis=(1, 2))


def binarize(images):
    return (images.reshape(len(images), -1) >= 8).astype(np.uint8)


def fit_linear(x, y):
    """Reference linear classifier. Returns W (10 x n), b (10)."""
    m = LogisticRegression(max_iter=2000, C=1.0).fit(x, y)
    return m.coef_, m.intercept_


def accuracy(scores, y):
    return float((scores.argmax(1) == y).mean())
