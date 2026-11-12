# parameter updates
import numpy as np


def sgd(w, grad, lr=0.001):
    w = w - lr * grad
    return w


def adamw(w, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    m = beta1 * m + (1 - beta1) * grad                     # first moment, eq. (50)
    v = beta2 * v + (1 - beta2) * grad ** 2                # second moment, eq. (50)
    m_hat = m / (1 - beta1 ** t)                           # bias correction, eq. (51)
    v_hat = v / (1 - beta2 ** t)
    w = w - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * w)    # eq. (54): decoupled weight decay
    return w, m, v
