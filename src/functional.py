# forward and backward of every operation in the network
import numpy as np


def linear(x, w, b):
    # z = x @ W.T + b
    out = np.matmul(x, np.transpose(w)) + b   # (B, n_out)
    return out


def relu(x):
    return np.where(x > 0, x, 0)


def relu_backward(grad_out, x, out=None):
    grad = np.where(x > 0, 1, 0)
    grad = grad * grad_out
    return grad


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)


def softmax_backward(grad_out, x=None, out=None):
    B, S = out.shape
    J = out[:, :, None] * np.eye(S) - np.matmul(out[:, :, None], out[:, None, :])
    grad = J * grad_out[:, :, None]
    grad = grad.sum(axis=1)
    return grad


def cross_entropy(x, y):
    # x -> logits (B,L); y -> labels (B,)
    B = x.shape[0]
    x = x - x.max(axis=-1, keepdims=True)   # bound to prevent exp overflow (underflow persists but doesnt matter in log_p)
    log_p = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))   # B,L
    ce = log_p[np.arange(B), y].mean()   # mean over B elements
    return -ce


def cross_entropy_backward(grad_out, x, label, out=None):
    B = x.shape[0]
    x = x - x.max(axis=-1, keepdims=True)
    probs = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)   # B,L
    probs[np.arange(B), label] -= 1    # p-1 for correct label and p for incorrect label
    return (probs / B) * grad_out   # split the gradient across batch


def linear_backward(dz, a_prev):
    # z = a_prev @ W.T + b  →  dW = dz.T @ a_prev,  db = Σ_batch dz
    dw = np.matmul(np.transpose(dz), a_prev)   # (n_out, n_in), same shape as W
    db = dz.sum(axis=0)                        # (n_out,)
    return dw, db


def batch_norm(x, gamma, beta, eps=1e-5):
    # x -> (B, n): statistics over the batch axis, one per feature
    mu = x.mean(axis=0)                            # (n,)    eq. (40)
    var = x.var(axis=0)                            # (n,)    biased variance, eq. (40)
    x_hat = (x - mu) / np.sqrt(var + eps)          # (B, n)  eq. (41)
    out = gamma * x_hat + beta                     # (B, n)  eq. (42)
    return out, x_hat, mu, var


def batch_norm_backward(grad_out, x_hat, var, gamma, eps=1e-5):
    B = grad_out.shape[0]
    dgamma = (grad_out * x_hat).sum(axis=0)        # eq. (44)
    dbeta = grad_out.sum(axis=0)                   # eq. (44)
    dx_hat = grad_out * gamma                      # eq. (44)
    dx = (B * dx_hat - dx_hat.sum(axis=0) - x_hat * (dx_hat * x_hat).sum(axis=0)) / (B * np.sqrt(var + eps))   # eq. (45)
    return dx, dgamma, dbeta


def l2_penalty(w, lam):
    return 0.5 * lam * (w ** 2).sum()          # eq. (37)

def l2_penalty_backward(w, lam):
    return lam * w

def l1_penalty(w, lam):
    return lam * np.abs(w).sum()               # eq. (39)

def l1_penalty_backward(w, lam):
    return lam * np.sign(w)
