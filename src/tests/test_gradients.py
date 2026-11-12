# finite-difference checks of every backward function and of the full networks
# run with: pytest (from the repo root)
import numpy as np

from functional import (linear, linear_backward, relu, relu_backward, softmax, softmax_backward, cross_entropy,
                        cross_entropy_backward, batch_norm, batch_norm_backward,
                        l1_penalty, l1_penalty_backward, l2_penalty, l2_penalty_backward)
from gradcheck import numerical_grad, rel_error
from data import make_spiral, split_train_test
from model import NeuralNetwork, NeuralNetworkBN


def test_relu_backward():
    rng = np.random.default_rng(0)
    x, G = rng.normal(size=(4, 5)), rng.normal(size=(4, 5))
    assert rel_error(relu_backward(G, x), numerical_grad(lambda x_: (relu(x_) * G).sum(), x)) < 1e-6


def test_linear_backward():
    rng = np.random.default_rng(1)
    X, W, b, G = rng.normal(size=(6, 4)), rng.normal(size=(3, 4)), rng.normal(size=3), rng.normal(size=(6, 3))
    dW, db = linear_backward(G, X)
    assert rel_error(dW, numerical_grad(lambda w_: (linear(X, w_, b) * G).sum(), W)) < 1e-6
    assert rel_error(db, numerical_grad(lambda b_: (linear(X, W, b_) * G).sum(), b)) < 1e-6
    assert rel_error(G @ W, numerical_grad(lambda x_: (linear(x_, W, b) * G).sum(), X)) < 1e-6


def test_softmax_backward():
    rng = np.random.default_rng(2)
    z, G = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    assert rel_error(softmax_backward(G, out=softmax(z)), numerical_grad(lambda z_: (softmax(z_) * G).sum(), z)) < 1e-6


def test_cross_entropy_backward():
    rng = np.random.default_rng(3)
    z, y = rng.normal(size=(8, 3)) * 3, rng.integers(0, 3, size=8)
    assert rel_error(cross_entropy_backward(1.0, z, y), numerical_grad(lambda z_: cross_entropy(z_, y), z)) < 1e-6


def test_penalty_backward():
    w = np.random.default_rng(6).normal(size=(4, 3))
    for f, f_back in [(l2_penalty, l2_penalty_backward), (l1_penalty, l1_penalty_backward)]:
        assert rel_error(f_back(w, 0.1), numerical_grad(lambda w_: f(w_, 0.1), w)) < 1e-6


def test_batch_norm_backward():
    rng = np.random.default_rng(7)
    x, G = rng.normal(size=(6, 4)) * 3 + 1, rng.normal(size=(6, 4))
    gamma, beta = rng.normal(size=4), rng.normal(size=4)
    _, x_hat, _, var = batch_norm(x, gamma, beta)
    dx, dgamma, dbeta = batch_norm_backward(G, x_hat, var, gamma)
    assert rel_error(dx, numerical_grad(lambda x_: (batch_norm(x_, gamma, beta)[0] * G).sum(), x)) < 1e-6
    assert rel_error(dgamma, numerical_grad(lambda g_: (batch_norm(x, g_, beta)[0] * G).sum(), gamma)) < 1e-6
    assert rel_error(dbeta, numerical_grad(lambda b_: (batch_norm(x, gamma, b_)[0] * G).sum(), beta)) < 1e-6


def spiral_batch(n):
    train_X, train_Y, _, _ = split_train_test(*make_spiral())
    return train_X[:n], train_Y[:n]


def loss_with(model, params, l_key, value, x, y):
    # loss with one parameter array temporarily replaced
    saved = params[l_key].weights
    params[l_key].weights = value
    try:
        return cross_entropy(model.forward_pass(x)[model.n_layers].z, y)
    finally:
        params[l_key].weights = saved


def test_network_backward():
    layers = [64, 32, 16, 8, 3]
    x, y = spiral_batch(8)
    model = NeuralNetwork(layers)
    model.weights = model.weight_init(layers, x.shape[-1])
    model.bias = model.bias_init(layers)
    grad = model.backward_pass(model.forward_pass(x), y)
    for l_key in range(1, model.n_layers + 1):
        for params, attr in [(model.weights, "dw"), (model.bias, "db")]:
            numeric = numerical_grad(lambda v: loss_with(model, params, l_key, v, x, y), params[l_key].weights)
            assert rel_error(getattr(grad[l_key], attr), numeric) < 1e-6


def test_batch_norm_network_backward():
    layers = [16, 8, 3]
    x, y = spiral_batch(16)
    model = NeuralNetworkBN(layers)
    model.weights = model.weight_init(layers, x.shape[-1])
    model.bias = model.bias_init(layers)
    model.bn_init()
    grad = model.backward_pass(model.forward_pass(x), y)
    for params, attr in [(model.weights, "dw"), (model.gamma, "dgamma"), (model.beta, "dbeta")]:
        for l_key in params:
            numeric = numerical_grad(lambda v: loss_with(model, params, l_key, v, x, y), params[l_key].weights)
            assert rel_error(getattr(grad[l_key], attr), numeric) < 1e-5
