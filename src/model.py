# the network, one feature per class:
# NeuralNetwork -> NeuralNetworkKaiming -> NeuralNetworkWD -> NeuralNetworkBN -> NeuralNetworkAdamW
from collections import defaultdict

import numpy as np

from functional import (linear, relu, relu_backward, cross_entropy, cross_entropy_backward, linear_backward,
                        batch_norm, batch_norm_backward, l1_penalty_backward, l2_penalty_backward)
from params import Parameters, kaiming_weight_init
from records import LayerOutput, LayerGrad, LayerOutputBN, LayerGradBN
from optim import sgd, adamw
from data import data_iterator


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        self.weights = None
        self.bias = None
        self.lr = None
        self.batch = None


    @staticmethod
    def weight_init(layers, feat_in, mean=0, std=1, seed_factor=42):
        weights = {}
        for l_i in range(len(layers)):
            seed = seed_factor + l_i
            if l_i == 0:
                weights[l_i + 1] =  Parameters(n_in=feat_in, n_out=layers[l_i], mean=mean, std=std, seed=seed)
            else:
                weights[l_i + 1] =  Parameters(n_in=layers[l_i - 1], n_out=layers[l_i], mean=mean, std=std, seed=seed)
        return weights


    @staticmethod
    def bias_init(layers, mean=0, std=1, seed_factor=1042):
        bias = {}
        for l_i in range(len(layers)):
            seed = seed_factor + l_i
            bias[l_i + 1] =  Parameters(n_in=1, n_out=layers[l_i], mean=mean, std=std, seed=seed)
        return bias


    def update_parameters(self, grad):
        # layers are keyed 1..n_layers (0 is the input)
        for l_key in range(1, self.n_layers + 1):
            self.weights[l_key].weights = sgd(self.weights[l_key].weights, grad[l_key].dw, self.lr)
            self.bias[l_key].weights = sgd(self.bias[l_key].weights, grad[l_key].db, self.lr)
        

    def forward_pass(self, x):
        nn_layer_out = {0: LayerOutput(z=x, act=x, nm="input")}
        
        for l_i in range(self.n_layers):
            l_key = l_i + 1
            w = self.weights[l_key].weights
            x = nn_layer_out[l_key - 1].activation
            b = self.bias[l_key].weights
            z = linear(x, w, b)
            if l_key < self.n_layers:
                out = relu(z)
                nn_layer_out[l_key] = LayerOutput(z=z, act=out, nm=f"{l_key}_output")
            else:
                nn_layer_out[l_key] = LayerOutput(z=z, nm=f"{l_key}_output")
        return nn_layer_out


    def backward_pass(self, nn_layer_out, y):
        nn_layer_grad = {}
        l_i = self.n_layers

        while l_i > 0:
            z = nn_layer_out[l_i].z               # this layer's pre-activation
            a_prev = nn_layer_out[l_i - 1].activation  # input to this layer
            d_act  = np.array([])

            if l_i == self.n_layers:
                # output layer: loss → logits
                dz = cross_entropy_backward(1.0, z, y)                       # (B, n_out)
            else:
                # hidden layer: next layer's dz → this layer's activation → z
                W_next = self.weights[l_i + 1].weights                       # (n_next, n_out)
                d_act  = np.matmul(nn_layer_grad[l_i + 1].dz, W_next)        # (B, n_out)
                dz     = relu_backward(d_act, z)                             # (B, n_out)

            dw, db = linear_backward(dz, a_prev)
            nn_layer_grad[l_i] = LayerGrad(dw, db, dz, d_act, nm=f"{l_i}_gradients")
            l_i -= 1
        
        return nn_layer_grad


    def predict(self, x):
        logits = self.forward_pass(x)[self.n_layers].z
        return logits.argmax(axis=-1)


    def evaluate(self, x, y):
        # full-set forward pass, no weight update
        logits = self.forward_pass(x)[self.n_layers].z
        loss = cross_entropy(logits, y)
        acc = (logits.argmax(axis=-1) == y).mean()
        return acc, loss


    def train_step(self, x, y):
        out = self.forward_pass(x)
        logits = out[self.n_layers].z
        loss = cross_entropy(logits, y)
        grad = self.backward_pass(out, y)
        self.update_parameters(grad)
        return logits, loss


    def train(self, train_x, train_y, test_x, test_y, **kwargs):
        self.lr = kwargs.get("lr", 0.001)
        self.batch = kwargs.get("batch_size", 32)
        
        wt_mean = kwargs.get("weights_mean", 0)
        wt_std = kwargs.get("weights_std", 1)
        epochs = kwargs.get("epochs", 5)
        log_every = kwargs.get("log_every", 1)
        rng = np.random.default_rng(kwargs.get("seed", 42))
        feat_in = train_x.shape[-1]
        
        self.weights = self.weight_init(self.layers, feat_in, wt_mean, wt_std)
        self.bias = self.bias_init(self.layers, wt_mean, wt_std)

        history = defaultdict(list)
        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_correct, n_seen = 0.0, 0, 0

            # new generator every epoch (a generator is exhausted after one pass); shared rng -> different shuffle each epoch
            for x, y in data_iterator((train_x, train_y), self.batch, shuffle=True, rng=rng):
                logits, loss = self.train_step(x, y)
                # loss is a batch mean -> weight by batch size so the smaller last batch counts correctly
                epoch_loss += loss * len(y)
                epoch_correct += (logits.argmax(axis=-1) == y).sum()
                n_seen += len(y)

            train_loss, train_acc = epoch_loss / n_seen, epoch_correct / n_seen
            test_acc, test_loss = self.evaluate(test_x, test_y)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            if epoch % log_every == 0 or epoch == epochs:
                print(f"epoch {epoch:>4}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | test loss {test_loss:.4f} acc {test_acc:.3f}")

            if not np.isfinite(train_loss):
                print(f"epoch {epoch}: loss is {train_loss}, stopping")
                break

        return history


class NeuralNetworkKaiming(NeuralNetwork):
    @staticmethod
    def weight_init(layers, feat_in, mean=0, std=1, seed_factor=42):
        # mean / std are ignored: each layer's std comes from its fan-in, eq. (33)
        weights = {}
        for l_i in range(len(layers)):
            seed = seed_factor + l_i
            n_in = feat_in if l_i == 0 else layers[l_i - 1]
            std_k = kaiming_weight_init((n_in, layers[l_i]))
            weights[l_i + 1] = Parameters(n_in=n_in, n_out=layers[l_i], mean=0, std=std_k, seed=seed)
        return weights


    @staticmethod
    def bias_init(layers, mean=0, std=1, seed_factor=1042):
        # biases start at exactly 0
        bias = {}
        for l_i in range(len(layers)):
            seed = seed_factor + l_i
            bias[l_i + 1] = Parameters(n_in=1, n_out=layers[l_i], mean=0, std=0, seed=seed)
        return bias


class NeuralNetworkWD(NeuralNetworkKaiming):
    def __init__(self, layers):
        super().__init__(layers)
        self.weight_decay = 0.0
        self.penalty = "l2"


    def train(self, train_x, train_y, test_x, test_y, **kwargs):
        self.weight_decay = kwargs.get("weight_decay", 0.0)     # lambda
        self.penalty = kwargs.get("penalty", "l2")             # "l2" or "l1"
        return super().train(train_x, train_y, test_x, test_y, **kwargs)


    def penalty_grad(self, w):
        if self.penalty == "l1":
            return l1_penalty_backward(w, self.weight_decay)
        return l2_penalty_backward(w, self.weight_decay)


    def update_parameters(self, grad):
        for l_key in range(1, self.n_layers + 1):
            w = self.weights[l_key].weights
            dw = grad[l_key].dw + self.penalty_grad(w)          # weights only, eqs. (38) and (39)
            self.weights[l_key].weights = sgd(w, dw, self.lr)
            self.bias[l_key].weights = sgd(self.bias[l_key].weights, grad[l_key].db, self.lr)   # biases: no decay


class NeuralNetworkBN(NeuralNetworkWD):
    def __init__(self, layers, momentum=0.1, eps=1e-5):
        super().__init__(layers)
        self.momentum = momentum
        self.eps = eps
        self.training = True
        self.gamma, self.beta = None, None
        self.running_mean, self.running_var = None, None


    def bn_init(self):
        # one (gamma, beta) pair and one set of running statistics per hidden layer
        hidden = range(1, self.n_layers)
        self.gamma = {l: Parameters(n_in=1, n_out=self.layers[l - 1], mean=1, std=0) for l in hidden}
        self.beta = {l: Parameters(n_in=1, n_out=self.layers[l - 1], mean=0, std=0) for l in hidden}
        self.running_mean = {l: np.zeros(self.layers[l - 1]) for l in hidden}
        self.running_var = {l: np.ones(self.layers[l - 1]) for l in hidden}


    def forward_pass(self, x):
        nn_layer_out = {0: LayerOutput(z=x, act=x, nm="input")}

        for l_i in range(self.n_layers):
            l_key = l_i + 1
            w = self.weights[l_key].weights
            x = nn_layer_out[l_key - 1].activation
            b = self.bias[l_key].weights
            z = linear(x, w, b)
            if l_key < self.n_layers:
                gamma, beta = self.gamma[l_key].weights, self.beta[l_key].weights
                if self.training:
                    z_bn, x_hat, mu, var = batch_norm(z, gamma, beta, self.eps)
                    m = self.momentum                                                   # eq. (43)
                    self.running_mean[l_key] = (1 - m) * self.running_mean[l_key] + m * mu
                    self.running_var[l_key] = (1 - m) * self.running_var[l_key] + m * var
                else:
                    var = self.running_var[l_key]
                    x_hat = (z - self.running_mean[l_key]) / np.sqrt(var + self.eps)
                    z_bn = gamma * x_hat + beta
                out = relu(z_bn)
                nn_layer_out[l_key] = LayerOutputBN(z=z, z_bn=z_bn, x_hat=x_hat, var=var, act=out, nm=f"{l_key}_output")
            else:
                nn_layer_out[l_key] = LayerOutput(z=z, nm=f"{l_key}_output")
        return nn_layer_out


    def backward_pass(self, nn_layer_out, y):
        nn_layer_grad = {}
        l_i = self.n_layers

        while l_i > 0:
            a_prev = nn_layer_out[l_i - 1].activation
            d_act, dgamma, dbeta = np.array([]), np.array([]), np.array([])

            if l_i == self.n_layers:
                # output layer: loss -> logits, no BatchNorm
                dz = cross_entropy_backward(1.0, nn_layer_out[l_i].z, y)
            else:
                # hidden layer: next layer's dz -> activation -> ReLU -> BatchNorm -> z
                out = nn_layer_out[l_i]
                W_next = self.weights[l_i + 1].weights
                d_act = np.matmul(nn_layer_grad[l_i + 1].dz, W_next)                   # eq. (25)
                dz_bn = relu_backward(d_act, out.z_bn)                                 # ReLU now sits after BN
                dz, dgamma, dbeta = batch_norm_backward(dz_bn, out.x_hat, out.var, self.gamma[l_i].weights, self.eps)

            dw, db = linear_backward(dz, a_prev)
            nn_layer_grad[l_i] = LayerGradBN(dw, db, dz, d_act, dgamma, dbeta, nm=f"{l_i}_gradients")
            l_i -= 1

        return nn_layer_grad


    def update_parameters(self, grad):
        super().update_parameters(grad)                     # weights (with weight decay) and biases
        for l_key in range(1, self.n_layers):
            self.gamma[l_key].weights = sgd(self.gamma[l_key].weights, grad[l_key].dgamma, self.lr)
            self.beta[l_key].weights = sgd(self.beta[l_key].weights, grad[l_key].dbeta, self.lr)


    def predict(self, x):
        self.training = False
        pred = super().predict(x)
        self.training = True
        return pred


    def evaluate(self, x, y):
        self.training = False
        result = super().evaluate(x, y)
        self.training = True
        return result


    def train(self, train_x, train_y, test_x, test_y, **kwargs):
        self.bn_init()
        return super().train(train_x, train_y, test_x, test_y, **kwargs)


class NeuralNetworkAdamW(NeuralNetworkBN):
    def __init__(self, layers, momentum=0.1, eps=1e-5):
        super().__init__(layers, momentum, eps)
        self.optimizer = "adamw"
        self.betas = (0.9, 0.999)
        self.state = {}       # (m, v) for every parameter array
        self.t = 0            # step count, for bias correction


    def train(self, train_x, train_y, test_x, test_y, **kwargs):
        self.optimizer = kwargs.get("optimizer", "adamw")      # "adamw" or "sgd"
        self.betas = kwargs.get("betas", (0.9, 0.999))
        self.state, self.t = {}, 0
        return super().train(train_x, train_y, test_x, test_y, **kwargs)


    def adamw_step(self, key, w, grad, weight_decay):
        m, v = self.state.get(key, (np.zeros_like(w), np.zeros_like(w)))
        w, m, v = adamw(w, grad, m, v, self.t, self.lr, *self.betas, weight_decay=weight_decay)
        self.state[key] = (m, v)
        return w


    def update_parameters(self, grad):
        if self.optimizer == "sgd":
            return super().update_parameters(grad)
        self.t += 1
        for l_key in range(1, self.n_layers + 1):
            self.weights[l_key].weights = self.adamw_step(("w", l_key), self.weights[l_key].weights, grad[l_key].dw, self.weight_decay)
            self.bias[l_key].weights = self.adamw_step(("b", l_key), self.bias[l_key].weights, grad[l_key].db, 0.0)
        for l_key in range(1, self.n_layers):
            self.gamma[l_key].weights = self.adamw_step(("gamma", l_key), self.gamma[l_key].weights, grad[l_key].dgamma, 0.0)
            self.beta[l_key].weights = self.adamw_step(("beta", l_key), self.beta[l_key].weights, grad[l_key].dbeta, 0.0)
