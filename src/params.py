# parameter containers and initialisation
import numpy as np


def get_dummy_inputs(input_dim, seed=42, mean=0, std=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=input_dim, loc=mean, scale=std)


class Parameters:
    def __init__(self, n_in, n_out, mean=0, std=1, seed=42):
        self.n = (n_out, n_in) if n_in > 1 else n_out
        self.mean = mean
        self.std = std
        self.seed = seed
        self.weights = self.set_weights(self.n, mean, std, seed)
    
    @staticmethod
    def set_weights(n, mean, std, seed):
        wt = get_dummy_inputs(n, seed, mean, std)
        return wt

    def __repr__(self):
        return f"Parameter: w -> {self.n}; w_dist -> mean - {self.mean}, std - {self.std}; seed -> {self.seed}"


def kaiming_weight_init(n, mode='n_in', dist='normal'):
    n_in, n_out = n
    num, denom = 0, 1
    if dist == 'normal':
        num = 2
    elif dist == 'uniform':
        num = 6

    if mode == 'n_in':
        denom = n_in
    elif mode == 'n_out':
        denom = n_out

    std = np.sqrt(num/denom)   # Var(w) = num / fan (eq. 33); for 'uniform' this is the bound a of eq. (35)
    return std
