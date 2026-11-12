# train the NumPy network and an identical PyTorch network on MNIST, from the same initial weights on the same batches
# needs: pip install torch datasets
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset

from data import data_iterator
from model import NeuralNetworkAdamW

# load MNIST (downloads ~20 MB the first time, then reads from the local cache)
mnist = load_dataset("ylecun/mnist")

def to_numpy(split):
    d = mnist[split].with_format("numpy")
    x = np.asarray(d["image"], dtype=np.float32).reshape(len(d), -1) / 255.0    # (N, 784), pixels in [0, 1]
    y = np.asarray(d["label"])
    return x, y

mnist_train_X, mnist_train_Y = to_numpy("train")
mnist_test_X, mnist_test_Y = to_numpy("test")
pixel_mean, pixel_std = mnist_train_X.mean(), mnist_train_X.std()
mnist_train_X = (mnist_train_X - pixel_mean) / pixel_std     # standardise with training-set statistics
mnist_test_X = (mnist_test_X - pixel_mean) / pixel_std
print(mnist_train_X.shape, mnist_test_X.shape, "| digits per class (train):", np.bincount(mnist_train_Y))


# the full library on MNIST
mnist_layers = [512, 256, 10]
mnist_config = dict(lr=1e-3, epochs=10, batch_size=128, weight_decay=1e-4, optimizer="adamw", log_every=1)
numpy_model = NeuralNetworkAdamW(mnist_layers)
start = time.perf_counter()
numpy_history = numpy_model.train(mnist_train_X, mnist_train_Y, mnist_test_X, mnist_test_Y, **mnist_config)
numpy_seconds = (time.perf_counter() - start) / mnist_config["epochs"]
print(f"NumPy: {numpy_seconds:.2f} s per epoch")


# the identical network in PyTorch
class TorchMLP(torch.nn.Module):
    def __init__(self, sizes):                          # sizes = [784, 512, 256, 10]
        super().__init__()
        mods = []
        for i in range(len(sizes) - 1):
            mods.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                mods += [torch.nn.BatchNorm1d(sizes[i + 1], momentum=0.1, eps=1e-5), torch.nn.ReLU()]
        self.net = torch.nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

torch_model = TorchMLP([784] + mnist_layers).double()
init_w = NeuralNetworkAdamW.weight_init(mnist_layers, 784)      # our Kaiming initial weights
init_b = NeuralNetworkAdamW.bias_init(mnist_layers)
linears = [m for m in torch_model.net if isinstance(m, torch.nn.Linear)]
with torch.no_grad():
    for l_key, lin in enumerate(linears, start=1):
        lin.weight.copy_(torch.from_numpy(init_w[l_key].weights))     # both store W as (n_out, n_in)
        lin.bias.copy_(torch.from_numpy(init_b[l_key].weights))

decay = [lin.weight for lin in linears]
no_decay = [p for name, p in torch_model.named_parameters() if not (name.endswith("weight") and p.dim() == 2)]
torch_opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

def torch_eval(x, y):
    torch_model.eval()
    with torch.no_grad():
        logits = torch_model(torch.from_numpy(x).double())
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y)).item()
        acc = (logits.argmax(dim=-1).numpy() == y).mean()
    torch_model.train()
    return acc, loss

torch_history = defaultdict(list)
rng = np.random.default_rng(42)                        # the same seed train() uses -> the same batches
start = time.perf_counter()
for epoch in range(1, mnist_config["epochs"] + 1):
    epoch_loss, n_seen = 0.0, 0
    for xb, yb in data_iterator((mnist_train_X, mnist_train_Y), 128, shuffle=True, rng=rng):
        logits = torch_model(torch.from_numpy(xb).double())
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(yb))
        torch_opt.zero_grad()
        loss.backward()
        torch_opt.step()
        epoch_loss += loss.item() * len(yb)
        n_seen += len(yb)
    test_acc, test_loss = torch_eval(mnist_test_X, mnist_test_Y)
    torch_history["train_loss"].append(epoch_loss / n_seen)
    torch_history["test_acc"].append(test_acc)
    torch_history["test_loss"].append(test_loss)
    print(f"epoch {epoch:>4}/{mnist_config['epochs']} | train loss {epoch_loss / n_seen:.4f} | test loss {test_loss:.4f} acc {test_acc:.3f}")
torch_seconds = (time.perf_counter() - start) / mnist_config["epochs"]
print(f"PyTorch: {torch_seconds:.2f} s per epoch")


# side by side
print(f"{'epoch':>5s} | {'NumPy train loss':>16s} {'PyTorch':>9s} | {'NumPy test acc':>14s} {'PyTorch':>8s}")
for e in range(mnist_config["epochs"]):
    print(f"{e + 1:>5d} | {numpy_history['train_loss'][e]:16.5f} {torch_history['train_loss'][e]:9.5f} | "
          f"{numpy_history['test_acc'][e]:14.4f} {torch_history['test_acc'][e]:8.4f}")
print(f"largest difference in epoch train loss: {np.max(np.abs(np.array(numpy_history['train_loss']) - np.array(torch_history['train_loss']))):.1e}")
