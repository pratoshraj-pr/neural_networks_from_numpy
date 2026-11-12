# torch_by_numpy

A neural network trained in pure NumPy, from backprop to BatchNorm and AdamW, that matches an identical PyTorch model step for step on MNIST (same initial weights, same batches, epoch train loss equal to ~1e-12, ~98% test accuracy).

This is the code from the blog post *Neural Networks from Scratch in NumPy, Verified Against PyTorch*. Every equation number in the code comments refers to that post.

## Files

| File | What's in it |
|---|---|
| `src/functional.py` | forward and backward of `linear`, `relu`, `softmax`, `cross_entropy`, `batch_norm`, and the L1/L2 penalties |
| `src/params.py` | `Parameters`, `get_dummy_inputs`, `kaiming_weight_init` |
| `src/records.py` | `LayerOutput`, `LayerGrad` (and their BatchNorm versions): what each layer stores in the forward pass and returns in the backward pass |
| `src/optim.py` | `sgd`, `adamw` |
| `src/data.py` | `make_spiral`, `split_train_test`, `data_iterator` |
| `src/gradcheck.py` | `numerical_grad`, `rel_error` (central finite differences) |
| `src/model.py` | the network, one feature added per class (below) |
| `src/tests/test_gradients.py` | finite-difference checks of every backward function and of the full networks |
| `src/train_spiral.py` | trains the full model on the 3-class spiral |
| `src/mnist_vs_torch.py` | trains the NumPy model and the same model in PyTorch on MNIST, and compares them epoch by epoch |
| `nn_from_scratch_numpy.ipynb` | the blog post as a notebook: math, code and experiments |

The model is built up one feature at a time, following the post:

```
NeuralNetwork            linear layers + ReLU, cross-entropy, SGD
└─ NeuralNetworkKaiming  Kaiming weight init, zero biases
   └─ NeuralNetworkWD    L2 / L1 weight decay
      └─ NeuralNetworkBN    BatchNorm before every hidden ReLU
         └─ NeuralNetworkAdamW  AdamW (or SGD via optimizer="sgd")
```

## Usage

```bash
pip install -r requirements.txt

python src/train_spiral.py        # ~seconds, numpy only
pytest                            # gradient checks in src/tests (pytest.ini puts src/ on the path)
python src/mnist_vs_torch.py      # downloads MNIST (~20 MB) on first run
```

```python
# from inside src/, or with src/ on PYTHONPATH
from data import make_spiral, split_train_test
from model import NeuralNetworkAdamW

train_X, train_Y, test_X, test_Y = split_train_test(*make_spiral())
model = NeuralNetworkAdamW([64, 32, 16, 8, 3])
history = model.train(train_X, train_Y, test_X, test_Y, optimizer="adamw", lr=0.003, epochs=50)
model.predict(test_X)
```

`train()` keyword arguments: `lr`, `epochs`, `batch_size`, `seed`, `log_every`, `weight_decay`, `penalty` (`"l2"`/`"l1"`, SGD only), `optimizer` (`"adamw"`/`"sgd"`), `betas`. Note that `train()` initialises the weights, so every call starts from scratch.
