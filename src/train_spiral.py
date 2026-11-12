# train the full network (Kaiming init + BatchNorm + AdamW) on the 3-class spiral
from data import make_spiral, split_train_test
from model import NeuralNetworkAdamW

layers = [64, 32, 16, 8, 3]

X, Y = make_spiral()
train_X, train_Y, test_X, test_Y = split_train_test(X, Y)

model = NeuralNetworkAdamW(layers)
history = model.train(train_X, train_Y, test_X, test_Y, optimizer="adamw", lr=0.003, epochs=50, log_every=10)
print(f"final test accuracy: {history['test_acc'][-1]:.3f}")
