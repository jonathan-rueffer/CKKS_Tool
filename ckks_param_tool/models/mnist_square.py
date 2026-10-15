import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .wrapper import WrappedModel

class SquareNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):

        x = self.fc1(x)
        x = x * x

        x = self.fc2(x)
        x = x * x

        x = self.fc3(x)

        return x

def build_wrapped_mnist_model(model):

    W1 = model.fc1.weight.detach().numpy().T
    b1 = model.fc1.bias.detach().numpy()

    W2 = model.fc2.weight.detach().numpy().T
    b2 = model.fc2.bias.detach().numpy()

    W3 = model.fc3.weight.detach().numpy().T
    b3 = model.fc3.bias.detach().numpy()

    def plain_forward(x):

        x = np.asarray(x)

        z1 = x @ W1 + b1
        a1 = z1 ** 2

        z2 = a1 @ W2 + b2
        a2 = z2 ** 2

        logits = a2 @ W3 + b3

        return logits

    def encrypted_forward_fn(x_enc):

        # layer 1
        z1 = x_enc.matmul(W1)
        z1 = z1 + b1

        z1.square_()

        # layer 2
        z2 = z1.matmul(W2)
        z2 = z2 + b2

        z2.square_()

        # output layer
        logits = z2.matmul(W3)
        logits = logits + b3

        return logits
    
    class PlainModel:

        def __call__(self, x):
            return plain_forward(x)

    return WrappedModel(
        plain_forward,
        depth=4,
        encrypted_forward_fn=encrypted_forward_fn
    )

def train_mnist_model(X, y):

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = SquareNet()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    epochs = 10

    for epoch in range(epochs):

        optimizer.zero_grad()

        logits = model(X_tensor)

        loss = criterion(
            logits,
            y_tensor
        )

        loss.backward()

        optimizer.step()

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss={loss.item():.4f}"
        )

    return build_wrapped_mnist_model(model)