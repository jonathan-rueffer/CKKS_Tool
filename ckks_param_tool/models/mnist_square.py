import logging

import numpy as np
import tenseal as ts
import torch
import torch.nn as nn
import torch.optim as optim

from .wrapper import WrappedModel

logger = logging.getLogger(__name__)

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

    # In TenSEAL, adding a plaintext vector to a matmul result also writes into the slots beyond
    # the result, and a following matmul's rotations pull those values in (errors of order 1).
    # Biases that are followed by another matmul are therefore added as encrypted vectors, whose
    # unused slots are zero. The last bias is followed by no matmul and can stay plaintext.
    def encrypted_forward_fn(x_enc):
        context = x_enc.context()

        # layer 1
        z1 = x_enc.matmul(W1)
        z1 = z1 + ts.ckks_vector(context, b1.tolist())

        z1.square_()

        # layer 2
        z2 = z1.matmul(W2)
        z2 = z2 + ts.ckks_vector(context, b2.tolist())

        z2.square_()

        # output layer
        logits = z2.matmul(W3)
        logits = logits + b3

        return logits
    
    # TenSEAL rescales after every multiplication, including ciphertext-plaintext ones, so the
    # depth counts levels consumed: 3 matmuls + 2 squares = 5
    return WrappedModel(
        plain_forward,
        depth=5,
        encrypted_forward_fn=encrypted_forward_fn,
        task="multiclass",
    )

# trains the network with mini-batch Adam. Square activations can make gradients explode, so
# the gradient norm is clipped. Seeded for reproducibility.
def train_mnist_model(X, y, epochs: int = 10, batch_size: int = 128, lr: float = 1e-3, seed: int = 0):

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = SquareNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        logger.info("epoch %d/%d loss=%.4f train accuracy=%.4f", epoch + 1, epochs, loss_sum / total, correct / total)

    model.eval()
    return build_wrapped_mnist_model(model)
