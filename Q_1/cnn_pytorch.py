import torch as T
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Utilities as utils
import pickle as pkl
import sys

# Set the random seeds to get the same results everytime
np.random.seed(47)
T.manual_seed(48)

# Global variables
EPOCHS = 40

if T.cuda.is_available():
    device="cuda"
else:
    device="cpu"

# Data loaders
mnist = utils.dataLoader(batch_size=64)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.upscale = nn.Conv2d(64, 128, kernel_size=1, padding=0)


        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(3200, 10)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = x1 + x3
        x5 = F.max_pool2d(x4, 2)
        x5_residual = F.relu(self.upscale(x5))
        x6 = F.relu(self.conv4(x5_residual))
        x7 = F.relu(self.conv5(x6))
        x8 = x5_residual + x7
        x8 = F.max_pool2d(x8, 2)
        x8 = x8.view(x8.shape[0], -1)
        x9 = self.fc1(x8)
        return x9

# Model settings
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.1)
scheduler = T.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

pytorch_total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)


print('Number of trainable params is : {}'.format(pytorch_total_params))

def numpy_to_torch(X, Y):
    X = np.reshape(X, (X.shape[0], 1, 28, 28))
    # Convert to Torch tensors
    X = T.from_numpy(X)
    Y = T.from_numpy(Y)

    return X, Y


def evaluate(model, split="Validation"):
    with T.no_grad():
        # Validation loop
        mnist.ResetDataSplit(split=split)
        acc = 0
        count = 0
        loss = 0
        while mnist.IsNextBatchExists(split=split):
            X, Y = mnist.GetNextBatch(split=split)
            X, Y = numpy_to_torch(X, Y)
            X = X.to(device)
            Y = Y.to(device)

            Y_pred = model(X)
            loss += F.cross_entropy(Y_pred, Y, reduction="sum").cpu().numpy()
            classes = T.max(Y_pred, -1)[1].long()
            acc += classes.eq(Y).float().sum().cpu().numpy()
            count += len(X)

        return (acc/count, loss/count)



train_losses = []
val_losses = []
val_accs = []
train_accs = []
model = model.to(device)
for epoch in range(EPOCHS):
    scheduler.step()
    batch_train_loss = []
    batch_val_loss = []
    # Training loop
    mnist.ResetDataSplit(split='Train')

    accs = 0
    count = 0
    losses = 0
    while mnist.IsNextBatchExists(split='Train'):
        # print(mnist.current_batch_start_train)
        X, Y = mnist.GetNextBatch(split='Train')
        # print(mnist.current_batch_start_train)
        X, Y = numpy_to_torch(X, Y)

        if device == "cuda":
            X = X.cuda()
            Y = Y.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        losses += loss*len(X)
        count += len(X)
        classes = T.max(Y_pred, -1)[1].long()
        accs += classes.eq(Y).float().sum()
        loss.backward()
        optimizer.step()


        batch_train_loss.append(loss.item())
    acc = accs/count
    train_accs.append(acc)
    train_losses.append(losses/count)
    val_acc, val_loss = evaluate(model)
    val_accs.append(val_acc)
    val_losses.append(val_loss)
    print("Epoch {} \t Loss {} \t Acc {} \t Val Loss {} \t Val Acc {}".format(epoch,
                                                                              train_losses[-1],
                                                                              acc,
                                                                              val_losses[-1],
                                                                              val_acc))

with open("cnn_trace.pkl", "wb") as f:
    pkl.dump([train_accs, val_accs, train_losses, val_losses], f)

print("Test:")
print(evaluate(model, "Test"))
