"""
Perform image classification on cats and dogs dataset

References:
- https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
- https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch
import torch as T
import torch.nn.functional as F
import numpy as np
import pickle as pkl

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils, datasets

from batchnorm import CustomBatchNorm2D
import argparse

np.random.seed(47)
T.manual_seed(48)

def conv3x3(in_maps, out_maps, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_maps, out_maps, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_maps, out_maps, kernel_size=1, stride=stride, bias=False)


class CustomMomentum(object):
    def __init__(self, params, lr=0.001, momentum=0.9, nesterov=False):
        self.momenta = []
        self.params = []
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        for i, param in enumerate(params):
            self.momenta.append(T.zeros_like(param.data))
            self.params.append(param)

    def step(self):
        with T.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                moment = self.momenta[i]
                update = moment * self.momentum - self.lr*param.grad
                if self.nesterov:
                    param.data.add_(-self.momentum * self.momenta[i] + (1 + self.momentum)*update)
                else:
                    param.data.add_(update)
                self.momenta[i] = update

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill_(0)


class ResNetBlock(nn.Module):
    """ Basic block of ResNet 
    Attributes
    ----------
    in_maps : int
        Number of input feature maps
    out_maps : int
        Number of output feature maps
    conv1 : TODO Find its type
        A convolution operation
    bn1 : TODO Find its type
        Batch normalization for the first layer
    conv2 :
    bn2 :
    stride : int
        Stride to take during the first convolution layer in a block
    downsample :
        Operator to perform downsampling on the input by a factor of 2

    Methods
    -------
    forward(x)
        Propogates the tensor x through various transforms defined over it and
        returns the final value

    """

    def __init__(self, in_maps, out_maps, batchnorm=True):
        super(ResNetBlock, self).__init__()
        # We can determine the stride for the first convolution block and
        # whether to downsample x or not based on the in and out feature maps
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.batchnorm=batchnorm
        self.stride = 2 if self.in_maps != self.out_maps else 1
        self.conv1 = conv3x3(in_maps, out_maps, self.stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.out_maps, self.out_maps)
        if self.batchnorm:
            self.bn1 = CustomBatchNorm2D(self.out_maps)
            self.bn2 = CustomBatchNorm2D(self.out_maps)
        if self.in_maps != self.out_maps:
            self.downsample = conv1x1(self.in_maps, self.out_maps, 2)

    def forward(self, x):
        identity = x
        # Layer 1
        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)
        # Layer 2
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)

        # Downsample x if it the number of in and out feature maps are not same
        # for this block
        if self.in_maps != self.out_maps:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ Residual Neural Network Class

    Attributes
    ----------
    in_maps : int
        Number of input feature maps
    out_maps : int
        Number of output feature maps
    conv1 : TODO Find its type
        A convolution operation
    bn1 : TODO Find its type
        Batch normalization for the first layer
    relu : TODO
        Rectified Linear Unit operator
    stage : dict
        Store composite neural operations required to perform per stage
    avgpool : TODO
        Operator to calculate the average value of each feature map
    fc1 :
        Parameters for first fully connected layer at the end
    fc2 :
        Parameters for second fully connected layer at the end

    Methods
    -------
    _make_stage(block, in_maps, out_maps, num_blocks)
        Create a stage containing residual blocks

    forward(x)
        Propogates the tensor x through various transforms defined over it and
        returns the final value
    """

    def __init__(self, block, stages, num_classes=2, batchnorm=True):
        super(ResNet, self).__init__()
        self.in_maps = 64
        self.out_maps = 64
        self.batchnorm = batchnorm
        # First layer to convert 3x64x64 input to 64x56x56 output
        self.conv1 = nn.Conv2d(3, self.in_maps, kernel_size=9, bias=False)
        if batchnorm:
            self.bn1 = CustomBatchNorm2D(64)
        self.relu = nn.ReLU(inplace=True)

        self.stage = nn.ModuleList()
        for idx, num_stages in enumerate(stages):
            self.stage.append(self._make_stage(
                block, self.in_maps, self.out_maps, num_stages, batchnorm=batchnorm))
            self.in_maps = self.out_maps
            self.out_maps *= 2
        print([k for k in self.stage])

        # Average the over the values of each feature map
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _make_stage(self, block, in_maps, out_maps, num_blocks, batchnorm=True):
        """Append a stage of blocks to the ResNet"""
        blocks = list()
        blocks.append(block(in_maps, out_maps, batchnorm=batchnorm))
        for _ in range(1, num_blocks):
            blocks.append(block(out_maps, out_maps, batchnorm=batchnorm))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)  # 3x64x64
        if self.batchnorm:
            x = self.bn1(x)  # 64x56x56
        x = self.relu(x)

        for stage in self.stage:
            x = stage(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def evaluate(model, loader, device, name="Valid", iterations=1, mode="argmax",):
    with T.no_grad():
        all_preds = []
        loss = 0.
        count = 0.
        for i in range(iterations):
            predictions = []
            labs = []
            # model.eval()
            for idx, data in enumerate(loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Generate outputs
                outputs = model(inputs)
                loss += F.cross_entropy(outputs, labels, reduction="sum").detach().cpu().numpy()
                count += len(inputs)

                outputs = outputs.detach().cpu()

                predictions.append(outputs)
                labs.append(labels.cpu())

            preds = T.cat(predictions, 0)
            labels = T.cat(labs, 0)
            all_preds.append(preds)

        preds = T.stack(all_preds, -1)
        if mode == "mean":
            preds = T.mean(preds, -1)
        elif mode == "argmax":
            preds = T.max(preds, -1)[0]
        preds = T.max(preds, -1)[1]

        acc = preds.eq(labels).sum().float()/len(labels)
        loss = loss/count

        print("{} Evaluation: acc {}, loss {}".format(name, acc, loss))

    return acc, preds, loss

def train(params):
    device = torch.device('cuda:{}'.format(params.cuda) if params.cuda != -1 and torch.cuda.is_available() else 'cpu')
    print("Running the code on {}".format(device))

    if params.no_augmentation:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif params.extra_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(.8, 1.2), shear=20, resample=False, fillcolor=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(.2, .2, .2, .2),
            transforms.RandomGrayscale(.1),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(.8, 1.2), shear=20, resample=False, fillcolor=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root='trainset',
                                         transform=train_transform)

    valid_dataset = datasets.ImageFolder(root='trainset',
                                         transform=test_transform)

    indices = np.arange(len(train_dataset))  # start with all the indices in training set
    split = 1000  # define the split size

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(train_dataset,
                              batch_size=128,
                              num_workers=4,
                              sampler=train_sampler)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=256,
                              sampler=validation_sampler)

    test_dataset = datasets.ImageFolder(root="testset", transform=test_transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=256,
                             shuffle=False)


    epochs = params.epochs
    # Follows an architecture similar to ResNet18
    model = ResNet(ResNetBlock,
                   [params.layers,
                    params.layers,
                    params.layers,
                    params.layers],
                   batchnorm=(not params.no_batchnorm))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model has {} trainable parameters".format(num_params))
    try:
        model.load_state_dict(T.load(params.load))
    except Exception as e:
        print("Couldn't load saved model: {}".format(e))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = CustomMomentum(model.parameters(), lr=params.lr, momentum=params.momentum, nesterov=params.nesterov)

    best_acc = 0
    accs = []
    valid_accs = []
    losses = []
    valid_losses = []
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = []
        running_acc = 0
        full_acc = 0
        full_count = 0
        full_loss = 0
        count = 0
        if epoch % params.lrstepfreq == 0:
            print("Annealing LR from {} to {}".format(optimizer.lr, optimizer.lr*params.lrstep))
            optimizer.lr = optimizer.lr * params.lrstep
        for idx, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Generate outputs
            outputs = model(inputs)

            preds = T.max(outputs, -1)[1]
            acc = preds.eq(labels).sum()
            count += len(labels)
            running_acc += acc.cpu().detach().numpy()

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss.append(loss.detach().cpu().numpy())

            # Calculate the gradient of parameter w.r.t loss
            loss.backward()

            # Update the parameters
            optimizer.step()

            if idx % params.log_interval == 0:
                acc = running_acc/count
                full_acc += running_acc
                full_count += count
                loss = np.mean(running_loss)
                running_acc = 0
                full_loss += loss * count
                running_loss = []
                count = 0
                print("Epoch {} batch {} loss {} acc {}".format(epoch, idx, loss, acc))

        accs.append(full_acc/full_count)
        losses.append(full_loss/full_count)
        # _, _ = evaluate(model, averaged_valid_loader, iterations=10, name="Averaged Valid")
        acc, _, valid_loss = evaluate(model, valid_loader, device)
        valid_accs.append(acc.detach().cpu().numpy())
        valid_losses.append(valid_loss)
        if acc > best_acc:
            best_acc = acc
            print("New checkpoint at {}: best validation accuracy: {}".format(epoch, best_acc))
            T.save(model.state_dict(), params.save)

    with open("{}_log.pkl".format(params.name), "wb") as f:
        pkl.dump((accs, valid_accs), f)

    with open("{}_log_losses.pkl".format(params.name), "wb") as f:
        pkl.dump((losses, valid_losses), f)

    model.load_state_dict(T.load(params.save))
    predictions = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            outputs = model(data)
            predictions.append(outputs.cpu())

    # Reorder the images out of the lexical sort the torch loader uses by default
    # Failing to do this screws everything up horribly
    numbers = list(range(1, len(test_dataset) + 1))
    strs = [str(number) for number in numbers]
    sorted_numbers = [int(x) for x in sorted(strs)]
    indices = np.argsort(sorted_numbers)

    predictions = T.cat(predictions, 0)
    prob_cat = F.softmax(predictions, -1)[:, 0].cpu().detach().numpy()
    classes = T.max(predictions, -1)[1]
    sorted_preds = []
    with open("./{}.csv".format(params.name), "w") as f:
        f.write("id,label\n")
        for i in indices:
            sorted_preds.append(prob_cat[i])
            c = train_dataset.classes[classes[i]]
            f.write("{},{}\n".format(sorted_numbers[i], c))

    with open("{}_preds.pkl".format(params.name), "wb") as f:
        pkl.dump(sorted_preds, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Variational Graph Memory')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument("--lrstep", type=float, default=0.8, help="Factor to decrease LR")
    parser.add_argument("--lrstepfreq", type=int, default=10, help="Frequency with which to decrease LR.")
    parser.add_argument('--cuda', type=int, default=0, help='Cuda GPU ID, -1 for CPU')
    parser.add_argument('--log_interval', type=int, default=10, metavar='L',  help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt',  help='path to save the final model')
    parser.add_argument('--name', type=str,  default='predictions.csv',  help='path to save the test predictions')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--layers', type=int, default=4, help='Layers per resnet block.')
    parser.add_argument('--no_batchnorm', action='store_true', help="Disable custom batchnorm")
    parser.add_argument('--no_augmentation', action='store_true', help="Disable data augmentation")
    parser.add_argument('--extra_augmentation', action='store_true', help="Use even more data augmentation")
    parser.add_argument('--nesterov', action='store_true', help="Use nesterov momentum")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum (0 = off)")
    parser.add_argument('--load', type=str,  default=None,
                        help='path to load a model from to resume training.  Keep blank for none.')

    params = parser.parse_args()

    if params.cuda != -1:
        print('Using CUDA.')
        params.device="cuda"
    else:
        print('Using CPU.')
        params.device="cpu"

    train(params)
