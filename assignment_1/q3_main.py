"""
Perform image classification on cats and dogs dataset

References:
- https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
- https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import os
import torch
import torch as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import csv

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils, datasets

from batchnorm import CustomBatchNorm2D

np.random.seed(47)
T.manual_seed(48)

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Running the code on {}".format(device))

train_transform = transforms.Compose([
    transforms.RandomAffine(20, translate=(
        0.1, 0.1), scale=(.8, 1.2), shear=20, resample=False, fillcolor=0),
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


sampled_valid_dataset = datasets.ImageFolder(root='trainset',
                                             transform=train_transform)

# start with all the indices in training set
indices = np.arange(len(train_dataset))
split = 1000  # define the split size

# Random, non-contiguous split
validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
validation_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)

train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          num_workers=4,
                          sampler=train_sampler)

valid_loader = DataLoader(valid_dataset,
                          batch_size=256,
                          sampler=validation_sampler)


averaged_valid_loader = DataLoader(valid_dataset,
                                   batch_size=256,
                                   sampler=validation_sampler)

test_dataset = datasets.ImageFolder(root="testset", transform=test_transform)
test_loader = DataLoader(test_dataset,
                         batch_size=256,
                         shuffle=False)


def conv3x3(in_maps, out_maps, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_maps, out_maps, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_maps, out_maps, kernel_size=1, stride=stride, bias=False)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


test_dataset = ImageFolderWithPaths(
    root='testset', transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)


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

    def __init__(self, in_maps, out_maps):
        super(ResNetBlock, self).__init__()
        # We can determine the stride for the first convolution block and
        # whether to downsample x or not based on the in and out feature maps
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.stride = 2 if self.in_maps != self.out_maps else 1
        self.conv1 = conv3x3(in_maps, out_maps, self.stride)
        self.bn1 = CustomBatchNorm2D(self.out_maps)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.out_maps, self.out_maps)
        self.bn2 = CustomBatchNorm2D(self.out_maps)
        if self.in_maps != self.out_maps:
            self.downsample = conv1x1(self.in_maps, self.out_maps, 2)

    def forward(self, x):
        identity = x
        # Layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Layer 2
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample x if the number of in and out feature maps are not same
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

    def __init__(self, block, stages, num_classes=2):
        super(ResNet, self).__init__()
        self.in_maps = 64
        self.out_maps = 64
        # First layer to convert 3x64x64 input to 64x56x56 output
        self.conv1 = nn.Conv2d(3, self.in_maps, kernel_size=9, bias=False)
        self.bn1 = CustomBatchNorm2D(64)
        self.relu = nn.ReLU(inplace=True)

        self.stage = nn.ModuleList()
        for idx, num_stages in enumerate(stages):
            self.stage.append(self._make_stage(
                block, self.in_maps, self.out_maps, num_stages))
            self.in_maps = self.out_maps
            self.out_maps *= 2

        # Average the over the values of each feature map
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _make_stage(self, block, in_maps, out_maps, num_blocks):
        """Append a stage of blocks to the ResNet"""
        blocks = list()
        blocks.append(block(in_maps, out_maps))
        for _ in range(1, num_blocks):
            blocks.append(block(out_maps, out_maps))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)  # 3x64x64
        x = self.bn1(x)  # 64x56x56
        x = self.relu(x)

        for stage in self.stage:
            x = stage(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def evaluate(model, loader, name="Valid", iterations=1, mode="argmax"):
    with T.no_grad():
        all_preds = []
        for i in range(iterations):
            predictions = []
            labs = []
            # model.eval()
            for idx, data in enumerate(loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Generate outputs
                outputs = model(inputs).detach().cpu()

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

        print("{} Evaluation: Acc {}".format(name, acc))

    return acc, preds


epochs = 100
# Follows an architecture similar to ResNet18
model = ResNet(ResNetBlock, [4, 4, 4, 4])
try:
    model.load_state_dict(T.load("./resnet18.pt"))
except Exception as e:
    print("Couldn't load saved model: {}".format(e))

model = model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0
for epoch in range(epochs):
    model.train()
    running_loss = []
    running_acc = []
    count = 0
    for idx, data in enumerate(train_loader, 0):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = ['Cat' if output == 0 else 'Dog' for output in outputs]

        # Generate outputs
        outputs = model(inputs)

        preds = T.max(outputs, -1)[1]
        acc = preds.eq(labels).sum()
        count += len(labels)
        running_acc.append(acc.detach().cpu().numpy())

        # Calculate loss
        loss = criterion(outputs, labels)
        running_loss.append(loss.detach().cpu().numpy())


def main():
    epochs = 1
    # Follows an architrcture similar to ResNet18
    model = ResNet(ResNetBlock, [2, 2, 2, 2])
    model = model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = train(epochs, model, criterion, optimizer)
    test(model)
    # test()

    if idx % 10 == 0:
            acc = np.sum(running_acc)/count
            loss = np.mean(running_loss)
            running_acc = []
            running_loss = []
            count = 0
            print("Epoch {} batch {} loss {} acc {}".format(epoch, idx, loss, acc))

    # _, _ = evaluate(model, averaged_valid_loader, iterations=10, name="Averaged Valid")
    acc, _ = evaluate(model, valid_loader)
    if acc > best_acc:
        best_acc = acc
        print("New checkpoint at {}: best validation accuracy: {}".format(
            epoch, best_acc))
        T.save(model.state_dict(), './resnet18.pt')


model.load_state_dict(T.load("./resnet18.pt"))
predictions = []
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        outputs = model(data)
        predictions.append(outputs.cpu())

# Reorder the images out of the lexical sort
numbers = list(range(1, len(test_dataset) + 1))
strs = [str(number) for number in numbers]
sorted_numbers = [int(x) for x in sorted(strs)]
indices = np.argsort(sorted_numbers)

predictions = T.cat(predictions, 0)
classes = T.max(predictions, -1)[1]
with open("./Test_submission.csv", "w") as f:
    f.write("id, label \n")
    for i in indices:
        c = train_dataset.classes[classes[i]]
        print("{}, {} \n".format(sorted_numbers[i], c))
        f.write("{}, {} \n".format(sorted_numbers[i], c))
