"""
Perform image classification on cats and dogs dataset

References:
- https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
- https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import csv

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils, datasets

from batchnorm import CustomBatchNorm2D

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Running the code on {}".format(device))

# data_transform = transforms.Compose([
#     transforms.RandomResizedCrop(64),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(hue=.05, saturation=.05),
#     transforms.RandomRotation(20),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

data_transform = transforms.ToTensor()
train_dataset = datasets.ImageFolder(root='trainset',
                                     transform=data_transform)
validation_dataset = datasets.ImageFolder(root='validationset',
                                          transform=data_transform)

train_loader = DataLoader(train_dataset,
                          batch_size=4, shuffle=True,
                          num_workers=4)
validation_loader = DataLoader(
    validation_dataset, batch_size=4, num_workers=4)


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
        blocks = []
        blocks.append(block(in_maps, out_maps))
        for _ in range(1, num_blocks):
            blocks.append(block(out_maps, out_maps))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)  # 3x64x64
        x = self.bn1(x)  # 64x56x56
        x = self.relu(x)

        x = self.stage[0](x)
        x = self.stage[1](x)
        x = self.stage[2](x)
        x = self.stage[3](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def train(epochs, model, criterion, optimizer):
    """Train the model
    """
    train_loss = []
    val_loss = []
    for epoch in range(1, epochs+1):
        print("Epoch {}/{}".format(epoch, epochs))
        for phase in ['train', 'validate']:
            # Iterate over the training set learn the model params
            if phase == 'train':
                model.train()
                running_loss = []
                for idx, data in enumerate(train_loader):
                    # Get the inputs
                    inputs, labels = data

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Generate outputs
                    outputs = model(inputs.to(device))

                    # Calculate loss
                    loss = criterion(outputs, labels.to(device))

                    # appending the loss to list containing current losses
                    running_loss.append(loss)

                    # Calculate the gradient of parameter w.r.t loss
                    loss.backward()

                    # Update the parameters
                    optimizer.step()

                    print("Epoch {} batch {} loss {}".format(
                        epoch, idx, loss))

                train_loss.append(torch.FloatTensor(running_loss).mean())
                print("Train error in epoch {}: {}".format(
                    epoch, train_loss[-1]))
                torch.save(model.state_dict(), './model/resnet18.pt')

            # Evaluate the performance of the trained model on the validation set
            elif phase == 'validate':
                model.eval()
                running_loss = []
                for idx, data in enumerate(validation_loader):
                    # Get the inputs
                    inputs, labels = data

                    # Generate outputs
                    outputs = model(inputs.to(device))

                    # Calculate loss
                    loss = criterion(outputs, labels.to(device))

                    # appending the loss to list containing current losses
                    running_loss.append(loss)

                val_loss.append(torch.FloatTensor(running_loss).mean())
                print("Validation error in epoch {}: {}".format(
                    epoch, val_loss[-1]))

    return model


def test(model=None):
    """Evaluate the model on the test set
    """
    model.eval()
    predictions = [['id', 'label']]
    for data in test_loader:
        inputs, _, paths = data
        ids = [path.split("/")[-1].split(".")[0] for path in paths]

        outputs = model(inputs)

        outputs = ['Cat' if output == 0 else 'Dog' for output in outputs]

        predictions.extend([[_id, output] for _id, output in zip(ids, output)])

    print(predictions)
    with open('predictions.csv', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(predictions)


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


if __name__ == '__main__':
    main()
