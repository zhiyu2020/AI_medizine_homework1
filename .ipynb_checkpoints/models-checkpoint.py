from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BrainAgeCNN(nn.Module):
    """
    The BrainAgeCNN predicts the age given a brain MR-image.
    """
    def __init__(self) -> None:
        super().__init__()

        # Feel free to also add arguments to __init__ if you want.
        # ----------------------- ADD YOUR CODE HERE --------------------------
        layers = [2,2,2,2]
        block_inplanes = [64, 128, 256, 512]
        self.conv1 = nn.Conv3d(1,
                               64,
                               kernel_size=(7, 7, 7),
                               stride=(3, 3, 3),
                               bias=False)
        self.in_planes = block_inplanes[0]
        self.max_3d = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(1024)
        self.drop=nn.Dropout(p=0.5)

        self.layer1 = self._make_layer(BasicBlock, block_inplanes[0], layers[0])
        self.layer2 = self._make_layer(BasicBlock,
                                       block_inplanes[1],
                                       layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(BasicBlock,
                                       block_inplanes[2],
                                       layers[2],
                                       stride=2)
        self.layer4 = self._make_layer(BasicBlock,
                                       block_inplanes[3],
                                       layers[3],
                                       stride=2)

        # ------------------------------- END ---------------------------------

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def _make_layer(self, block, planes, blocks, shortcut_type = 'B', stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'B':
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Forward pass of your model.

        :param imgs: Batch of input images. Shape (N, 1, H, W, D)
        :return pred: Batch of predicted ages. Shape (N)
        """
        # ----------------------- ADD YOUR CODE HERE --------------------------
        pred = None
        ####

        x = self.conv1(imgs)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.max_3d(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch(x)
        x = self.drop(x)
        x = self.fc2(x)
        pred = x
        ####
        # imgs = self.conv_layer1(imgs)
        # imgs = self.conv_layer2(imgs)
        # imgs = imgs.view(imgs.size(0), -1)
        # imgs = self.fc1(imgs)
        # imgs = self.relu(imgs)
        # imgs = self.batch(imgs)
        # imgs = self.drop(imgs)
        # imgs = self.fc2(imgs)
        # pred = imgs
        # ------------------------------- END ---------------------------------
        return pred

    def train_step(
        self,
        imgs: Tensor,
        labels: Tensor,
        return_prediction: Optional[bool] = False
    ):
        """Perform a training step. Predict the age for a batch of images and
        return the loss.

        :param imgs: Batch of input images (N, 1, H, W, D)
        :param labels: Batch of target labels (N)
        :return loss: The current loss, a single scalar.
        :return pred
        """
        pred = self(imgs)  # (N)
        error = nn.MSELoss()
        # ----------------------- ADD YOUR CODE HERE --------------------------
        loss = None
        y_hat = self.forward(imgs)
        pred = y_hat
        loss = error(y_hat, labels.float())
        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss