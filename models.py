from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class BrainAgeCNN(nn.Module):
    """
    The BrainAgeCNN predicts the age given a brain MR-image.
    """
    def __init__(self) -> None:
        super().__init__()

        # Feel free to also add arguments to __init__ if you want.
        # ----------------------- ADD YOUR CODE HERE --------------------------
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv3d(128, 1, kernel_size=11)
        self.max_pool1 = nn.MaxPool3d(kernel_size=4, stride=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=4, stride=4)
        self.relu = nn.ReLU()
        # ------------------------------- END ---------------------------------

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Forward pass of your model.

        :param imgs: Batch of input images. Shape (N, 1, H, W, D)
        :return pred: Batch of predicted ages. Shape (N)
        """
        # ----------------------- ADD YOUR CODE HERE --------------------------
        pred = None
        ####
        pred = self.conv1(pred)
        pred = self.max_pool1(pred)
        pred = self.relu(pred)
        pred = self.conv2(pred)
        pred = self.max_pool2(pred)
        pred = self.relu(pred)
        pred = self.conv3(pred)
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