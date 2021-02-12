import torch
from torch import nn

class binary_cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.BCELoss()

    def forward(self,y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()
        y_pred = y_pred.view(size=(-1, ))
        y_true = y_true.view(size=(-1,))
        loss = self.loss_func(input=y_pred, target=y_true)
        return -loss
