import torch
from torch import nn

class multilabel_cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y_pred, y_true):
        y_true = y_true.float()
        y_pred = torch.mul((1.0 - torch.mul(y_true,2.0)),y_pred)
        y_pred_neg = y_pred - torch.mul(y_true,1e12)
        # y_pred_neg = y_pred_neg        
        y_pred_pos = y_pred - torch.mul(1.0 - y_true,1e12)
        # y_pred_pos = y_pred_pos
        zeros = torch.zeros_like(y_pred[..., :1])
        # zeros = zeros
        y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
        pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
        loss = torch.mean(neg_loss + pos_loss)
        return loss
