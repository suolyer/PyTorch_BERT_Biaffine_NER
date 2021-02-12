import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss

    # def forward(self,start_logits,end_logits,start_label, end_label,seq_mask):
    #     start_label, end_label = start_label.view(size=(-1,)), end_label.view(size=(-1,))
    #     start_logits, end_logits = start_logits.view(size=(-1, 2)), end_logits.view(size=(-1, 2))
    #     start_loss= self.focal(start_logits,start_label)
    #     end_loss=self.focal(end_logits,end_label)
    #     sum_loss = start_loss + end_loss
    #     # sum_loss *= seq_mask.view(size=(-1,))
    #     # avg_se_loss = torch.sum(sum_loss) / seq_mask.size()[0]
    #     return sum_loss