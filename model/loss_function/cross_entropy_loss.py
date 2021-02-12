import torch
from torch import nn
from utils.arguments_parse import args

class cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self,start_logits,end_logits,start_label, end_label,seq_mask):
        start_label, end_label = start_label.view(size=(-1,)), end_label.view(size=(-1,))
        start_loss = self.loss_func(input=start_logits.view(size=(-1, 2)), target=start_label)
        end_loss = self.loss_func(input=end_logits.view(size=(-1, 2)), target=end_label)

        # 加入focal loss实现 => 效果更差了, emmmm
        """start_prob_seq, end_prob_seq = start_prob_seq.view(size=(-1, 2)), end_prob_seq.view(size=(-1, 2))
        start_loss = (1 - start_label) * ((1 - start_prob_seq[:, 0]) ** self.gamma) * start_loss + \
                        self.alpha * start_label * ((1 - start_prob_seq[:, 1]) ** self.gamma) * start_loss
        end_loss = (1 - end_label) * ((1 - end_prob_seq[:, 0]) ** self.gamma) * end_loss + \
                    self.alpha * end_label * ((1 - end_prob_seq[:, 1]) ** self.gamma) * end_loss"""
                    
        sum_loss = start_loss + end_loss
        sum_loss *= seq_mask.view(size=(-1,))

        avg_se_loss = torch.sum(sum_loss) / seq_mask.size()[0]
        # avg_se_loss = torch.sum(sum_loss) / bsz
        return avg_se_loss

        