import torch
from torch import nn
from utils.arguments_parse import args
from data_preprocessing import tools

label2id,id2label,num_labels=tools.load_schema()
num_label = num_labels+1

class Span_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self,span_logits,span_label,seq_mask):
        # batch_size,seq_len,hidden=span_label.shape
        span_label = span_label.view(size=(-1,))
        span_logits = span_logits.view(size=(-1, num_label))
        span_loss = self.loss_func(input=span_logits, target=span_label)

        # start_extend = seq_mask.unsqueeze(2).expand(-1, -1, seq_len)
        # end_extend = seq_mask.unsqueeze(1).expand(-1, seq_len, -1)
        span_mask = seq_mask.view(size=(-1,))
        span_loss *=span_mask
        avg_se_loss = torch.sum(span_loss) / seq_mask.size()[0]
        # avg_se_loss = torch.sum(sum_loss) / bsz
        return avg_se_loss





