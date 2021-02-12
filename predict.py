import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch import optim
import numpy as np
from utils.arguments_parse import args
import data_preprocessing
from model.model import myModel
from model.loss_function import multilabel_cross_entropy
from model.metrics import metrics
from data_preprocessing import *
import json
from tqdm import tqdm
import unicodedata, re
from data_preprocessing import tools


device = torch.device('cuda')

added_token = ['[unused1]', '[unused1]']
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_path, additional_special_tokens=added_token)
label2id,id2label,num_labels = tools.load_schema()


def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        arguments = []
        for line in lines:
            data = json.loads(line)
            text = data['text']
            entity_list = data['entity_list']
            args_dict={}
            if entity_list != []:
                for entity in entity_list:
                    entity_type = entity['type']
                    entity_argument=entity['argument']
                    args_dict[entity_type] = entity_argument
                sentences.append(text)
                arguments.append(args_dict)
        return sentences, arguments


def get_mapping(text):
    text_token=tokenizer.tokenize(text)
    text_mapping = tools.token_rematch().rematch(text,text_token)
    return text_mapping


def sapn_decode(span_logits):
    arg_index=[]
    for i in range(len(span_logits)):
        for j in range(i,len(span_logits[i])):
            if span_logits[i][j]>0:
                arg_index.append((i,j,id2label[span_logits[i][j]-1]))
    return arg_index


def main():

    with torch.no_grad():
        model = myModel(pre_train_dir=args.pretrained_model_path, dropout_rate=0.5).to(device)
        model.load_state_dict(torch.load(args.checkpoints))
        sentences,_ = load_data(args.test_path)
        with open('./output/result.json','w',encoding='utf-8') as f:
            for sent in tqdm(sentences):
                encode_dict = tokenizer.encode_plus(sent,
                                        max_length=args.max_length,
                                        pad_to_max_length=True)
                input_ids = encode_dict['input_ids']
                input_seg = encode_dict['token_type_ids']
                input_mask = encode_dict['attention_mask']

                input_ids = torch.Tensor([input_ids]).long()
                input_seg = torch.Tensor([input_seg]).long()
                input_mask = torch.Tensor([input_mask]).float()
                span_logits = model( 
                            input_ids=input_ids.to(device), 
                            input_mask=input_mask.to(device),
                            input_seg=input_seg.to(device),
                            is_training=False)
                
                span_logits = torch.argmax(span_logits,dim=-1)[0].to(torch.device('cpu')).numpy().tolist()
                args_index=sapn_decode(span_logits)
                text_mapping=get_mapping(sent)
                entity_list=[]
                
                for k in args_index:
                    dv = 0
                    while text_mapping[k[0]-1+dv] == []:
                        dv+=1
                    start_split=text_mapping[k[0]-1+dv]

                    while text_mapping[k[1]-1+dv] == []:
                        dv+=1
                    
                    end_split=text_mapping[k[1]-1+dv]

                    argument=sent[start_split[0]:end_split[-1]+1]
                    entity_type=k[2]
                    entity_list.append({'type':entity_type,'argument':argument})
                result={'text':sent,'entity_list':entity_list}
                json_data=json.dumps(result,ensure_ascii=False)
                f.write(json_data+'\n')
            
if __name__ == '__main__': 
    main()