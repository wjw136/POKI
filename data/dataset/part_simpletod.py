import random
from pip import main
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer
import os
import logging
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerateDataSet(Dataset):
    def __init__(self,tokenizer, file_path, block_size=512):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)
        with open(file_path, 'r') as load_f:
            pre_data=json.load(load_f)
        self.data=[]
        for key,item in pre_data.items():
            for gold_r, init_r, context in zip(item['target_response'],item['generated_response'],item['generated_withoutresponse']):
                self.data.append({
                    'name': key,
                    'gold_r':gold_r,
                    'gen_r': init_r,
                    'context': context
                })
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data[i]


# if __name__ == "__main__":
#     dataset=GenerateDataSet(None,'/home/jwwang/DialogGenerate/simpletod/simpletod_test_oracleDB_context[history=full_history]_nocarry.json')
#     for item in dataset:
#         print(item)


# tokenizer = GPT2Tokenizer.from_pretrained('/data/jwwang/dialogGen/simpletod/gpt2/checkpoint-20000')

# print(tokenizer.special_token)