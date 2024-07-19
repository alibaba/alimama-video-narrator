import time
import copy
import logging
import os
import json
import pickle
from tqdm import tqdm
from train.llava_trainer import VideoChatGPTTrainer
os.environ['ENABLE_FLOPS_STATISTICS'] = '-1'
os.environ['MODEL_PARAMS_THOP'] = '-1'
os.environ['NCCL_MIN_NCHANNELS'] = '16'
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import sys

import torch
import transformers
import utils
from torch.utils.data import Dataset
# TODO yansong
# from transformers import Trainer

from psutil import cpu_percent,virtual_memory
from tokenization_baichuan import BaiChuanTokenizer
from modeling_lmm_baichuan import BaiChuanVideoCapLlamaForCausalLM, VideoCapLlamaModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

if __name__ == "__main__":
    load_dir = "/pretrained_models/baichuan-7b-sft/"
    tokenizer = BaiChuanTokenizer.from_pretrained(load_dir,model_max_length=999999999)
    print('+' * 30, 'tokenizer', tokenizer)
    out = {}
    fin = json.load(open(sys.argv[1]))
    out_dir = "/".join(sys.argv[1].split("/")[:-1])
    print(out_dir)
    fout = open(out_dir+"/out_tokens.json","w")
    for k in fin.keys():
        out[k] = {}
        info = fin[k]
        if("gt" not in info.keys()):continue
        for t_k in info.keys():
            t_text = info[t_k]
            #print(t_text)
            #if("." in t_text):t_text = t_text.split(".")[1]
            #if("您好" in t_text or "你好" in t_text):continue
            #if("广告词为：" in t_text):t_text=t_text.split("广告词为：")[1].split("\n")[0]
            if(len(t_text)<4):continue
            if(len(t_text)==0 or t_text[-1]!="。"):t_text=t_text+"。"

            t_text=t_text.replace("。。","。")
            #last_index = t_text.find("。。")
            #if(last_index!=-1):t_text = t_text[:last_index]
            #t_text = t_text.lower()
            out[k][t_k] = tokenizer(t_text)["input_ids"][1:]
    json.dump(out,fout,ensure_ascii=False, indent=2)