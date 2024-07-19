from transformers import BertTokenizer, BertModel
import torch
import json
import sys
from tqdm import tqdm
import jieba
import jieba.posseg as pseg
import transformers
from typing import Dict, Optional, Sequence
jieba.initialize()
jieba.enable_paddle()
jieba.load_userdict("./words.txt")
IGNORE_INDEX = -100
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def count_word_sim(caps, ori_points,model,tokenizer,ori_idf_w):
    left_pos = ["l","n", "s", "nr", "ns", "nt", "nw", "nz", "a"]
    #left_pos = ["n","l","s","ns","v","nv","nz","a"]
    points = []#points[1:]
    idf_w = []
    for i,p in enumerate(ori_points[:]):
        if(len(p)>1):
            points.append(p)
            idf_w.append(ori_idf_w[i])
    #idf_w[1:]
    t_length = 0
    cap_word_length = [0]
    vid_words = []
    max_len = 77
    tags = []

    for cap in caps:
        words = pseg.cut(cap, use_paddle=True)
        t_ws = [cap[:max_len]]
        vid_words.append(cap[:max_len])
        tags.append("sen")
        for word, flag in words:
            if (flag in left_pos):
                t_ws.append(word)
                vid_words.append(word)
                tags.append(flag)
        if(len(t_ws)==1):
            t_ws.append(cap)
            vid_words.append(cap[:max_len])
            tags.append(flag)
        t_length += len(t_ws)
        cap_word_length.append(t_length)

    #同时进行编码
    text_inputs = tokenizer(vid_words+points,padding=True, return_tensors="pt").to(model.device)
    outputs = model(**text_inputs)
    concat_fea = mean_pooling(outputs, text_inputs['attention_mask'])#outputs[0][:, 0, :]
    concat_fea = concat_fea / concat_fea.norm(dim=-1, keepdim=True)
    #print(concat_fea.shape)
    text_fea = concat_fea[:len(vid_words),:]
    point_fea = concat_fea[len(vid_words):,:]

    word_sims = text_fea @ point_fea.t()

    idf_w = torch.tensor(idf_w).to(model.device)
    sims = word_sims# * (idf_w)  # [t_s*idf_w[j] for j,t_s in enumerate(sims)]

    top_num = 1
    if (len(points) < top_num): top_num = len(points)
    top_value, indices = sims.topk(top_num, dim=1, largest=True)
    # print(indices.shape)
    final_sims = torch.stack([t_sim[t_i]*idf_w[t_i] for t_sim, t_i in zip(sims, indices)])
    final_idfs = torch.stack([idf_w[t_i] for t_i in indices])

    max = torch.sum(final_sims,dim=1)/torch.sum(final_idfs,dim=1)
    #max = torch.mean(torch.stack([t_sim[t_i] for t_sim, t_i in zip(sims, indices)]), dim=1)
    final_max = []
    for t_i, t_length in enumerate(cap_word_length[:-1]):
        final_max.append((max[t_length]+torch.mean(max[t_length+1:cap_word_length[t_i + 1]]).to(device))/2)
        #final_max.append(torch.mean(max[t_length+1:cap_word_length[t_i + 1]]))
        relative_i=0
        for t_p,loc in zip(vid_words[t_length:cap_word_length[t_i + 1]],indices[t_length:cap_word_length[t_i + 1]]):
            print(t_p,tags[t_length+relative_i],points[loc],max[t_length+relative_i])
            relative_i+=1
    #print(vid_words)
    return final_max

def count_sim(caps, points,model,tokenizer,idf_w):
    text_inputs = tokenizer(caps, padding=True, return_tensors="pt").to(model.device)
    point_inputs = tokenizer(points, padding=True, return_tensors="pt").to(model.device)

    outputs = model(**text_inputs)
    outputs_2 = model(**point_inputs)
    #print(outputs.pooler_output.shape)
    text_fea = outputs[1]#.pooler_output
    point_fea = outputs_2[1]#.pooler_output
    '''
    text_fea = model.get_text_features(**text_inputs).float()
    point_fea = model.get_text_features(**point_inputs).float()
    '''
    ori_sims = torch.cosine_similarity(text_fea.unsqueeze(1), point_fea.unsqueeze(0), dim=2).to(model.device)
    sims = ori_sims
    #print(sims)
    idf_w = torch.tensor(idf_w).to(model.device)
    sims = ori_sims*(idf_w)#[t_s*idf_w[j] for j,t_s in enumerate(sims)]

    top_num = 1
    if(len(points)<top_num):top_num=len(points)
    top_value,indices = sims.topk(top_num, dim=1, largest=True)
    #print(indices.shape)
    final_sims = torch.stack([t_sim[t_i] for t_sim,t_i in zip(sims,indices)])
    final_idfs = torch.stack([idf_w[t_i] for t_i in indices])
    for i,cap in enumerate(caps):
        print(cap)
        print(final_sims[i])
        print([points[int(j.cpu())] for j in indices[i]])
    #max = torch.sum(final_sims,dim=1)/torch.sum(final_idfs,dim=1)
    max = torch.mean(torch.stack([t_sim[t_i] for t_sim,t_i in zip(sims,indices)]), dim=1)

    #print(caps)
    #print(max)
    return max

MODEL_LOC = "/metrics/chinese-roberta-large/"
tokenizer = BertTokenizer.from_pretrained(MODEL_LOC)
model = BertModel.from_pretrained(MODEL_LOC)

device = "cuda:0"
model.to(device)

ori_info = json.load(open(sys.argv[1]))
out_file = json.load(open(sys.argv[2]))
idfs = json.load(open(sys.argv[3]))
caps = []
pre_key = None
#effi = None
effi = []
def format_key(t_k):
    if("】" in t_k):t_k=t_k.split("】")[1]
    if(" " in t_k):t_k = t_k.split(" ")[0]
    return t_k
for i,key_frame in tqdm(enumerate(out_file.keys()),total=len(out_file.keys())):
    key,frame = key_frame.split("_")
    if((key!=pre_key or i==len(out_file)-1) and i!=0):
        #进行上一轮的计算
        points = []
        t_info = ori_info[pre_key+".mp4"]
        p_name = t_info["p_name"]
        if ("】" in p_name): p_name = p_name.split("】")[1]
        t_idf = idfs[p_name]
        idf_w = [t_idf[t_p] for t_p in t_idf.keys()]
        points = [format_key(t_p) for t_p in t_idf.keys()]#[t_info["p_name"]]

        if(len(caps)>0):
            with torch.no_grad():
                t_score = count_word_sim(caps,points,model,tokenizer,idf_w)

                for s in t_score:
                    effi.append(s.cpu().detach().numpy())

        caps =[]
    if("0.1" not in out_file[key_frame]):continue
    t_cap=out_file[key_frame]["0.1"]
    if(len(t_cap)==0):continue
    if(t_cap[-1]!="。"):t_cap =t_cap+"。"
    caps.append(t_cap)
    pre_key = key

#print(effi)
print(sum(effi)/len(effi))


