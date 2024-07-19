import sys
import json
import jieba
import torch
import jieba.posseg as pseg
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
jieba.initialize()
jieba.enable_paddle()
jieba.load_userdict("./words.txt")
test_id = open("/data/split/test_ids.txt").read().split("\n")
ori_info = json.load(open("/data/split/all_video_data.json"))
out_f = json.load(open(sys.argv[1]))
idfs = json.load(open(sys.argv[2]))



MODEL_LOC = "/metrics/chinese-roberta-large/"
tokenizer = BertTokenizer.from_pretrained(MODEL_LOC)
model = BertModel.from_pretrained(MODEL_LOC)
device = "cuda:0"
model.to(device)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def count_word_sim(caps, points,model,tokenizer,idf_w):

    #left_pos = ["n","l","s","ns", "v", "vn", "nz","a"]
    points = points[1:]
    t_length = 0
    cap_word_length = [0]
    vid_words = []
    all_cap_len = 0

    for cap in caps:
        words = pseg.cut(cap, use_paddle=True)
        t_ws=[]
        for word, flag in words:
            if(len(word)==1):continue
            all_cap_len+=1
            if(word not in vid_words):
                t_ws.append(word)
                vid_words.append(word)
        if(len(t_ws)==0):
            t_ws = [cap]
            vid_words.append(cap)
        t_length += len(t_ws)
        cap_word_length.append(t_length)

    #同时进行编码
    all_words= vid_words+points
    if(len(all_words)>128):
        concat_fea=None
        batch_num = int(len(all_words)/128)+1
        for i in range(batch_num):
            if(len(all_words[128*i:128*(i+1)])==0):continue
            text_inputs_1 = tokenizer(all_words[128*i:128*(i+1)],padding=True, return_tensors="pt").to(model.device)
            outputs_1 = model(**text_inputs_1)
            concat_fea_1 = mean_pooling(outputs_1, text_inputs_1['attention_mask'])
            if(concat_fea is None):concat_fea=concat_fea_1
            else:concat_fea = torch.cat([concat_fea,concat_fea_1])
    else:
        text_inputs = tokenizer(all_words, padding=True, return_tensors="pt").to(model.device)
        outputs = model(**text_inputs)
        concat_fea = mean_pooling(outputs, text_inputs['attention_mask'])  # outputs[0][:, 0, :]
    concat_fea = concat_fea / concat_fea.norm(dim=-1, keepdim=True)
    #print(concat_fea.shape)
    text_fea = concat_fea[:len(vid_words),:]
    point_fea = concat_fea[len(vid_words):,:]


    #max = torch.mean(torch.stack([t_sim[t_i] for t_sim, t_i in zip(sims, indices)]), dim=1)
    final_max = []
    lefted_words =[]
    covered_points = []
    point_sims = point_fea @ text_fea.t()
    top_value, indices = point_sims.topk(1, dim=1, largest=True)
    #print(top_value)
    used = [True for i in range(len(text_fea))]
    for i,t_max in enumerate(top_value):
        if(t_max>0.9 and used[indices[i]]):
            covered_points.append(points[i])
            used[indices[i]]=False

    print(caps)
    print(covered_points)
    print("all_points",points)
    print(len(covered_points)/all_cap_len)
    #print(vid_words)
    return lefted_words,covered_points,all_cap_len

def format_key(t_k):
    if("】" in t_k):t_k=t_k.split("】")[1]
    if(" " in t_k):t_k = t_k.split(" ")[0]
    return t_k
reps = []
per_time =[]
for id in tqdm(test_id[:-1],total=len(test_id)):
    points = []
    t_info = ori_info[id]
    p_name = t_info["p_name"]
    if ("】" in p_name): p_name = p_name.split("】")[1]
    t_idf = idfs[p_name]
    idf_w = [t_idf[t_p] for t_p in t_idf.keys()]
    points = [format_key(t_p) for t_p in t_idf.keys()]
    info = ori_info[id]
    frame_num = len(info["frames"])
    vid = id.split(".")[0]
    caps = []
    out_words= []
    out_tri = []
    t_reps = []
    time_length =0
    for i in range(frame_num):
        t_id = "{}_{}".format(vid,i)
        t_out_w = []
        if(t_id in out_f and "0.1" in out_f[t_id]):
            #if(len(out_f[t_id]["0.1"])<3):continue
            caps.append(out_f[t_id]["0.1"])#.split("。")[0])
            time_length+=abs(info["frames"][i]["time"][1]-info["frames"][i]["time"][0])
    if(len(caps)==0):continue

    out_words,covered,story_w_len=count_word_sim(caps,points,model,tokenizer,idf_w)

    if(len(covered)==0):
        reps.append(0)
        per_time.append(0)
    else:
        reps.append(len(covered)/float(story_w_len))
        per_time.append(len(covered)/time_length)
    #reps.append(sum(t_reps)/len(t_reps))

#print(sum(reps)/len(reps))
print(sum(per_time)/len(per_time))



