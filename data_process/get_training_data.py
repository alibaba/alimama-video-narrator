import pickle
import json
import sys

import torch
import numpy as np
from tqdm import tqdm

fin = json.load(open(sys.argv[2]))
fea_loc = sys.argv[1]
fout = open("training_data.json","w")
out_json = []
out_w_k = {}
t_i =0
device = "cuda:0"
sim_func = torch.nn.functional.cosine_similarity
def merge(ori_tokens,mem_compress):
    t_len = ori_tokens.shape[0]
    short_len = min(int((t_len+1)/2),5)
    t_tokens = torch.from_numpy(ori_tokens).to(device)
    #print(t_tokens.shape)
    mean_tok = torch.mean(t_tokens,dim=1).to(device)
    #print(mean_tok.shape)
    ori_sims = None
    left_time = torch.tensor([i for i in range(t_len)]).to(device)
    while(len(t_tokens)>1):
        sims = sim_func(mean_tok[:-1],mean_tok[1:],)
        if(ori_sims is None):ori_sims=sims.clone()
        loc = sims.topk(1).indices.to(device)
        # 存储memory信息时，压缩到一帧中
        if(sims[loc]<0.9 and mem_compress==False):break


        merged = (t_tokens[loc] + t_tokens[loc+1])/2 #t_tokens[loc]
        merged_mean = (mean_tok[loc] + mean_tok[loc+1])/2
        if (mem_compress==False):
            merged = t_tokens[loc]
            merged_mean = mean_tok[loc]
        t_tokens = torch.cat([t_tokens[0:loc],merged.to(device),t_tokens[loc+2:]])
        mean_tok = torch.cat([mean_tok[0:loc],merged_mean.to(device),mean_tok[loc+2:]])
        left_time = torch.cat([left_time[0:loc],left_time[loc],left_time[loc+2:]])
    return t_tokens.detach().cpu().numpy(),left_time.detach().cpu().numpy()

for k in tqdm(fin.keys()):
    t_id = k.split(".")[0]
    p_info = fin[k]["p_info"]
    p_name = fin[k]["p_name"]
    frames_input= []
    video_fea_loc = []
    extract_images = []
    frame_num = len(fin[k]["frames"])
    memory_fea = []
    all_clip_fea = []
    for fi,tf in enumerate(fin[k]["frames"]):
        st = int(tf["time"][0])+1
        et = int(tf["time"][1])+1
        if(et<st or et==st):et=st+1
        t_label = tf["label"]
        if("new_text" not in tf):continue
        t_text = tf["new_text"]
        t_feas = []
        for i in range(st,et):
            fname = t_id+"_%03d"%i+".jpg.npy"
            try:
               t_fea = np.load(fea_loc+"/"+fname)
               t_feas.append(t_fea)
            except:
                print(fname)
        if(len(t_feas)==0):continue
        all_fea  = np.stack(t_feas)
        compress_fea,t_left = merge(all_fea,mem_compress=False)
        new_fea = compress_fea#np.concatenate([t_arr for t_arr in compress_fea])
        if(len(memory_fea)>=1):
            mem_fea,left_time = merge(np.stack(memory_fea),mem_compress=True)
            frame_fea = np.concatenate([mem_fea,new_fea])
            frame_fea=np.concatenate([t_arr for t_arr in frame_fea])
        else:
            frame_fea = np.concatenate([t_arr for t_arr in new_fea])
        memory_fea+=t_feas

        frame_loc = "/data_process/video_feas_mem_compress/{}_{}.pkl".format(t_id,fi)
        video_fea_loc.append("{}_{}.pkl".format(t_id,fi))
        t_length = len(t_text)
        with open(frame_loc, 'wb') as f:
            pickle.dump(frame_fea, f)
        frames_input.append("镜头{}/{}：\n场景：<vid_start><vid_end>\n字数要求：{}-{}个字\n剧本类型：{}\n{}.{}".format(fi+1,frame_num, t_length-3,t_length+3,t_label,fi+1,t_text))
    if(len(frames_input)==0):continue
    t_input = "商品信息：\n名称：{}\n{}\n\n广告视频：\n{}".format(p_name,p_info,"\n".join(frames_input))
    #print(t_input)
    out_json.append({"inputs":t_input,"id":t_id,"frames":video_fea_loc})
    out_w_k[t_id+".mp4"]=out_json[-1]
    t_i+=1

json.dump(out_json,fout,ensure_ascii=False)

