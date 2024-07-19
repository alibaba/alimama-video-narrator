import pickle
import json
import sys
import os

import torch
import numpy as np
from tqdm import tqdm

fea_loc = sys.argv[1]
files = os.listdir("./Text/")
test_ids = open("test_ids.txt").read().split("\n")
fout = open("training_data.json","w")
fout_2 = open("test_data.json","w")
clip_loc = "/data/video_story/clip/clip_fea/video_cuts/"
out_json = []
out_json_test = []
t_i =0
device = "cuda:0"
sim_func = torch.nn.functional.cosine_similarity
frame_num =[]

def merge(ori_tokens,clip_tokens, mem_compress):
    t_len = ori_tokens.shape[0]
    short_len = min(int((t_len+1)/2),5)
    t_tokens = torch.from_numpy(ori_tokens).to(device)
    #print(t_tokens.shape)
    mean_tok = torch.from_numpy(clip_tokens).to(device)
    #print(mean_tok.shape)
    ori_sims = None
    left_time = torch.tensor([i for i in range(t_len)]).to(device)
    while(len(t_tokens)>1):
        sims = sim_func(mean_tok[:-1],mean_tok[1:],)
        if(ori_sims is None):ori_sims=sims.clone()
        loc = sims.topk(1).indices.to(device)
        # 存储memory信息时，压缩到一帧中
        if(len(t_tokens)<=short_len and mem_compress==False):break

        merged = (t_tokens[loc] + t_tokens[loc+1])/2 #t_tokens[loc]
        merged_mean = (mean_tok[loc] + mean_tok[loc+1])/2
        #print(t_tokens.shape)
        #print(loc)
        t_tokens = torch.cat([t_tokens[0:loc],merged.to(device),t_tokens[loc+2:]])
        mean_tok = torch.cat([mean_tok[0:loc],merged_mean.to(device),mean_tok[loc+2:]])
        left_time = torch.cat([left_time[0:loc],left_time[loc],left_time[loc+2:]])
    return t_tokens.detach().cpu().numpy(),left_time.detach().cpu().numpy()

def get_time_loc(times,t_time,shot_len):
    #times = [f["time"][1]-f["time"][0] for f in v["frames"]]
    all_len = sum(times)
    #t_time  = [f["time"][1]-f["time"][0] for f in v["frames"][:i+1]]
    #t_len = sum(t_time)-(v["frames"][i]["time"][1]-v["frames"][i]["time"][0])/2
    t_len = sum(t_time) - (shot_len)/2
    return int((t_len/all_len)*100)

video_len = []
for t_file in tqdm(files):
    t_info = open("./Text/"+t_file).read().split("\n")
    t_id = t_info[0].split(".")[0]
    frames = t_info[1:-1]
    story_id = t_file.split(".")[0]
    frames_input= []
    video_fea_loc = []
    times = []
    print("nums",len(frames))
    frame_num.append(len(frames))
    t_video_len =0
    memory_fea = []
    all_clip_fea = []
    for fi,tf in enumerate(frames):
        all_text = tf.split(" ")
        t0 = all_text[0].split(":")
        t1 = all_text[1].split(":")
        st = int(t0[0])*60+int(t0[1])
        if(st==0):st=1
        et = int(t1[0])*60+int(t1[1])
        t_video_len +=(et-st)
        t_text = " ".join(all_text[2:]).capitalize()
        if(t_text[-1]!="."):t_text = t_text+"."
        times.append(et-st)
        if(et<st or et==st):et=st+1
        #print(st,et,tf)
        t_feas = []
        clip_feas = []
        for i in range(st,et+1):
            fname = t_id+"_%03d"%i+".jpg.npy"
            try:
               t_fea = np.load(fea_loc+"/"+fname)
               clip_fea = np.load(clip_loc+fname)
               t_feas.append(t_fea)
               clip_feas.append(clip_fea)
            except:
                print(fname)
        if(len(t_feas)==0):print("null_shot")
        all_fea = np.stack(t_feas)
        new_fea = all_fea  # compress_fea#np.concatenate([t_arr for t_arr in compress_fea])
        if (len(memory_fea) >= 1):
            mem_fea, left_time = merge(np.stack(memory_fea), np.stack(all_clip_fea), mem_compress=True)
            frame_fea = np.concatenate([mem_fea, new_fea])
            frame_fea = np.concatenate([t_arr for t_arr in frame_fea])
        else:
            frame_fea = np.concatenate([t_arr for t_arr in new_fea])
        memory_fea += t_feas
        all_clip_fea += clip_feas
        frame_loc = "/data/video_story/video_story_feas/{}_{}.pkl".format(story_id,fi)
        video_fea_loc.append("{}_{}.pkl".format(story_id,fi))
        t_length = len(t_text.split(" "))
        with open(frame_loc, 'wb') as f:
            pickle.dump(frame_fea, f)
        frames_input.append("### Input:\n Video shot {}:\n visual scene: <vid_start><vid_end>\n word count requirement: {}-{} words\n ### Response:\n {}".format(fi+1,t_length-5,t_length+5,t_text))
    video_len.append(t_video_len)
    if(len(frames_input)==0):continue
    t_input = "\n ".join(frames_input)
    whether_test = False
    if(t_id in test_ids): whether_test=True
    #print(t_input)
    concat_outs = {"inputs":t_input,"id":story_id,"video":t_id,"frames":video_fea_loc}
    if(whether_test):
        out_json_test.append(concat_outs)
    else:
        for fi,tf in enumerate(frames):
            t_pos = get_time_loc(times, times[:fi+1],times[fi])
            if (t_pos < 0 or t_pos > 99): print("no")
            if (fi == 0): t_pos = 0
            if (fi == len(frames) - 1): t_pos = 99
            t_outs = {"shot_pos":t_pos, "shot_loc":(fi+1), "inputs":t_input.replace("visual scene: <vid_start><vid_end>\n ","").replace("Video shot {}:\n ".format(fi+1),"Video shot {}:\n visual scene: <vid_start><vid_end>\n ".format(fi+1)),
                      "id":story_id,"frames":[video_fea_loc[fi]]}
            out_json.append(t_outs)
    t_i+=1
    #if(t_i>50):break

json.dump(out_json,fout,ensure_ascii=False)
json.dump(out_json_test,fout_2,ensure_ascii=False)
print(min(video_len))
print(max(video_len))

print(sum(frame_num)/len(frame_num))