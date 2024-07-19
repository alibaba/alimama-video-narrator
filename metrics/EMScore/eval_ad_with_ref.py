import re
import os
import argparse
import pickle
import numpy as np
import json
import glob
import torch
import math
from tqdm import tqdm
from emscore_Chinese import EMScorer
from emscore_Chinese.utils import get_idf_dict, compute_correlation_uniquehuman
import clip
import pdb

def get_feats_dict(feat_dir_path,video_ids):
    print('loding cache feats ........')
    file_path_list = glob.glob(feat_dir_path+'/*.pt')
    feats_dict = {}
    for file_path in tqdm(file_path_list):
        vid = file_path.split('/')[-1][:-3]
        if(vid not in video_ids):continue
        data = torch.load(file_path)
        feats_dict[vid] = data
    return feats_dict

def read_json_line(path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data

def get_label(key_and_frame, data):
    key,frmae = key_and_frame.split("_")
    t_l = data[key+".mp4"]["frames"][int(frmae)]
    return t_l["label"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', default='/data_process/ChineseCLIP_fea/test_fea/ViT-B-16', type=str, help='The path you storage dataset')
    parser.add_argument('--vid_base_path', default='', type=str, help='The path you storage videos (optinal, if you use prepared video feats, You do not need to consider this)')
    parser.add_argument('--use_n_refs', default=1, type=int, help='How many references do you want to use for evaluation (1~9)')
    parser.add_argument('--use_feat_cache', default=True, action='store_true', help='Whether to use pre-prepared video features')
    parser.add_argument('--use_idf', action='store_true', default=True)
    parser.add_argument('--inpath', default='../output.json', type=str)
    #parser.add_argument('--train_file', required=True, default='', type=str, help="file to compute tf-idf")
    opt = parser.parse_args()

    """
    Dataset prepare
    """
    video_info = json.load(open("/data/split/all_video_data.json"))
    vid_base_path = 'your path to save vatex val videos'  # optional
    def remove_english(string):
        #return string
        pattern = r'[a-zA-Z]' # 定义要删除的模式为所有大小写英文字母
        result = re.sub(pattern, '', string) # 使用re.sub函数将匹配到的部分替换成空字符串
        if("\n" in result): result = result.split("\n")[0]
        if("！" in result): result = result.split("！")[0]
        if("？" in result): result = result.split("？")[0] 
        if(result[-1]!="。"):result = result+"。"
        return result.replace("'","") 
    if opt.inpath == "":
        samples_list = pickle.load(open(os.path.join(opt.storage_path, 'candidates_list.pkl'), 'rb'))
        gts_list = pickle.load(open(os.path.join(opt.storage_path, 'gts_list.pkl'), 'rb'))
        video_ids = pickle.load(open(os.path.join(opt.storage_path, 'video_ids.pkl'), 'rb'))
        cands = samples_list.tolist() # 18000  generated captions; list [cand1, cand2, ... ]
        refs = gts_list.tolist() # 18000 gt captions; list [[ref1, .., ref10], [...], ...]
    else:
        datas = json.load(open(opt.inpath))
        cands = []
        refs = []
        video_ids = []
        video_i=0
        for k in datas.keys():
            video_i+=1
            #if(video_i>300):break
            jterm = datas[k]
            t_label = get_label(k,video_info)

            if("0.1" not in jterm):continue
            gt_length = len(jterm["gt"])
            t_out = jterm["0.1"]
            #t_out=t_out.replace(" ","")
            if(len(t_out)<4):continue
            print(t_out)
            #if(t_out in cands):continue
            cands.append(t_out)
            short_gts = [] # filter gt sent > 77 tokens (limited by CLIP Text Encoder)
            short_gts.append(jterm["gt"][:77])
            refs.append(short_gts)
            video_ids.append(k)

    """
    Video feats prepare
    """

    # use_uniform_sample = 10
    use_uniform_sample = 10
    vid_clip_feats_dir = os.path.join(opt.storage_path, 'chinese_clip_vid_feats')
    video_clip_feats_dict = get_feats_dict(vid_clip_feats_dir,video_ids)
    if use_uniform_sample:
        for vid in video_clip_feats_dict:
            data = video_clip_feats_dict[vid]
            select_index = np.linspace(0, len(data)-1, use_uniform_sample)
            select_index = [int(index) for index in select_index]
            video_clip_feats_dict[vid] = data#[select_index]
            if(len(data)>use_uniform_sample):video_clip_feats_dict[vid] = data[select_index]
            # pdb.set_trace()

    vids = video_ids
    metric = EMScorer(vid_feat_cache=video_clip_feats_dict)


    """
    Prepare IDF
    """
    use_idf=False
    if use_idf:
        vatex_train_corpus_path = os.path.join(opt.storage_path, opt.train_file) # TODO
        vatex_train_corpus = json.load(open(vatex_train_corpus_path))
        vatex_train_corpus_list = []
        for vid in vatex_train_corpus:
            vatex_train_corpus_list.extend(vatex_train_corpus[vid])

        emscore_idf_dict = get_idf_dict(vatex_train_corpus_list, clip.tokenize, nthreads=4)
        # max token_id are eos token id
        # set idf of eos token are mean idf value
        emscore_idf_dict[max(list(emscore_idf_dict.keys()))] = sum(list(emscore_idf_dict.values()))/len(list(emscore_idf_dict.values()))
    else:
        emscore_idf_dict = False

    """
    Metric calculate
    """

    results = metric.score(cands, refs=refs, vids=vids, idf=emscore_idf_dict)
    if 'EMScore(X,V)' in results:
        print('EMScore(X,V) correlation --------------------------------------')
        vid_full_res_F = results['EMScore(X,V)']['full_F']
        vid_loc_res_F = results['EMScore(X,V)']['figr_F']
        #print(results)
        for j,s in enumerate(vid_loc_res_F):
            #print(j,s)
            if(s>0.6):print(vids[j],cands[j])
        # vid_full_res_F: torch.Size([31844])  save the Fscore for each generated sent
        print('EMScore(X,V) -> full_F: {:.3f}'.format(vid_full_res_F.mean().item()*100))
        print('EMScore(X,V) -> figr_F: {:.3f}'.format(vid_loc_res_F.mean().item()*100))


    if 'EMScore(X,X*)' in results:
         print('EMScore(X,X*) correlation --------------------------------------')
         refs_full_res_F = results['EMScore(X,X*)']['full_F']
         print('EMScore(X,X*) -> full_F: ', refs_full_res_F.mean().item()*100)
         refs_loc_res_F = results['EMScore(X,X*)']['figr_F']
         print('EMScore(X,X*) -> figr_F: ', refs_loc_res_F.mean().item()*100)


    if 'EMScore(X,V,X*)' in results:
         print('EMScore(X,V,X*) correlation --------------------------------------')
         vid_refs_full_res_F = results['EMScore(X,V,X*)']['full_F']
         print('EMScore(X,V,X*) -> full_F: ', vid_refs_full_res_F.mean().item()*100)
         vid_refs_loc_res_F = results['EMScore(X,V,X*)']['figr_F']
         print('EMScore(X,V,X*) -> figr_F: ', vid_refs_loc_res_F.mean().item()*100)
