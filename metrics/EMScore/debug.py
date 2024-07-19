import os
import argparse
import pickle
import numpy as np
import json
import glob
import torch
import math
from tqdm import tqdm
from emscore import EMScorer
from emscore.utils import get_idf_dict, compute_correlation_uniquehuman
import clip
import pdb
import random 
random.seed(0)

def get_feats_dict(feat_dir_path):
    print('loding cache feats ........')
    file_path_list = glob.glob(feat_dir_path+'/*.pt')
    feats_dict = {}
    for file_path in tqdm(file_path_list):
        vid = file_path.split('/')[-1][:-3]
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', default='./VATEX-EVAL', type=str, help='The path you storage VATEX-EVAL dataset')
    parser.add_argument('--vid_base_path', default='', type=str, help='The path you storage VATEX-EVAL videos (optinal, if you use prepared video feats, You do not need to consider this)')
    parser.add_argument('--use_n_refs', default=1, type=int, help='How many references do you want to use for evaluation (1~9)')
    parser.add_argument('--use_feat_cache', default=True, action='store_true', help='Whether to use pre-prepared video features')
    parser.add_argument('--use_idf', action='store_true', default=True)
    parser.add_argument('--inpath', default='./data1/yll/ControlVideoCap/metrics/coco_caption/predict_files_solo/7_all.json', type=str)
    opt = parser.parse_args()

    """
    Video feats prepare
    """

    # use_uniform_sample = 10
    use_uniform_sample = 10
    vids = []
    if not opt.use_feat_cache:
        vids = [vid_base_path+vid+'.mp4' for vid in video_ids]
        metric = EMScorer(vid_feat_cache=[])
    else:
        # vid_clip_feats_dir = os.path.join(opt.storage_path, 'VATEX-EVAL_video_feats') # feature文件夹包含3000个视频的特征
        vid_clip_feats_dir = os.path.join(opt.storage_path, 'en_clip_feats/clip_vid_feats')
        video_clip_feats_dict = get_feats_dict(vid_clip_feats_dir)
        if use_uniform_sample:
            for vid in video_clip_feats_dict:
                data = video_clip_feats_dict[vid]
                select_index = np.linspace(0, len(data)-1, use_uniform_sample)
                select_index = [int(index) for index in select_index]
                video_clip_feats_dict[vid] = data[select_index]
                vids.append(vid)
                # pdb.set_trace()
        metric = EMScorer(vid_feat_cache=video_clip_feats_dict)
    

    """
    Dataset prepare
    """
    cache_vids = vids
    vid_base_path = 'your path to save vatex val videos'  # optional
    if opt.inpath == "":
        samples_list = pickle.load(open(os.path.join(opt.storage_path, 'candidates_list.pkl'), 'rb'))
        gts_list = pickle.load(open(os.path.join(opt.storage_path, 'gts_list.pkl'), 'rb'))
        video_ids = pickle.load(open(os.path.join(opt.storage_path, 'video_ids.pkl'), 'rb')) 
        cands = samples_list.tolist() # 18000  generated captions; list [cand1, cand2, ... ]
        refs = gts_list.tolist() # 18000 gt captions; list [[ref1, .., ref10], [...], ...]
    else:
        datas = read_json_line(opt.inpath)
        cands = []
        refs = []
        video_ids = [] 
        for jterm in datas:
            if jterm["vid"] not in cache_vids:
                continue
            # cands.append(jterm["newcap_generated"])
            
            short_gts = [] # filter gt sent > 77 tokens (limited by CLIP Text Encoder)
            for gt in jterm["newcap_gt"]:
                if len(gt.split(" ")) < 75:
                    short_gts.append(gt)
            if len(short_gts) < 1:
                continue
            refs.append(short_gts)
            cands.append(random.sample(short_gts, 1)[0])
            video_ids.append(jterm["vid"])

    vids = video_ids
    """
    Prepare IDF
    """
    if opt.use_idf:
        vatex_train_corpus_path = os.path.join(opt.storage_path, 'vatex_train_en_annotations.json')
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
    # pdb.set_trace()
    # refs = np.array(refs)[:, :opt.use_n_refs].tolist()
    # newrefs = []
    # for ref in refs:
    #     if opt.use_n_refs >= len(ref):
    #         newrefs.append(ref[:opt.use_n_refs])
    #     else:
    #         print(len(ref), ref)
    #         newrefs.append(ref)
    # refs = newrefs
    # results = metric.score(cands, refs=refs, vids=vids, idf=emscore_idf_dict) # TODO
    results = metric.score(cands, refs=[], vids=vids, idf=emscore_idf_dict) 
    if 'EMScore(X,V)' in results:
        print('EMScore(X,V) correlation --------------------------------------')
        vid_full_res_F = results['EMScore(X,V)']['full_F']
        # vid_full_res_F: torch.Size([31844])  save the F score for each generated sent
        print('EMScore(X,V) -> full_F: {:.1f}'.format(vid_full_res_F.mean().item()*100))
        

    # if 'EMScore(X,X*)' in results:
    #     print('EMScore(X,X*) correlation --------------------------------------')
    #     refs_full_res_F = results['EMScore(X,X*)']['full_F']
    #     print('EMScore(X,X*) -> full_F: ', refs_full_res_F.mean())


    # if 'EMScore(X,V,X*)' in results:
    #     print('EMScore(X,V,X*) correlation --------------------------------------')
    #     vid_refs_full_res_F = results['EMScore(X,V,X*)']['full_F']
    #     print('EMScore(X,V,X*) -> full_F: ', vid_refs_full_res_F.mean())
        
