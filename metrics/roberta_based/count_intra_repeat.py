import json
import jieba
import jieba.posseg as pseg
import sys
jieba.initialize()
jieba.enable_paddle()
jieba.load_userdict("./words.txt")
test_id = open("../../data/split/test_ids.txt").read().split("\n")
ori_info = json.load(open("../../data/all_video_data.json"))
out_f = json.load(open(sys.argv[1]))
reps = []
for id in test_id[:-1]:
    info = ori_info[id]
    frame_num = len(info["frames"])
    vid = id.split(".")[0]
    outs = []
    out_words= []
    out_tri = []
    t_reps = []
    out_len = []
    for i in range(frame_num):
        t_id = "{}_{}".format(vid,i)
        if("new_text" in info["frames"][i]):
            gt_length=len(info["frames"][i]["new_text"])
        else:
            continue
        t_out_w = []
        t_processed = []
        if(t_id in out_f and "0.1" in out_f[t_id]):
            outs.append(out_f[t_id]["0.1"])#[:gt_length+10])#[:gt_length+10])
            words = pseg.cut(out_f[t_id]["0.1"])#[:gt_length+10], use_paddle=True)
            t_len = 0
            for word, flag in words:
                t_out_w.append(word)
            #print("w_w_out",t_out_w)
            for t_j,w in enumerate(t_out_w[:-1]):
                j=-t_j-1
                if(len(t_out_w[j])==1 and t_out_w[j] not in ["，","。"] and t_out_w[j-1] not in ["，","。"]):
                   t_out_w[j]=t_out_w[j-1]+t_out_w[j]
                #elif(w not in #t_processed.append(t_out_w[j-1]+t_out_w[j])
            for j,w in enumerate(t_out_w):
                #j=-t_j-1
                if(len(w)!=1):t_processed.append(w)
                elif(j<len(t_out_w)-1 and w!=t_out_w[j+1][0] and w not in ["，","。"]):t_processed.append(t_out_w[j]+t_out_w[j+1])
            #print("pre",t_out_w)
            #print("process",t_processed)
            if(len(t_processed)==0):
                print(out_f[t_id]["0.1"])
            else:
                out_len.append(len(t_processed))
                out_words.append(t_processed)
    if(len(out_words)==0 or len(out_words)==1):continue
    all_rep_words =0
    all_len = 0
    #print("out",out_words)
    for i,w_list in enumerate(out_words):
        #if(i==0):continue
        t_len = len(outs[i])
        rep_word = []
        rep_label = []
        for j,w in enumerate(w_list):
            if(len(w)==1):
                rep_label.append(False)
                continue
            for k,other_list in enumerate(out_words[:i]):
                if(k==i):continue
                whether_rep = False
                if(w in other_list):
                    whether_rep=True
                if(whether_rep):
                    rep_word.append(w)
                rep_label.append(whether_rep)
        all_rep_words+=len(rep_word)
        all_len+=len(w_list)
        if(out_len[i]==0):continue
        #print(i,rep_word,w_list)
        t_reps.append(float(len(rep_word)) / float(len(w_list)))
    if(len(t_reps)==0):
        print(out_words)
    else:
        reps.append(float(all_rep_words)/float(all_len))


print(sum(reps)/len(reps))
