import sys
import json
import random
fin = json.load(open(sys.argv[1]))
info = json.load(open(sys.argv[2]))
train = []
test = []
type_id = {}
test_id = open("test_ids.txt").read().split("\n")
for i,data in enumerate(fin):
    data["inputs"]=data["inputs"].replace("场景描述","场景")
    t_id = data["id"]+".mp4"
    t_l = info[data["id"]+".mp4"]["type"]

    data["images"]=[]
    data["link"]=info[t_id]["link"]
    data["p_info"]=info[t_id]["p_info"]
    for fi,tf in enumerate(info[t_id]["frames"]):
        st = int(tf["time"][0])+1
        et = int(tf["time"][1])+1
        if(et<st or et==st):et=st+1
        #t_label = tf["label"]
        if("new_text" not in tf):continue

        t_image = []
        for i in range(st,et):
            fname = t_id.split(".")[0]+"_%03d"%i+".jpg"
            t_image.append(fname)
        data["images"].append(t_image)
    if(t_id in test_id):test.append(data)
    else:train.append(data)

json.dump(train,open("train.json","w"),ensure_ascii=False)
json.dump(test,open("test.json","w"),ensure_ascii=False)
