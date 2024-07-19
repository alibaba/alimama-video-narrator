import sys
import json
fin = json.load(open(sys.argv[1]))
fout = open(sys.argv[2],"w")

out = []
def process(inputs):
    shots = inputs.split("个字")
    processed_out = []
    processed = False
    for shot in shots[:-1]:
        nums = shot.count("。。")
        if(nums>0):
            processed = True
            s1,s2=shot.split("字数要求：")
            s1=s1.replace("。。","。")
            n1,n2=s2.split("-")
            shot = s1+"字数要求："+str(int(n1)-nums)+"-"+str(int(n2)-nums)
            print(shot)
        processed_out.append(shot)
    processed_out.append(shots[-1])
    return "个字".join(processed_out),processed

def remove_label(text):
    texts = text.split("剧本类型：")
    outs = [texts[0]]
    for t_text in texts[1:]:
        outs.append("\n".join(t_text.split("\n")[1:]))
    #print("".join(outs))
    return "".join(outs)

ori_infos = json.load(open("../all_video_data.json"))
def get_time_loc(k_i,i):
    k,_ = k_i.split("_")
    v = ori_infos[k+".mp4"]
    if(k=="411766054621"):v["frames"][0]["time"][0]=5
    times = [f["time"][1]-f["time"][0] for f in v["frames"]]
    all_len = sum(times)
    t_time  = [f["time"][1]-f["time"][0] for f in v["frames"][:i+1]]
    t_len = sum(t_time)-(v["frames"][i]["time"][1]-v["frames"][i]["time"][0])/2
    return int((t_len/all_len)*100)
for j,info in enumerate(fin):
    ori_t_input = info["inputs"].replace("场景：<vid_start><vid_end>\n","")
    t_input,processed = process(ori_t_input)
    if(processed or j==0):
    #print(ori_t_input==t_input)
        print("replace")
    f_num = len(info["frames"])
    for i,t_f in enumerate(info["frames"]):
        t_pos = get_time_loc(t_f,i)
        if(t_pos<0 or t_pos >99):print(t_f,t_pos,"no")
        if(i==0):t_pos = 0
        if(i==f_num-1):t_pos = 99
        #t_input = remove_label(t_input)
        t_out={"shot_pos":t_pos,"shot_loc":(i+1),"inputs":"","id":info["id"],"frames":[t_f],"images":[info["images"][i]]}
        t_out["inputs"]=t_input.replace("镜头{}/{}：\n".format(i+1,f_num),"镜头{}/{}：\n场景：<vid_start><vid_end>\n".format(i+1,f_num))
        out.append(t_out)
    #if(j>50):break
json.dump(out,fout,ensure_ascii=False)
