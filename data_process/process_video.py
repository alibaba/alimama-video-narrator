import os
import json
from tqdm import tqdm
filepath = "./all_videos/"
files = os.listdir(filepath)
blank_links = []
fout = open("success.txt","w")
suc_f = open("false.txt","w")
for fname in files:
    #link = t_data["link"]
    v_name = fname.split(".")[0]
    cmd_line = "ffmpeg -ss 00:00:00 -i ./all_videos/{} -f image2 -r 1 -t 01:00:00 ./images/video_cuts/{}_%03d.jpg".format(fname,v_name)#"wget {} -P ./all_videos/".format(link)
    try:
        os.popen(cmd_line)
        suc_f.write(fname+"\n")
    except:
        blank_links.append(fname)
        fout.write(fname+"\n")
        continue