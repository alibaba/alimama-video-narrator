import time
import copy
import logging
import os
import json
import pickle
from tqdm import tqdm
os.environ['ENABLE_FLOPS_STATISTICS'] = '-1'
os.environ['MODEL_PARAMS_THOP'] = '-1'
os.environ['NCCL_MIN_NCHANNELS'] = '16'

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import sys

import torch
import transformers

from psutil import cpu_percent,virtual_memory
from tokenization_baichuan import BaiChuanTokenizer
from modeling_lmm import VideoCapLlamaForCausalLM, VideoCapLlamaModel
from modeling_lmm_baichuan import BaiChuanVideoCapLlamaForCausalLM


IGNORE_INDEX = -100
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "instruction":
        "你是一位广告商，擅长为商品撰写具有吸引力的广告文案，并且能够准确理解商品信息和视频信息。\n我将为你提供商品信息以及该商品对应的广告视频的每个镜头的信息。你的任务是结合商品信息，为视频的每个镜头撰写符合当前字数要求的广告文案，吸引消费者购买。你需要判断每个镜头的剧本类型，确保文案围绕镜头的剧本类型展开，与当前场景紧密相关，并且与前面的文案保持连贯。\n\n请注意，在撰写文案时，主要使用到提供的商品信息以及镜头的场景信息。\n\n",
    "instruction_no_label":
        "你是一位广告商，擅长为商品撰写具有吸引力的广告文案，并且能够准确理解商品信息和视频信息。\n我将为你提供商品信息以及该商品对应的广告视频的每个镜头的信息。你的任务是结合商品信息，为视频的每个镜头撰写符合当前字数要求的广告文案，吸引消费者购买。你需要确保文案与当前场景紧密相关，并且与前面的文案保持连贯。\n\n请注意，在撰写文案时，主要使用到提供的商品信息以及镜头的场景信息。\n\n",
    "en_instruction":
        "You are an AI assistant that can understand videos. You can observe multiple video shots and write a coherent story based on them. You are given sequential video shots, along with the corresponding visual scene and the word count requirement. Your task is to create a narrative for each video shot. Make sure the narrative is highly related to the visual scene, meets the word count requirement, and maintains consistency with the preceding narratives.\n\n ",
    "en_instruction_new":
        "You are an AI assistant that can understand videos. You are given sequential video shots related to a specific event, along with the corresponding visual scene and the word count requirement. Your task is to create a narrative for each video shot. Ensure that the narrative closely aligns with the visual scene, meets the word count requirement, and maintains consistency with the preceding narratives.\n\n ",
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_vid_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    video_data_path: str = field(default="/video_feas_all", metadata={"help": "Path to the test data."})
    pretrain_path: str = field(default="/pretrain/", metadata={"help": "Path to the pretrain model."})
    test_data: str = field(default="/data/test.json", metadata={"help": "Path to the inference file."})
    video_infos: str = field(default="/data/all_video_data.json", metadata={"help": "Path to the video info file."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_video_conv_front: bool = False
    video_token_len: int = 0
    video_folder: Optional[str] = field(default=None)
    frame_aspect_ratio: str = 'square'
    chk_path: str = field(default=None, metadata={"help": "Path to the saved model."})
    one_shot_vision: bool = False
    no_label: bool = False
    offered_label: bool = False
    infer_type: str = field(default="ch", metadata={"help": "inference type: ch or en"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens


def video_chatgpt_infer(inputs, model, tokenizer, video_spatio_temporal_features, video_frame_position, temperature=1.0, repetition_penalty=1.1,cap_eos_id=5):
    """
    :param inputs:
    :param model:
    :param tokenizer:
    :param video_spatio_temporal_features:
    :param video_frame_position:
    :param temperature:
    :param repetition_penalty:
    :param cap_eos_id:
    :return:
    """


    # Tokenize the prompt
    inputs = tokenizer(inputs)

    # Move inputs to GPU
    input_ids = copy.deepcopy(torch.as_tensor(inputs.input_ids).cuda())
    #print(input_ids.shape)

    # Define stopping criteria for generation

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            video_frame_position=video_frame_position,
            do_sample=True,
            # num_beams=1,
            temperature=temperature,
            # top_p=0.9,
            repetition_penalty=repetition_penalty,
            eos_token_id=cap_eos_id,
            max_new_tokens=100)

    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Clean output string
    outputs = outputs.strip()#.rstrip(stop_str).strip()

    return outputs

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



if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    load_dir =  data_args.pretrain_path
    if(data_args.infer_type=="en"):load_dir = data_args.pretrain_path


    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if (data_args.infer_type == "ch"):
        model = BaiChuanVideoCapLlamaForCausalLM.from_pretrained(
            load_dir,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float,
        )
        tokenizer = BaiChuanTokenizer.from_pretrained(
            load_dir,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        model = VideoCapLlamaForCausalLM.from_pretrained(
            load_dir,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            load_dir,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    model_vision_dict = model.get_model().initialize_vision_modules(
        pretrain_mm_mlp_adapter= None # model_args.pretrain_mm_mlp_adapter
    )
    vision_config = model_vision_dict['vision_config']


    model_path = data_args.chk_path
    model_base = data_args.pretrain_path
    if (data_args.infer_type == "en"):model_base = data_args.pretrain_path

    mm_use_vid_start_end = True
    model.initialize_vision_tokenizer(mm_use_vid_start_end=mm_use_vid_start_end, tokenizer=tokenizer,
                                      device=training_args.device, tune_mm_mlp_adapter=False,
                                      pretrain_mm_mlp_adapter=None)

    cap_eos_id = 5
    if(data_args.infer_type == "en"):cap_eos_id=13

    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(
            torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(
            torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    # 加载lora finetune的模型
    print('Loading additional weights...')
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),map_location='cpu')
    else:
        # this is probably from HF Hub
        print("non lora model")
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                           non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                               non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')
    

    device = torch.device(0)
    model.to(device)
    # 使用mean_pooling方式所得的视频镜头编码长度
    cur_token_len = 64

    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * cur_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
    fin = json.load(open(data_args.test_data))
    if (data_args.infer_type == "en"):fin = json.load(open(data_args.test_data))
    max_frame=20
    out_result = {}
    out_frames = {}
    test_length = len(fin)
    ori_infos = json.load(open(data_args.video_infos))
    if (data_args.infer_type == "en"):ori_infos = json.load(open("data/video_story/final_data.json"))


    #根据视频的各镜头的时间段计算每个镜头的相对位置
    def get_time_loc(v_k, frame_i):
        v_info = ori_infos[v_k + ".mp4"]
        times = [f["time"][1] - f["time"][0] for f in v_info["frames"]]
        all_len = sum(times)
        t_time = [f["time"][1] - f["time"][0] for f in v_info["frames"][:frame_i + 1]]
        t_len = sum(t_time) - (v_info["frames"][frame_i]["time"][1] - v_info["frames"][frame_i]["time"][0]) / 2
        return int((t_len / all_len) * 100)

    for t_id in tqdm(range(test_length),total=test_length):
        video_fea = []
        t_frame = 0
        for video_file in fin[t_id]['frames']:
            video_folder = data_args.video_data_path
            if(data_args.infer_type =="en"): video_folder = data_args.video_data_path
            t_frame+=1
            if(t_frame>max_frame):break
            try:
                with open(f"{video_folder}/{video_file}", "rb") as f:
                    t_fea = torch.tensor(pickle.load(f))
                    #若是一次性生成多个镜头，pad为同样的长度
                    if("shot_pos" in fin[t_id] or data_args.one_shot_vision==False):
                        if(t_fea.size(0)<64):
                            video_fea.append(torch.cat([t_fea, torch.zeros(64 - t_fea.size(0), t_fea.size(1))]))
                        else:
                            video_fea.append(t_fea[:cur_token_len,:])
                    else:video_fea.append(t_fea)#
            except:
                print("load false")
                continue
        if(len(video_fea)==0):
            print("no feature")
            continue
        if (data_args.one_shot_vision):
            if (data_args.infer_type == "ch"):
                #若按镜头顺序依次生成，等到生成该镜头时再加对应的<vid_start><vid_end>读入视觉信息
                full_infos = fin[t_id]["inputs"].replace("场景描述","场景").replace("场景：<vid_start><vid_end>\n", "")
            else:
                full_infos = fin[t_id]["inputs"].replace("visual scene: <vid_start><vid_end>\n ", "")
        else:
            full_infos = fin[t_id]["inputs"].replace("场景描述","场景").replace(DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN,
                                                                     replace_token)
        video_spatio_temporal_features = video_fea

        frames = full_infos.split("个字\n")
        rep=1.1
        if(data_args.infer_type=="ch"):
            print(fin[t_id]["link"])
            print(fin[t_id]["p_info"])
        else:
            frames = full_infos.split(" words\n ")
            print(fin[t_id]["id"])
        video_id = fin[t_id]["id"]

        skip = False
        # 从test file读入gt信息
        if "shot_pos" in fin[t_id]:
            for t_i,t_cap in enumerate(fin[t_id]["inputs"].split("\n\n")[-1].split("\n")):
                out_frames[video_id + "_" + str(t_i)] = {"gt": t_cap}
        else:
            for t_i, frame in enumerate(frames[1:]):
                if (data_args.infer_type == "ch"):
                    try:
                        t_cap = frame.split("\n")[1].split(str(t_i + 1) + ".")[1]
                    except:
                        skip = True
                if(data_args.infer_type=="en"):
                    t_cap = frame.split(": ")[1].split("\n ")[0]
                    #t_cap = frame.split("\n ")[1]
                out_frames[video_id + "_" + str(t_i)] = {"gt":t_cap}
        if(skip):continue


        for rep in [1.1]:
            for temp in [0.1]:
                t_input = PROMPT_DICT["instruction"]
                if(data_args.infer_type =="en"): t_input = PROMPT_DICT["en_instruction_new"]
                print("temp:{}, rep:{}".format(temp,rep))
                #一次性生成所有的文案
                if("shot_pos" in fin[t_id]):
                    input_video_features = torch.stack(video_spatio_temporal_features).to(torch.float).to(device)  # [t_i:t_i + 1]
                    cur_token_len = input_video_features.shape[-2]
                    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * cur_token_len
                    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
                    t_input+= "\n\n".join(fin[t_id]["inputs"].split("\n\n")[:-1])+"\n\n"
                    t_input = t_input.replace(
                        DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN,
                        replace_token)
                    video_frame_position = torch.tensor(fin[t_id]["shot_pos"]).unsqueeze(0).to(device)
                    t_out = video_chatgpt_infer([t_input], model, tokenizer, input_video_features, video_frame_position,
                                                temperature=temp, repetition_penalty=rep, cap_eos_id=None)

                    print(t_out)
                    out_caps = t_out.split("\n")
                    for t_i,out_cap in enumerate(out_caps):
                        if video_id + "_" + str(t_i) in out_frames and "." in out_cap:
                            out_frames[video_id + "_" + str(t_i)][temp] = out_cap.split(".")[1]
                    continue
                f_num = len(ori_infos[video_id+".mp4"]["frames"])
                all_frame_loc = []
                #按镜头顺序依次生成文案
                for t_i, frame in enumerate(frames[:-1]):
                    t_cats = frame
                    if (t_i+1 > max_frame): break
                    if(t_i>0):
                        t_cats = "\n".join(frame.split("\n")[2:])
                        if(data_args.infer_type=="en"):t_cats = "\n ".join(frame.split("\n ")[1:])

                    out_label = "剧本类型"
                    if(data_args.one_shot_vision):
                        input_video_features = video_spatio_temporal_features[t_i]
                        input_video_features = input_video_features.to(torch.float).to(device).unsqueeze(0)#[t_i:t_i + 1]
                        cur_token_len = input_video_features.shape[-2]
                        replace_token = DEFAULT_VIDEO_PATCH_TOKEN * cur_token_len
                        replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
                        if (data_args.infer_type == "en"):
                            t_input += (t_cats + " words\n ")
                        else:
                            t_input += (t_cats + "个字\n剧本类型：")
                        if(data_args.infer_type=="ch"):
                            t_input = t_input.replace("镜头{}/{}：\n".format(t_i + 1,f_num), "镜头{}/{}：\n场景：<vid_start><vid_end>\n".format(t_i + 1,f_num)).replace(DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN,
                                                                     replace_token)
                        else:
                            #t_input = t_input.replace("Video shot {}:\n ".format(t_i + 1), "Video shot {}:\n Visual scene: <vid_start><vid_end>\n ".format(t_i + 1)).replace(DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN, replace_token)
                            t_input = t_input.replace("Video shot {}:\n ".format(t_i + 1),
                                                      "Video shot {}:\n visual scene: <vid_start><vid_end>\n ".format(
                                                          t_i + 1)).replace(
                                DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN,
                                replace_token).replace("Word count requirement", "word count requirement")
                        t_input=t_input.replace("  "," ")
                        #print(t_input)
                        frame_pos = get_time_loc(video_id,t_i)
                        if(t_i==0):frame_pos = 0
                        if(t_i == len(frames)-2):frame_pos=99
                        video_frame_position = torch.stack([torch.tensor(frame_pos)]).to(device)
                        print("loc",frame_pos)
                        if(data_args.infer_type=="ch"):
                            if(data_args.offered_label):
                                out_label = frames[t_i+1].split("\n")[0].split("剧本类型：")[-1]
                            else:
                                out_label = video_chatgpt_infer([t_input], model, tokenizer, input_video_features, video_frame_position,temperature=temp,repetition_penalty=1,cap_eos_id=cap_eos_id)
                            out_label = out_label.split("\n")[0]
                            t_input += (out_label + "\n{}.".format(t_i + 1))
                        else:t_input+="Narrative {}:".format(t_i + 1)#t_input+="### Response:\n"
                        #print("t_input",t_input)
                        t_out = video_chatgpt_infer([t_input], model, tokenizer, input_video_features, video_frame_position,temperature=temp,repetition_penalty=rep,cap_eos_id=cap_eos_id)

                        if (data_args.infer_type == "ch"):
                            t_input = t_input.replace(replace_token, DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN).replace("场景：<vid_start><vid_end>\n", "")
                        else:
                            #t_input = t_input.replace(replace_token,
                            #                          DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN).replace(
                            #    "Visual scene: <vid_start><vid_end>\n ", "")
                            t_input = t_input.replace(replace_token,
                                                      DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN).replace(
                                "visual scene: <vid_start><vid_end>\n ", "")
                    else:
                        frame_pos = get_time_loc(video_id, t_i)
                        if (t_i == 0): frame_pos = 0
                        if (t_i == len(frames) - 2): frame_pos = 99
                        all_frame_loc.append(torch.tensor(frame_pos))
                        video_frame_position = torch.stack(all_frame_loc).to(device)
                        print("loc", video_frame_position)
                        input_video_features=torch.stack(video_spatio_temporal_features[:t_i+1]).to(torch.float).to(device)
                        t_input += (t_cats + "个字\n剧本类型：")
                        if (data_args.offered_label):
                            out_label = frames[t_i + 1].split("\n")[0].split("剧本类型：")[-1]
                        else:
                            t_temp=0.1
                            out_label = video_chatgpt_infer([t_input], model, tokenizer, input_video_features,video_frame_position, temperature=t_temp,repetition_penalty=1, cap_eos_id=cap_eos_id)
                        out_label = out_label.split("\n")[0]
                        t_input += (out_label + "\n{}.".format(t_i + 1))
                        #print(t_input)
                        t_out = video_chatgpt_infer([t_input], model, tokenizer, input_video_features, video_frame_position, temperature=temp, repetition_penalty=rep, cap_eos_id=cap_eos_id)
                    if(data_args.infer_type=="ch"):
                        t_cap = "\n".join(t_out.split("\n")[0:2])
                        t_label = frames[t_i+1].split("\n")[0].split("剧本类型：")[-1]
                        t_range = frame.split("字数要求：")[-1]
                        t_word_num = len(t_cap.split("\n")[-1])
                        out_frames[video_id + "_" + str(t_i)]["label"] = t_label
                        print(t_label, t_range + " words", out_frames[video_id + "_" + str(t_i)]["gt"])
                        out_frames[video_id + "_" + str(t_i)]["out_label"] = out_label
                        print(out_label)
                        print(str(t_i + 1) + "." + t_cap + "(" + str(t_word_num) + ")")
                        out_frames[video_id + "_" + str(t_i)][temp] = t_cap
                    if(data_args.infer_type=="en"):
                        t_label = " "
                        #print("t_out",t_out)
                        t_cap = t_out.split("\n")[0]
                        if (": " in t_out): t_cap = t_out.split(": ")[1]
                        t_range = frame.split("word count requirement: ")[-1]
                        t_word_num = len(t_cap.split(" "))
                        print("gt",t_label, t_range + " words", out_frames[video_id + "_" + str(t_i)]["gt"])
                        print(str(t_i + 1) + "." + t_cap + "(" + str(t_word_num) + ")")
                        out_frames[video_id + "_" + str(t_i)][temp] = t_cap
                    #out_result[video_id + ".mp4"]["frames"][t_i]["new_text"] = t_cap

                    if(data_args.infer_type =="en"):t_input +=(" "+t_cap+"\n ")
                    else:t_input += (t_cap+"\n")
    if(data_args.offered_label):
        with open(data_args.chk_path+"/output_label.json","w") as fout:
            json.dump(out_frames,fout,ensure_ascii=False,indent=2)
    else:
        with open(data_args.chk_path+"/output.json","w") as fout:
            json.dump(out_frames,fout,ensure_ascii=False,indent=2)





