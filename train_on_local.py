#    Based on Video-ChatGPT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import time
import copy
import logging
import os
import json
import pickle
import torch.distributed as dist
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from train.llava_trainer import VideoChatGPTTrainer

os.environ['ENABLE_FLOPS_STATISTICS'] = '-1'
os.environ['MODEL_PARAMS_THOP'] = '-1'
os.environ['NCCL_MIN_NCHANNELS'] = '16'

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset

from psutil import cpu_percent, virtual_memory
from tokenization_baichuan import BaiChuanTokenizer
from modeling_lmm_baichuan import BaiChuanVideoCapLlamaForCausalLM
from modeling_lmm import VideoCapLlamaForCausalLM, VideoCapLlamaModel

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
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    video_data_path: str = field(default=None, metadata={"help": "Path to the video features."})
    pretrain_path: str = field(default=None, metadata={"help": "Path to the pretrained model."})
    train_type: str = field(default="ch", metadata={"help": "ch/en model"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_video_conv_front: bool = False
    video_token_len: int = 0
    video_folder: Optional[str] = field(default=None)
    frame_aspect_ratio: str = 'square'


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
    lora_enable: bool = field(default=False)
    resume_path: str = field(default="None")


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


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_multimodal(
        sources: Sequence[str],
        cur_token_len: int,  # 视频镜头的token长度
        train_type: str
) -> Dict:
    video_token_len = cur_token_len

    # 将video_embedding的位置先占住
    for i, source in enumerate(sources):
        # for sentence in source:
        replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
        replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
        if (train_type == "en"):
            sources[i] = PROMPT_DICT["en_instruction_new"] + source.replace(
                DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN, replace_token)
        else:
            sources[i] = PROMPT_DICT["instruction"] + source.replace(DEFAULT_VID_START_TOKEN + DEFAULT_VID_END_TOKEN,
                                                                     replace_token)

    return sources


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        shot_loc: int,
        train_type: str
) -> Dict:
    """
    :param shot_loc: 镜头的位置，为0表示对所有镜头都计算loss，否则对第i个镜头计算loss
    :param train_type: en/ch
    """
    # conv = conversation_lib.default_conversation.copy()

    # Apply prompt templates

    # Tokenize conversations
    input_ids = tokenizer(
        sources,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    # assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    # print("token_length",tokenizer.model_max_length)
    for source, target in zip(sources, targets):
        # 若训练数据的输入发生变化，此处的数据处理进行相应改变
        instrutions = "\n\n".join(source.split("\n\n")[:-1]) + "\n\n"
        if (train_type == "en"): instrutions = "\n\n ".join(source.split("\n\n ")[:-1]) + "\n\n"
        instruct_length = len(tokenizer(instrutions).input_ids)

        cur_len = instruct_length
        target[:cur_len] = IGNORE_INDEX

        split_text = "个字\n剧本类型："
        frames = source.split("\n\n")[-1].split(split_text)
        if (train_type == "en"):
            # split_text = "Response:\n "
            split_text = "\n Narrative "  # "Narrative:\n "
            frames = source.split("\n\n ")[-1].split(split_text)

        # 如果是进行global一次性生成, 计算所有镜头的generation loss
        if (isinstance(shot_loc, list)):
            shot_loc = 0
        # 如果是分镜头进行生成
        for i, t_f in enumerate(frames[:-1]):
            t_info = t_f + split_text
            info_len = len(tokenizer(t_info).input_ids) - 1
            if (train_type == "en"): info_len -= 1
            if (i == 0):
                target[cur_len: cur_len + info_len] = IGNORE_INDEX
            else:
                # 如果只计算一个镜头的话，对于该镜头之外的都要设为ignore
                if (shot_loc != 0 and i != shot_loc):
                    target[cur_len: cur_len + info_len] = IGNORE_INDEX
                else:
                    t_out = t_f.split("\n")[0] + "\n" + t_f.split("\n")[1]
                    if (train_type == "en"): t_out = t_f.split("\n")[0]
                    out_len = len(tokenizer(t_out + "\n").input_ids) - 1
                    target[cur_len + out_len: cur_len + info_len] = IGNORE_INDEX
            cur_len += info_len
        # 如果只计算一个镜头的话，对于该镜头之外的都要设为ignore
        if (shot_loc != 0 and shot_loc < len(frames) - 1):
            target[cur_len:] = IGNORE_INDEX
        else:
            cur_len += (len(tokenizer(frames[-1]).input_ids) - 1)
            target[cur_len:] = IGNORE_INDEX

    # print("target",target)

    return dict(
        input_ids=input_ids,
        labels=targets
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, video_data_path: str, train_type: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        self.video_data_path = video_data_path
        self.train_type = train_type

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video_fea = []
        shot_loc = 0
        # 如果视频存在多个镜头
        if 'frames' in sources[0]:
            cur_token_len = 64
            for video_file in self.list_data_dict[i]['frames']:
                video_folder = self.video_data_path
                with open(f"{video_folder}/{video_file}", "rb") as f:
                    t_fea = torch.tensor(pickle.load(f))
                    # 如果需要批量处理多个镜头的数据，padding使得长度一致
                    if ('shot_loc' in self.list_data_dict[i] and isinstance(self.list_data_dict[i]['shot_loc'], list)):
                        if (cur_token_len > t_fea.size(0)):
                            t_fea = torch.cat([t_fea, torch.zeros(cur_token_len - t_fea.size(0), t_fea.size(1))])
                        else:
                            t_fea = t_fea[:cur_token_len, :]
                    video_fea.append(t_fea)

            cur_token_len = video_fea[0].size(0)
            sources = preprocess_multimodal(
                copy.deepcopy([(e["inputs"] + "\n") for e in sources]),
                cur_token_len,
                train_type=self.train_type
            )
        # 读取镜头的绝对位置
        if 'shot_loc' in self.list_data_dict[i]:
            shot_loc = self.list_data_dict[i]['shot_loc']
        else:
            shot_loc = 0

        data_dict = preprocess(
            sources,
            self.tokenizer,
            shot_loc=shot_loc,
            train_type=self.train_type
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # video exist in the data
        if 'frames' in self.list_data_dict[i]:
            data_dict['frames'] = torch.stack(video_fea)

        # 读取镜头的相对位置（0-99)
        if 'shot_pos' in self.list_data_dict[i]:
            data_dict['frame_pos'] = torch.tensor(self.list_data_dict[i]['shot_pos'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        features = []
        for instance in instances:
            features.append(instance['frames'])
        if 'frames' in instances[0]:
            features = [instance['frames'].clone().detach() for instance in instances]
            frame_pos = [instance['frame_pos'].clone().detach() for instance in instances]
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['video_spatio_temporal_features'] = torch.stack(features)
            else:
                batch['video_spatio_temporal_features'] = features
            batch['video_frame_position'] = frame_pos
        torch.set_printoptions(profile="full")
        # print("fea",batch['video_spatio_temporal_features'].shape)
        # print("input_ids",batch['input_ids'].shape)
        # print("input_ids",batch['input_ids'])
        # print("labels", batch['labels'])

        return batch


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", True):
        # Only save Adapter
        print("only save projecter")
        keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          video_data_path=data_args.video_data_path,
                                          train_type=data_args.train_type,
                                          multimodal_cfg=dict(
                                              is_multimodal=True,
                                              sep_video_conv_front=False,
                                              # video_token_len=data_args.video_token_len,
                                              # video_folder=data_args.video_folder,
                                              # frame_aspect_ratio=data_args.frame_aspect_ratio,
                                              use_vid_start_end=True))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(time.localtime(time.time()), '-' * 30 + 'model args', model_args)
    print('-' * 30 + 'data args', data_args)
    print('-' * 30 + 'trainig args', training_args)

    training_args.output_dir = '%s' % (training_args.output_dir)
    training_args.save_total_limit = 2


    load_dir = data_args.pretrain_path  # './pretrain/'

    local_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if (data_args.train_type == "ch"):
        model = BaiChuanVideoCapLlamaForCausalLM.from_pretrained(
            load_dir,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float,
        )
        print('+' * 30, 'model', model)
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

        print('+' * 30, 'model', model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            load_dir,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    print('+' * 30, 'tokenizer', tokenizer)

    if (training_args.lora_enable):
        lora_config = LoraConfig(
            r=64,  # training_args.lora_r,
            lora_alpha=16,  # training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.05,  # training_args.lora_dropout,
            bias="none",  # training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("initialize lora model")

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
    print(time.localtime(time.time()), '-' * 30 + 'tokenzier done', tokenizer)
    print('load tokenizer and model,cpu percent', cpu_percent(interval=2))
    print('memory', virtual_memory())


    pretrain_mm_mlp_adapter = None
    model_vision_dict = model.get_model().initialize_vision_modules(
        pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter  # model_args.pretrain_mm_mlp_adapter
    )
    vision_config = model_vision_dict['vision_config']

    data_args.video_token_len = model_vision_dict['video_token_len']
    data_args.is_multimodal = True

    tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter

    if tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.use_cache = False

    mm_use_vid_start_end = True
    # model.config.sep_video_conv_front = data_args.sep_video_conv_front

    model.initialize_vision_tokenizer(mm_use_vid_start_end=mm_use_vid_start_end, tokenizer=tokenizer,
                                      device=training_args.device, tune_mm_mlp_adapter=tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter)

    #params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]

    # load数据
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    print(time.localtime(time.time()), '-' * 30 + 'data module done', data_module)


    trainer = VideoChatGPTTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    # print(time.localtime(time.time()),'-'*15+'finetune firefly model'+'-'*15)
    torch.distributed.init_process_group(backend='nccl')
    train()
