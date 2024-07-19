CUDA_VISIBLE_DEVICES=0 python lora_video_infer.py \
    --pretrain_path /pretrained_models/baichuan-7b-sft/ \
    --video_data_path /data_process/video_feas_mem_compress \
    --test_data data/split/test.json \
    --video_infos data/all_video_data.json \
    --output_dir ./ \
    --offered_label False \
    --one_shot_vision True \
    --chk_path tmp_ckpt_lora_mem_compress/checkpoint
