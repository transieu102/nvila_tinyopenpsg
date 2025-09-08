#!/bin/bash
export NCCL_P2P_DISABLE="1" #disable for rtx 4000 series gpus which dont support p2p
export NCCL_IB_DISABLE="1" #disable for rtx 4000 series gpus which dont support p2p

RUN_NAME=${1:-"NVILA-Lite-2B-finetune-tiny_openpsg_lora"} #if tune with lora the run name must contains "lora"
GLOBAL_TRAIN_BATCH_SIZE=${2:-"2"}
GRADIENT_ACCUMULATION_STEPS=${3:-"2"}

MODELBASE=${4:-"Efficient-Large-Model/NVILA-Lite-2B"}
DATA_MIXTURE=${5:-"TinyOpenPSGBasic+TinyOpenPSGMultipleChoice"}
OUTPUT_DIR="runs/$RUN_NAME"
echo "OUTPUT_DIR = $OUTPUT_DIR"
NNODES=1
echo "NNODES = $NNODES"
GPUS_PER_NODE=1
echo "GPUS_PER_NODE = $GPUS_PER_NODE"
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=25001
PER_DEVICE_TRAIN_BATCH_SIZE=$((GLOBAL_TRAIN_BATCH_SIZE / NNODES / GPUS_PER_NODE ))
echo "PER_DEVICE_TRAIN_BATCH_SIZE = $PER_DEVICE_TRAIN_BATCH_SIZE"
torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $MODELBASE \
        --data_mixture $DATA_MIXTURE \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_3x3_fix \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model False \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 31000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --vflan_no_system_prompt True \
        --lora_enable True \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_llm True \
        --lora_vt False \
        --report_to wandb
