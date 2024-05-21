#!/bin/bash
#
#SBATCH --job-name=dual
#SBATCH --error=/datasets/jchen293/logs/exp/llava/1dpool8_2_2layer2_2_16pivot1300_2600_3900prog_dual.err
#SBATCH --output=/datasets/jchen293/logs/exp/llava/1dpool8_2_2layer2_2_16pivot1300_2600_3900prog_dual.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --partition=main
#SBATCH --exclude=ccvl[14,30-34,37-38]
#SBATCH --cpus-per-task=4

# Path to output directory
OUTPUT=/datasets/jchen293/logs/exp/llava/

# Set the number of GPUs per node
export GPUS_PER_NODE=8

# Define the task name
TASK_NAME="dual"

# Create the output directory
export CPT="$OUTPUT/$TASK_NAME"
mkdir -p $CPT

# Create a script file in the output directory
SCRIPT=$CPT/script.sh

# Number of SLURM tasks
echo "SLURM_NTASKS=$SLURM_NTASKS"

# Define hostfile and update it with node and slot information
export HOSTFILE=".hostfile"
echo "[slurm node name] slots=$SLURM_NTASKS" > $HOSTFILE

# Declare an associative array
declare -A IBO

# Populate the array with node IP addresses
node_count=2  # Assuming there are 2 nodes, adjust as necessary
for i in $(seq 0 $node_count); do
    key=$(printf "NODE_NAME%02d" "$i")
    value=$(printf "xx.xx.xx.%02d" "$i")
    IBO["$key"]="$value"
done

# Extract the number from SLURM_JOB_NODELIST and echo it
regex="ccvl35,ccvl36"
if [[ $SLURM_JOB_NODELIST =~ $regex ]]; then
    extracted_number="${BASH_REMATCH[0]}"
    echo "Extracted number: $extracted_number"
else
    echo "No match found"
fi

# Export keys and values for further use
export MASTER_KEY="NODE_NAME$extracted_number"
export MASTER_PORT=31217


# export MASTER_PORT=12802
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr



export WANDB_API_KEY='46e587ae4112a04da96b68ba807395204be787c9'
export WANDB_PROJECT='llava_team'
export WANDB_ENTITY='jchen293'

module purge
module load conda
conda activate llava_git

ROOT_DATA=/datasets/jchen293/data/llava_datasets
ROOT_WEIGHT=/datasets/jchen293/weights/llava/checkpoint

layers=2,2,16,0
strides=8,2,2,1
pivots=1300,2600,3900
grouping=avgpool1d
unified_vpe=False
progressive=True
name=1dpool8_2_2layer2_2_16pivot1300_2600_3900prog_dual


deepspeed --hostfile=$HOSTFILE llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path $ROOT_DATA/LLaVA-Tuning/llava_v1_5_mix665k.json \
    --image_folder $ROOT_DATA/LLaVA-Tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $ROOT_WEIGHT/llava-v1.5-7b-pretrain-reprod/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir  $ROOT_WEIGHT/llava-v1.5-7b-$name \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $name \
    --strides $strides \
    --layers $layers \
    --pivots $pivots \
    --grouping $grouping \
    --unified_vpe $unified_vpe \
    --progressive $progressive \