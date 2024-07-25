#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=1 # number of nodes to use
#SBATCH --gpus-per-node=1
#SBATCH --job-name=tune # name of your job
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=0 # all memory on the node

set -ex;



###############
## Environment Variables ##
###########################

#export NCCL_SOCKET_IFNAME=en
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# For EFA
export NCCL_DEBUG=INFO
export FI_PROVIDER="efa"



# async runtime error ...
export CUDA_DEVICE_MAX_CONNECTIONS=1



#########################
## Command and Options ##

declare -a ARGS=(
    --container-image ${PWD}/llm.sqsh
    --container-mounts ${PWD}
    --container-workdir /home/ubuntu \
    --container-mount-home
)


export GPUS_PER_NODE=1
export NNODES=1
export NUM_PROCESSES=1

export LAUNCHER="accelerate launch \
    --num_processes $NUM_PROCESSES \
    --mixed_precision no \
    --dynamo_backend no \
    "

export SCRIPT="sft.py"

export RUN_NAME="Meta-Llama-3.1-8B-SFT-$(date +%Y-%m-%d-%H-%M)"

export SCRIPT_ARGS=" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_name="HuggingFaceH4/no_robots" \
    --output_dir /home/ubuntu/Meta-Llama-3.1-8B-SFT \
    --report_to="wandb" \
    --run_name $RUN_NAME \
    --push_to_hub true \
    --hub_model_id "<user_name>/llama3.1-8b-sft" \
    --bf16 true \
    --learning_rate=1e-05 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --logging_steps=1 \
    --eval_strategy="no" \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --optim adamw_hf \
    --attn_implementation="flash_attention_2" \
    --torch_dtype="bfloat16" \
    --bnb_4bit_quant_type="nf4" \
    --use_peft true \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
 
srun -l "${ARGS[@]}" \
      bash -c 'echo "Node ID $SLURM_NODEID"; $CMD'
