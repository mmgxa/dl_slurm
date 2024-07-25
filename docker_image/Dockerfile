FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

RUN pip install --no-cache-dir \
            transformers "huggingface_hub[cli,hf_transfer]" accelerate einops \
            peft deepspeed trl bitsandbytes datasets wandb && \
    pip install --no-cache-dir flash-attn --no-build-isolation
    
CMD ["/bin/bash"]
