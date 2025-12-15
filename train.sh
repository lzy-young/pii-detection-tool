export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1,2
export DISABLE_VERSION_CHECK=1
 accelerate launch \
    --config_file fsdp_config.yaml \
    ../LLaMA-Factory/src/train.py qwen3-14b_lora_sft.yaml