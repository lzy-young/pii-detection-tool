export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=2,3 python ../qwen3-14b-sft5/code/eval_ai4privacy300k.py  \
    --data_path ../ai4privacy/english_data \
    --export_path ./eval_results_sft5 \
    --batch_size 200 \
    --model_path ../qwen3-14b-sft5/ \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.85\
    --seed 42 \
    --debug