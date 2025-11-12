export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../gemma-3-12b-it/code/conll2003.py  \
    --data_path ../conll2003 \
    --export_path ./conll2003_results/gemma3-12b \
    --batch_size 200 \
    --model_path ../gemma-3-12b-it/ \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.9 \
    --seed 42 \