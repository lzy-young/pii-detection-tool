# Qwen3-14B 微调
## 1. 准备工作
### 模型下载
- 如果可以直接访问到huggingface，
    - `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3-14B`
    - `cd Qwen3-14B`
    - `git lfs pull`
- 使用huggingface-cli
    - `pip install -U huggingface_hub`
    - 在`root/.bashrc`中添加`export HF_ENDPOINT=https://hf-mirror.com`
    - `huggingface-cli download --resume-download Qwen/Qwen3-14B --local-dir Qwen3-14B`

### llama-factory 安装
**为了避免依赖版本号冲突，建议先conda创建新的虚拟环境再执行以下操作。**
- `git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git`
- `cd LLaMA-Factory`
- `pip install -e ".[torch,metrics]"`

## 2. 运行程序
### 检查模型路径
- 检查`qwen3-14b_lora_sft.yaml`里的`model_name_or_path`,这里是默认模型在和该文件在同一文件夹，如果不在请自行修改。
### 运行
- `cd qwen3-14b_sft`
- 按需修改`CUDA_VISIBLE_DEVICES`
- `./train.sh`

