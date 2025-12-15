# PII检测器
## 介绍
本工具是以Qwen3-14B为基座模型，使用ai4privacy_500k训练集进行微调，并在ai4privacy_500k验证集测试，测试结果放在`run_llm/eval_results_sft5`文件夹中。
### 工作流程
- 基座模型的选定：选取目前主流的开源LLM，大小为4B-15B之间，使用`ai4privacy_300k`、`ontonotes5`、`conll2003`三个专门用于训练PII识别和NER任务的数据集进行测试，选取基础能力最好的模型进行微调。
    - 测试代码放在`model_path/code`文件夹中。
    - 测试结果放在`run_llm/dataset_name/model_name`文件夹中。
- PII范围划分：根据调研，我们发现数据集中某些标签单独出现并不构成PII，因此我们只考虑以下PII：
    - `Location`：粒度小于等于街道的地点或地址
    - `Email`：电子邮箱
    - `Name`：包括人的姓、名和用户名
    - `PhoneNumber`：电话号码
    - `TaxNumber`：税收代码
    - `SocialSecurityNumber`：社会安全号码
    - `DriveLicenseNumber`：驾驶证号码
    - `IDCardNumber`：身份证号码
    - `CreditCardNumber`：信用卡号码
- 模型的微调与评估
## Qwen3-14B 微调
### 1. 模型下载
- 如果可以直接访问到huggingface，
    - `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3-14B`
    - `cd Qwen3-14B`
    - `git lfs pull`
- 使用huggingface-cli
    - `pip install -U huggingface_hub`
    - 在`root/.bashrc`中添加`export HF_ENDPOINT=https://hf-mirror.com`
    - `huggingface-cli download --resume-download Qwen/Qwen3-14B --local-dir Qwen3-14B`

### 2. llama-factory 安装
**为了避免依赖版本号冲突，建议先conda创建新的虚拟环境再执行以下操作。**
- `git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git`
- `cd LLaMA-Factory`
- `pip install -e ".[torch,metrics]"`

### 3. 检查模型路径
- 检查`qwen3-14b_lora_sft.yaml`里的`model_name_or_path`,这里是默认模型在和该文件在同一文件夹，如果不在请自行修改。
### 4. 运行训练文件
- `cd qwen3-14b_sft`
- 按需修改`CUDA_VISIBLE_DEVICES`
- `./train.sh`

## 模型使用
### 1. tokenizer+peft
**具体代码参考`test.py`**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base_model_path = "./Qwen3-14B"           # 基座模型
adapter_path = "saves/qwen3-14b/lora/sft_5" # 微调后的权重
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# 3. 定义测试函数
def predict(text):
    # 构造 Prompt
    # LLaMA-Factory 在处理 Alpaca 格式时，通常会将 instruction 和 input 拼接作为 User 输入
    messages = [
        {"role": "user", "content": f"{TRAIN_INSTRUCTION}\n\n{text}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.1, 
            top_p=0.9
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

result = predict(text)
```

### 2. LlamaFactory+tokenizer+vllm (推理速度最快)
- 参考LlamaFactory文档合并权重
- 使用vllm加快推理 (具体代码参考`rum_llm/eval.py`):
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(args.model_path,use_fast=True)
llm = LLM(model=args.model_path,tensor_parallel_size=args.tensor_parallel_size,seed=args.seed,gpu_memory_utilization=args.gpu_memory_utilization)
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1024,
    repetition_penalty=1.05,
    stop=["\n\n\n","###","```python","<END>"],
    n=1,
    skip_special_tokens=True,
    )


prompt=f'{instruction}\n\n{texts}'
messages=[{"role": "user", "content": prompt}]
inputs=tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
    )
batch_resp=model.generate(inputs,sampling_params)
```



