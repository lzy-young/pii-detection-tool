---
library_name: peft
license: other
base_model: /nfs/FM/huhaoxuan/model_hub/Qwen/Qwen3-14B
tags:
- base_model:adapter:/nfs/FM/huhaoxuan/model_hub/Qwen/Qwen3-14B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: sft_5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_5

This model is a fine-tuned version of [/nfs/FM/huhaoxuan/model_hub/Qwen/Qwen3-14B](https://huggingface.co//nfs/FM/huhaoxuan/model_hub/Qwen/Qwen3-14B) on the ai4privacy_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0111

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5

### Training results

| Training Loss | Epoch  | Step  | Validation Loss |
|:-------------:|:------:|:-----:|:---------------:|
| 0.0243        | 0.2655 | 1000  | 0.0263          |
| 0.0173        | 0.5310 | 2000  | 0.0175          |
| 0.0162        | 0.7964 | 3000  | 0.0141          |
| 0.0082        | 1.0619 | 4000  | 0.0122          |
| 0.007         | 1.3273 | 5000  | 0.0116          |
| 0.0101        | 1.5928 | 6000  | 0.0115          |
| 0.009         | 1.8583 | 7000  | 0.0111          |
| 0.006         | 2.1237 | 8000  | 0.0119          |
| 0.0051        | 2.3892 | 9000  | 0.0125          |
| 0.0045        | 2.6547 | 10000 | 0.0119          |
| 0.0062        | 2.9202 | 11000 | 0.0118          |
| 0.0025        | 3.1856 | 12000 | 0.0167          |
| 0.0026        | 3.4511 | 13000 | 0.0158          |
| 0.0008        | 3.7165 | 14000 | 0.0174          |
| 0.0036        | 3.9820 | 15000 | 0.0163          |
| 0.0009        | 4.2474 | 16000 | 0.0224          |
| 0.001         | 4.5129 | 17000 | 0.0232          |
| 0.0002        | 4.7784 | 18000 | 0.0236          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.9.1+cu128
- Datasets 4.0.0
- Tokenizers 0.22.1