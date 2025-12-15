import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 路径设置
base_model_path = "./Qwen3-14B"           # 基座模型
adapter_path = "saves/qwen3-14b/lora/sft" # 微调后的权重

# 2. 加载模型
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", trust_remote_code=True)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# 3. 定义测试函数
TRAIN_INSTRUCTION = """The output JSON uses Chinese keys for PII types.
Examples of PII keys include (but are not limited to):

"Location": [],
"Name": [],
"TaxNumber": [],
"Email": [],
"DriverLicenseNumber": [],
"IDCardNumber": [],
"SocialNumber": [],
"CreditCardNumber": [],
"PhoneNumber": [],

Only include keys that appear in the input text.

All values must be arrays, even if there is only one item.
Output JSON only."""

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

# 4. 开始测试
test_cases = [
    "My phone number is +1-555-0199 and I live in New York.",
    "Please contact support@google.com for help.",
    "There is no private info in this sentence.",  # 负样本测试
    "Mr. Li Si met Zhang San at the Starbucks on 5th Avenue." # 多实体测试
    "My credit card number is 1234-5678-9012-3456. And my TaxNumber is 987-65-4321. And my social number is 123-45-6789." # 信用卡测试
]

print("\n=== 开始测试 ===")
for text in test_cases:
    print(f"\n[Input]: {text}")
    result = predict(text)
    print(f"[Output]: {result}")