from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from argparse import ArgumentParser
import os
import re,gc
import json
from itertools import islice
import datasets


def is_only_special_chars(text):
    """检查是否只包含非字母数字字符"""
    if not text or not text.strip():
        return True
    
    # 只匹配字母和数字，如果没有匹配到，说明只有特殊字符
    return not re.search(r'[a-zA-Z0-9]', text.strip())

def parse_args():
    parser = ArgumentParser(description="Process some integers.")
    parser.add_argument('--data_path', type=str,required=True, help='数据集路径')
    parser.add_argument('--export_path', type=str, help='导出路径')
    parser.add_argument('--model_path', type=str, default='starpii', help='模型路径')
    parser.add_argument('--evaluate_device', type=str, default='cuda:0', help='评估设备')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='张量并行大小')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.80, help='GPU内存利用率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，处理少量数据并显示详细信息')
    return parser.parse_args()


def create_jsonl_prompt(text):
    """创建要求JSONL输出的提示词"""
    prompt = f"""TASK: Extract PII information and output in JSONL format.
===========================================================================================================
INSTRUCTIONS:
- Extract personal information from the INPUT text
- DO NOT make up any information and ONLY extract what is explicitly present in the INPUT provided below
- NEVER output your thoughts, reasoning process, or any explanations, ONLY output the extracted information in JSONL format
- Use this format: {{"value": "extracted_text", "label": "CATEGORY"}} like the EXAMPLES shown below
- MAKE SURE the final output is a valid JSONL format, with one JSON object per line
- You can ONLY label the extracted information with the categories listed in the CATEGORIES LIST section below
- If you think you don't detect any PII, you MUST STOP extraction immediately and respond with: {{"message": "No PII detected"}}
============================================================================================================
CATEGORIES LIST:
- [PER]: Person names (e.g., "John Doe", "Alice")
- [LOC]: Locations including cities, states and countries (e.g., "New York", "Paris", "China", "JAPAN", etc.)
- [ORG]: Organizations such as companies, institutions,  (e.g., "Google", "UNICEF")
============================================================================================================
EXAMPLES:
EXAMPLE1:
INPUT: <B> "John Smith's phone is 123-456-7890 and email is john@example.com, and he lives in New York. He works at OpenAI. \
            But now he is living in Japan." <E>
OUTPUT:
{{"value": "John Smith", "label": "PER"}}
{{"value": "New York", "label": "LOC"}}
{{"value": "OpenAI", "label": "ORG"}}
{{"value": "Japan", "label": "LOC"}}
<END>

EXAMPLE2:
INPUT: <B> "China: 3 - Liao Ziyu ; 4 -  Zhu Ziqing , 5 - Qian Long. " <E>
OUTPUT:
{{"value": "Liao Ziyu", "label": "PER"}}
{{"value": "Zhu Ziqing", "label": "PER"}}
{{"value": "Qian Long", "label": "PER"}}
<END>

END OF EXAMPLES
===============================================================================
INPUT: 
<B> {text} <E>
OUTPUT:
"""
    return prompt


def create_y(labels):
    res={'PER':[],'LOC':[],'ORG':[],'MISC':[]}
    for label in labels:
        if label['label'] in ['PER','LOC','ORG']:
            res[label['label']].append(label['text'])
    return res

def convert_to_jsonl(outputs):
    jsonl_lines=[]
    lines=outputs.strip().split("\n")
    if len(lines)==0 or len(outputs.strip())==0:
        return jsonl_lines
    for line in lines:
        try:
            obj=json.loads(line)
            if obj.get("value", None) is not None and obj.get("label", None) is not None:
                jsonl_lines.append(obj)
        except json.JSONDecodeError:
            continue
    return jsonl_lines


def calculate_metrics(results, labels):
    # 将结果转换为集合 (去重并便于计算)
    metrics = {}
    tp=fp=fn=0
    results_sets={}
    for type in results.keys():
        results_sets[type] = set(results.get(type, []))

    # 将标签转换为集合
    labels_sets={}
    for type in labels.keys():
        labels_sets[type] = set(labels.get(type, []))
    
    all_preds=set()
    all_labels=set()
    for entity_type in ['PER', 'LOC', 'ORG']:
        all_preds.update(results_sets[entity_type])
        all_labels.update(labels_sets[entity_type])
    fn=len(all_labels - all_preds)  # 计算未预测的实际标签数量

    for entity_type in ['PER', 'LOC', 'ORG']:
        predicted = results_sets[entity_type]
        actual = labels_sets[entity_type]
        
        # 使用集合运算计算指标
        tp += len(predicted & actual)  # 交集
        fp += len(predicted - actual)  # 预测但不在实际中
        # fn += len(actual - predicted)  # 实际但未预测
        # 计算精确率和召回率
    precision = tp / (tp+fp) if (tp+fp) > 0 else 0
    recall = tp / (tp+fn) if (tp+fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    metrics = {
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return metrics

def batched(iterable,n):
    it = iter(iterable)
    while True:
        batch=list(islice(it, n))
        if not batch:
            break
        yield batch


def update_results(results_file,results,batch_cnt,completed=False):
    if not os.path.exists(results_file):
        results_data={'batches':{},'batch_cnt':0,'completed':False}
    else:
        with open(results_file,'r',encoding='utf-8') as rf:
            results_data=json.load(rf)
    if completed:
        results_data['completed']=True
        with open(results_file,'w',encoding='utf-8') as wf:
            json.dump(results_data,wf,indent=4)
            return
    results_data['batch_cnt']=batch_cnt
    results_data['batches'][f'batch_{batch_cnt}']=results
    results_data['completed']=completed
    with open(results_file,'w',encoding='utf-8') as wf:
        json.dump(results_data,wf,indent=4)


def build_resume_list(results_path,data_path):
    resume_list=[]
    filename2cnt={}
    result_data={}
    if not os.path.exists(results_path):
        return resume_list
    for file in os.listdir(results_path):
        if not file.endswith('.json'):
            continue
        file_path=os.path.join(results_path,file)
        try:
            with open(file_path,'r',encoding='utf-8') as f:
                result_data=json.load(f)
            is_completed=result_data.get('completed',False)
            batch_cnt=result_data.get('batch_cnt',0)
        except Exception as e:
            batch_cnt=0
        filename=file[:-len('_results.json')]
        filename2cnt[filename]=batch_cnt if not is_completed else -1

    for file in os.listdir(data_path):
        if not file.endswith('.jsonl'):
            continue
        filename=file[:-len('.jsonl')]
        resume_cnt=filename2cnt.get(filename,0)
        if resume_cnt!=-1:
            tp=result_data.get('batches',{}).get(f'batch_{resume_cnt}',{}).get('tp',0)
            fp=result_data.get('batches',{}).get(f'batch_{resume_cnt}',{}).get('fp',0)
            fn=result_data.get('batches',{}).get(f'batch_{resume_cnt}',{}).get('fn',0)
            metrics=[tp,fp,fn]
            resume_list.append((filename,resume_cnt,metrics))
    return resume_list


def process_batch(file_index,data_path,export_path,model,sampling_params,batch_size=1000,
                  batch_cnt=0,metrics=[0,0,0]):
    data_file=os.path.join(data_path,f'{file_index}.jsonl')
    rpath=os.path.join(export_path,f"{file_index}_results.json")
    detail_path=os.path.join(export_path,f"{file_index}_details.json")
    error_count=0
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return
    skip=batch_cnt*batch_size
    dataset =datasets.load_dataset("json",data_files=data_file,streaming=True)
    iter=dataset['train']
    iter=iter.skip(skip)
    fn=metrics[2]
    fp=metrics[1]
    tp=metrics[0]
    for i,batch_items in enumerate(batched(iter,batch_size)):
        batch_index=batch_cnt+i+1
        error_count=0
        details={'PER':[],'LOC':[],'ORG':[],'MISC':[]}
        for item in batch_items:
            result={'PER':[],'LOC':[],'ORG':[],'MISC':[]}
            y_hat=create_y(item['entities'])
            try:
                texts=item['sentence'].strip().split('\n')
                texts=[text for text in texts if not is_only_special_chars(text) and len(text.strip())>0]
                batch_prompts=[create_jsonl_prompt(text) for text in texts]
                batch_resp=model.generate(batch_prompts,sampling_params)
                for resp in batch_resp:
                    for r in resp.outputs:
                        jsonl_outputs=convert_to_jsonl(r.text)
                        for line in jsonl_outputs:
                            if line['label'] in ['PER','LOC','ORG']:
                                result[line['label']].append(line['value'])
                                details[line['label']].append(line['value'])
                metric=calculate_metrics(result,y_hat)
                tp += metric['tp']
                fp += metric['fp']
                fn += metric['fn']
                del texts, y_hat, batch_prompts, batch_resp, result,metric
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"处理文本时出错: {e}")
                torch.cuda.empty_cache()
                error_count+=1
                continue
        recall= tp / (tp + fn) if (tp + fn) > 0 else 0
        precision= tp / (tp + fp) if (tp + fp) > 0 else 0
        f1= 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        res={
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
        update_results(rpath,res,batch_index,completed=False)
        update_results(detail_path,details,batch_index,completed=False)
        print(f'文件{file_index}.json - 批次 {batch_index}处理完成({len(batch_items)-error_count}/{len(batch_items)}) -- 已处理 {batch_index*batch_size} 条数据')
    update_results(rpath,None,None,completed=True)
    update_results(detail_path,None,None,completed=True)
    print(f'文件{file_index}.json 处理完成')
    torch.cuda.empty_cache()

def debug(data_path, model, sampling_params, debug_count=50):
    """
    调试函数：处理前N条数据，显示详细的生成情况
    """
    print(f"=== 开始调试模式，处理前 {debug_count} 条数据 ===")
    
    # 找到第一个jsonl文件
    data_files = [f for f in os.listdir(data_path) if f.endswith('.jsonl')]
    if not data_files:
        print("没有找到.jsonl文件")
        return
    
    data_file = os.path.join(data_path, data_files[0])
    print(f"使用数据文件: {data_file}")
    
    try:
        dataset = datasets.load_dataset("json", data_files=data_file, streaming=True)
        iter_data = dataset['train']
        
        processed_count = 0
        success_count = 0
        error_count = 0
        tp=0
        fp=0
        fn=0

        print("\n" + "="*80)
        print("开始逐条处理...")
        print("="*80)
        
        for i, item in enumerate(iter_data):
            if processed_count >= debug_count:
                break
                
            processed_count += 1
            print(f"\n--- 第 {processed_count} 条数据 ---")
            
            # 显示原始数据
            print(f"原始文本: {item['sentence'][:100]}..." if len(item['sentence']) > 100 else f"原始文本: {item['sentence']}")
            
            # 显示真实标签
            print("真实标签:")
            y_true = create_y(item['entities'])
            for label_type, entities in y_true.items():
                if entities:
                    print(f"  {label_type}: {entities}")
            
            try:
                # 处理文本（可能包含多行）
                texts = item['sentence'].strip().split('\n')
                texts = [text for text in texts if not is_only_special_chars(text) and len(text.strip()) > 0]
                print(f"分割后文本数量: {len(texts)}")
                
                # 为每个文本创建提示词
                batch_prompts = [create_jsonl_prompt(text) for text in texts]
                
                # 生成结果
                print("生成中...")
                batch_resp = model.generate(batch_prompts, sampling_params)
                
                # 解析结果
                result = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
                raw_outputs = []
                
                for j, resp in enumerate(batch_resp):
                    for r in resp.outputs:
                        raw_output = r.text.strip()
                        raw_outputs.append(raw_output)
                        print(f"  文本{j+1}原始输出: {raw_output}")
                        
                        # 解析JSONL输出
                        try:
                            jsonl_outputs = convert_to_jsonl(raw_output)
                            print(f"  文本{j+1}解析结果: {jsonl_outputs}")
                            
                            for line in jsonl_outputs:
                                if 'label' in line and 'value' in line:
                                    result[line['label']].append(line['value'])
                        except Exception as parse_e:
                            print(f"  文本{j+1}解析失败: {parse_e}")
                
                # 显示预测结果
                print("预测标签:")
                for label_type, entities in result.items():
                    if entities:
                        print(f"  {label_type}: {entities}")
                
                # 计算指标
                metric = calculate_metrics(result, y_true)
                print(f"指标 - TP: {metric['tp']}, FP: {metric['fp']}, FN: {metric['fn']}")

                tp += metric['tp']
                fp += metric['fp']
                fn += metric['fn']
                success_count += 1
                # 清理内存
                del texts, batch_prompts, batch_resp, result
                torch.cuda.empty_cache()
                
            except Exception as e:
                error_count += 1
                print(f"❌ 处理失败: {e}")
                torch.cuda.empty_cache()
                continue
            
            print("-" * 80)
            
            # 每10条显示一次进度
            if processed_count % 10 == 0:
                print(f"\n📊 进度报告: 已处理 {processed_count}/{debug_count}")
                print(f"   成功: {success_count}, 失败: {error_count}")
                print(f"   成功率: {success_count/processed_count*100:.1f}%")
        
        # 最终统计
        print("\n" + "="*80)
        print("🎯 调试完成！最终统计:")
        print("="*80)
        print(f"总处理数量: {processed_count}")
        print(f"成功处理: {success_count}")
        print(f"处理失败: {error_count}")
        print(f"成功率: {success_count/processed_count*100:.1f}%")
        recall= tp / (tp + fn) if (tp + fn) > 0 else 0
        precision= tp / (tp + fp) if (tp + fp) > 0 else 0
        f1= 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"总TP: {tp}, 总FP: {fp}, 总FN: {fn}")
        print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
        
        if error_count > 0:
            print(f"\n⚠️ 有 {error_count} 条数据处理失败，请检查:")
            print("1. 模型输出格式是否符合JSONL要求")
            print("2. 提示词是否需要优化")
            print("3. 数据格式是否正确")
        
    except Exception as e:
        print(f"❌ 调试函数执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    args=parse_args()
    llm=LLM(model=args.model_path,tensor_parallel_size=args.tensor_parallel_size,seed=args.seed,gpu_memory_utilization=args.gpu_memory_utilization,dtype="bfloat16")
    sampling_params = SamplingParams(
    temperature=0,
    max_tokens=512,
    stop=["\n\n\n","###","```","python","<END>","<E>","<B>","Example","Okay"],
    n=1
    )
    data_path=args.data_path
    export_path=args.export_path
    batch_size=args.batch_size
    
    # 添加调试模式支持
    import sys
    if '--debug' in sys.argv:
        print("🔍 启动调试模式...")
        debug(data_path, llm, sampling_params, debug_count=50)
        exit(0)
    
    resume_list=build_resume_list(export_path,data_path)
    if not resume_list:
        print("没有待处理的文件")
        exit(0)
    print(f"待处理文件数: {len(resume_list)}")
        # 检查数据集路径
    for file_index,resume_cnt,metrics in resume_list:
        print(f"开始处理文件: {file_index}.jsonl, 从批次 {resume_cnt} 继续")
        process_batch(file_index,data_path,export_path,llm,sampling_params,
                          batch_size=batch_size,batch_cnt=resume_cnt,metrics=metrics)
    

