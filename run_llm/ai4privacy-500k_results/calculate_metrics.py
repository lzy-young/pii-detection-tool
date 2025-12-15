from multiprocessing import Pool
from argparse import ArgumentParser
import os
import re
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
    parser.add_argument('--data_path', type=str,required=True, help='导入路径')
    parser.add_argument('--export_path', type=str, help='导出路径')
    parser.add_argument('--workers', type=int, default=4, help='工作进程数')
    parser.add_argument('--max_tasks_per_child', type=int, default=0, help='每个子进程最大任务数，0表示无限制')
    return parser.parse_args()


def createY_for_batch(batch_items):
    res={'STREET':[],'TELEPHONENUM':[],'CREDITCARDNUMBER':[],
         'EMAIL':[],'BUILDINGNUM':[],"GIVENNAME":[],'SURNAME':[],'IDCARDNUM':[],'DRIVERLICENSENUM':[],
         'SOCIALNUM':[],'PASSPORTNUM':[],'TAXNUM':[],'TIME':[]}
    for item in batch_items:
        if item['language']!='en':
            continue
        for label in item['privacy_mask']:
            if label['label'] in res.keys():
                res[label['label']].append(label['value'])
    return res


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
    for entity_type in results_sets.keys():
        all_preds.update(results_sets[entity_type])
        all_labels.update(labels_sets[entity_type])
    fn=len(all_labels - all_preds)  # 计算未预测的实际标签数量

    for entity_type in results_sets.keys():
        
        predicted = results_sets[entity_type]
        actual = labels_sets[entity_type]
        # 使用集合运算计算指标
        tp += len(predicted & actual)  # 交集
        fp += len(predicted - actual)  # 预测但不在实际中
        # fn += len(actual - predicted)  # 实际但未预测
        # 计算精确率和召回率
        
    metrics = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
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
        if not file.endswith('_results_new.json'):
            continue
        file_path=os.path.join(results_path,file)
        try:
            with open(file_path,'r',encoding='utf-8') as f:
                result_data=json.load(f)
            is_completed=result_data.get('completed',False)
            batch_cnt=result_data.get('batch_cnt',0)
        except Exception as e:
            batch_cnt=0
        filename=file[:-len('_results_new.json')]
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


def process_batch(file_index,data_path,export_path,details={},batch_size=200,
                  batch_cnt=0,metrics=[0,0,0]):
    if details=={}:
        return
    data_file=os.path.join(data_path,f'{file_index}.jsonl')
    rpath=os.path.join(export_path,f"{file_index}_results_new.json")
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
        res={'STREET':[],'TELEPHONENUM':[],'CREDITCARDNUMBER':[],
         'EMAIL':[],'BUILDINGNUM':[],"GIVENNAME":[],'SURNAME':[],'IDCARDNUM':[],'DRIVERLICENSENUM':[],
         'SOCIALNUM':[],'PASSPORTNUM':[],'TAXNUM':[],'TIME':[]}
        y_hat=createY_for_batch(batch_items)
        try:
            for key in res.keys():
                res[key]=details['batches'][f'batch_{batch_index}'].get(key,[])
            metric=calculate_metrics(res,y_hat)
            tp += metric['tp']
            fp += metric['fp']
            fn += metric['fn']
            del  y_hat,metric
        except Exception as e:
            print(f"该批次处理出错: {e}")
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
            'f1': f1,
        }
        update_results(rpath,res,batch_index,completed=False)
        print(f'文件{file_index}.json - 批次 {batch_index}处理完成({len(batch_items)-error_count}/{len(batch_items)}) -- 已处理 {batch_index*batch_size} 条数据')
    update_results(rpath,None,None,completed=True)
    print(f'文件{file_index}.json 处理完成')



def process(data_path,export_path):    
    resume_list=build_resume_list(export_path,data_path)
    if not resume_list:
        print("没有待处理的文件")
        return
    print(f"待处理文件数: {len(resume_list)}")
        # 检查数据集路径
    for file_index,resume_cnt,metrics in resume_list:
        print(f"开始处理文件: {file_index}.jsonl, 从批次 {resume_cnt} 继续")
        print(f"当前指标 - TP: {metrics[0]}, FP: {metrics[1]}, FN: {metrics[2]}")
        dpath=os.path.join(export_path,f"{file_index}_details.json")
        with open(dpath,'r',encoding='utf-8') as f:
            details=json.load(f)
        process_batch(file_index,data_path,export_path,details=details,batch_cnt=resume_cnt,metrics=metrics)

def run(args):
    dpath,epath=args
    return process(dpath,epath)

if __name__ == "__main__":
    args=parse_args()
    data_path=args.data_path
    export_path=args.export_path
    pool_kwargs={'processes':args.workers}
    if args.max_tasks_per_child and args.max_tasks_per_child>0:
        pool_kwargs['maxtasksperchild']=args.max_tasks_per_child
    task_args=[(data_path,os.path.join(export_path,folder))for folder in os.listdir(export_path) if os.path.isdir(os.path.join(export_path,folder))]
    with Pool(**pool_kwargs) as pool:
        summary=pool.imap_unordered(run, task_args)
        for _ in summary:
            pass
    print("所有文件处理完成")