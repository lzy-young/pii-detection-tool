import json
import datasets
import os
import pandas as pd

def extract_addresses_from_file(dataset_name,file_name, dataset_info):
    path=os.path.join(dataset_info['path'],f'{file_name}.jsonl')
    data=datasets.load_dataset('json', data_files=path,streaming=True)['train']
    res={'dataset':f'{dataset_name}/{file_name}','total_addresses':0}
    label_list=['STATE','SECADDRESS','LOC','CITY','COUNTRY','STREET','BUILDING','GPE','ADDRESS','GEOCOORD']
    for i,item in enumerate(data):
        item_record={'id':i,'sentence':'', 'addresses':[],'count':0}
        entities=item[dataset_info['entities_key']]
        count=0
        for entity in entities:
            label=entity[dataset_info['label_key']]
            if label in label_list:
                item_record['addresses'].append(entity[dataset_info['value_key']])
                count+=1
        if count>0:
            item_record['count']=count
            item_record['sentence']=item[dataset_info['sentence_key']]
            res[f'sentence_{i}']=item_record
            res['total_addresses']+=count
    with open(f'./{file_name}_addresses.json','w') as f:
        json.dump(res,f,indent=4)

def process(dataset_name,dataset_info):
    for file in os.listdir(dataset_info['path']):
        if file.endswith('jsonl'):
            filename=file[:-len('.jsonl')]
            print(f'正在处理文件: {file}')
            extract_addresses_from_file(dataset_name, filename, dataset_info)


if __name__ == "__main__":
    dataset_list={  
    'ai4privacy': {'path':'../ai4privacy/english_data','entities_key':'privacy_mask','sentence_key':'source_text','label_key':'label','value_key':'value'},
    'conll2003': {'path':'../conll2003','entities_key':'entities','sentence_key':'sentence','label_key':'label','value_key':'text'},
    'ontonotes5': {'path':'../ontonotes5/dataset/cleaned','entities_key':'entities','sentence_key':'sentence','label_key':'label','value_key':'text'},
    'SPY_Dataset': {'path':'../SPY_Dataset/output','entities_key':'entities','sentence_key':'sentence','label_key':'label','value_key':'text'}
}
    for dataset_name,dataset_info in dataset_list.items():
        print(f'正在处理数据集: {dataset_name}')
        process(dataset_name,dataset_info)
        print(f'完成数据集: {dataset_name}')
    print('所有数据集处理完成！')
    