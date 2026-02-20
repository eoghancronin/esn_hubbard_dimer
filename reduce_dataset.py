# -*- coding: utf-8 -*-
import json

file_name = 'sec3a/test1_test_data.json'
file_name_2 = 'sec3b/test2_test_data.json'
file_name_3 = 'sec3c/test3_test_data.json'
file_name_4 = 'sec3a/test1_train_data.json'
file_name_5 = 'sec3b/test2_train_data.json'
file_name_6 = 'sec3c/test3_train_data.json'
file_name_7 = 'sec3c/test3_validation_data.json'
file_name_8 = 'appA/test1_train_data.json'
file_name_9 = 'appA/test1_test_data.json'

file_names = [file_name, file_name_2, file_name_3, file_name_4, file_name_5, file_name_6, file_name_7, file_name_8, file_name_9]

def reduce_dataset(file_name):
    with open(file_name,mode='r') as f:
        data = json.load(f)
        
    num_sys=len(data.keys())
    data2 = {}
    for i in range(num_sys):
        data2[f'system_{i}'] = {'density': data[f'system_{i}']['density'],
                                'ext_potential': data[f'system_{i}']['ext_potential']}
    with open(file_name,mode='w') as f:
        json.dump(data2, f)
    
for name in file_names:
    reduce_dataset(name)
        