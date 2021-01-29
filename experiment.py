# encoding:utf-8
import json

with open('cfg.json',encoding='utf-8') as f:
    json_file = json.load(f)
    print((json_file['feature_size'][0])['feature_maps'])
