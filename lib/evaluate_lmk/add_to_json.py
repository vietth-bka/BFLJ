import os
import json
import pickle

eval_source = os.path.join('../../data/CrownHuman/Annotations/instances_val_full_bhf_new.json')
anns = json.load(open(eval_source))
print(anns.keys())
print(anns['annotations'][1])
print(anns['images'][0])
print(anns['categories'][0])

fpath = '../model/rcnn_fpn_baseline/outputs_bfj/eval_dump/dump-30.json'
if 'odgt' in fpath or 'dump' in fpath:
    with open(fpath, "r") as f:
        lines = f.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]

    print('Loading detections..')
    print(records[0]['dtboxes'])
# print(records[0]['gtboxes'])

records = pickle.load(open('./lib/data/CrHu_val4370.pkl','rb'))   
print(records[0])

