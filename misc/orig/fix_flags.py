import json
from utils import replace_entities
import pickle
from utils import extract_entities

def read_json(path):
    with open(path, "r") as file:
        lines = file.readlines()
        lines = list(map(lambda x: x.strip('\n'), lines))
    return lines

def write_json(output_json, path):
    json_object = json.dumps(output_json, indent=4)
    with open(path, "w") as outfile:
        outfile.write(json_object)

ASSET_FILES = {'name':'asset',
               'train_input':'data/raw/asset/asset.valid.orig',
               'train_labels':['data/raw/asset/asset.valid.simp.0',
                                'data/raw/asset/asset.valid.simp.1',
                                'data/raw/asset/asset.valid.simp.2',
                                'data/raw/asset/asset.valid.simp.3',
                                'data/raw/asset/asset.valid.simp.4',
                                'data/raw/asset/asset.valid.simp.5',
                                'data/raw/asset/asset.valid.simp.6',
                                'data/raw/asset/asset.valid.simp.7',
                                'data/raw/asset/asset.valid.simp.8',
                                'data/raw/asset/asset.valid.simp.9'],
               'test_input':'data/raw/asset/asset.test.orig',
               'test_labels':['data/raw/asset/asset.test.simp.0',
                                'data/raw/asset/asset.test.simp.1',
                                'data/raw/asset/asset.test.simp.2',
                                'data/raw/asset/asset.test.simp.3',
                                'data/raw/asset/asset.test.simp.4',
                                'data/raw/asset/asset.test.simp.5',
                                'data/raw/asset/asset.test.simp.6',
                                'data/raw/asset/asset.test.simp.7',
                                'data/raw/asset/asset.test.simp.8',
                                'data/raw/asset/asset.test.simp.9']}
COCH_FILES = {'name':'cochrane',
               'train_input':'data/raw/cochrane/train.source',
               'train_labels':['data/raw/cochrane/train.target'],
               'test_input':'data/raw/cochrane/test.source',
               'test_labels':['data/raw/cochrane/test.target']}
RADR_FILES = {'name':'radiology',
               'train_input':'data/raw/radiology/chest.source',
               'train_labels':['data/raw/radiology/chest.target'],
               'test_input':'data/raw/radiology/chest.source',
               'test_labels':['data/raw/radiology/chest.target']}
TURK_FILES = {'name':'turkcorpus',
               'train_input':'data/raw/turkcorpus/tune.8turkers.tok.norm',
               'train_labels':['data/raw/turkcorpus/tune.8turkers.tok.turk.0',
                                'data/raw/turkcorpus/tune.8turkers.tok.turk.1',
                                'data/raw/turkcorpus/tune.8turkers.tok.turk.2',
                                'data/raw/turkcorpus/tune.8turkers.tok.turk.3',
                                'data/raw/turkcorpus/tune.8turkers.tok.turk.4',
                                'data/raw/turkcorpus/tune.8turkers.tok.turk.5',
                                'data/raw/turkcorpus/tune.8turkers.tok.turk.6',
                                'data/raw/turkcorpus/tune.8turkers.tok.turk.7'],
               'test_input':'data/raw/turkcorpus/test.8turkers.tok.norm',
               'test_labels':['data/raw/turkcorpus/test.8turkers.tok.turk.0',
                                'data/raw/turkcorpus/test.8turkers.tok.turk.1',
                                'data/raw/turkcorpus/test.8turkers.tok.turk.2',
                                'data/raw/turkcorpus/test.8turkers.tok.turk.3',
                                'data/raw/turkcorpus/test.8turkers.tok.turk.4',
                                'data/raw/turkcorpus/test.8turkers.tok.turk.5',
                                'data/raw/turkcorpus/test.8turkers.tok.turk.6',
                                'data/raw/turkcorpus/test.8turkers.tok.turk.7']}

input_dict = ASSET_FILES
bio_flag = False

with open("train_asset_lst.pkl", "rb") as input_file:
    train_wiki_input_lst = pickle.load(input_file)
with open("test_asset_lst.pkl", "rb") as input_file:
    test_wiki_input_lst  = pickle.load(input_file)


new_train_lst = []
for i, (item, orig) in enumerate(zip(train_wiki_input_lst, train_input_lst)):
    # break
    if i%100==0: print(i)
    entities   = extract_entities(orig, False)
    total_ents = len(entities)
    not_found_ents = item[0].count('()')
    found_ents = total_ents - not_found_ents
    new_flag = 'none' if (total_ents == 0 or found_ents == 0) else 'all' if total_ents==found_ents else 'some'
    new_str  = item[0].replace('()','').replace('  ',' ')
    new_train_lst.append((new_str,new_flag))
    
new_test_lst = []
for i, (item, orig) in enumerate(zip(test_wiki_input_lst, test_input_lst)):
    # break
    if i%100==0: print(i)
    entities   = extract_entities(orig, False)
    total_ents = len(entities)
    not_found_ents = item[0].count('()')
    found_ents = total_ents - not_found_ents
    new_flag = 'none' if (total_ents == 0 or found_ents == 0) else 'all' if total_ents==found_ents else 'some'
    new_str  = item[0].replace('()','').replace('  ',' ')
    new_test_lst.append((new_str,new_flag))
    
with open('train_asset_lst.pkl', 'wb') as f:
    pickle.dump(new_train_lst, f)
with open('test_asset_lst.pkl', 'wb') as f:
    pickle.dump(new_test_lst, f)