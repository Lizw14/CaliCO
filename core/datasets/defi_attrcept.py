
from core.datasets.program_translator import gqa_to_nsclseq
from collections import Counter
import json
import csv
import pdb

def build_vocab(train_json):
    train_cept_counters = {}
    train_rel_counters = Counter()
    for i, q in enumerate(train_json.values()):
        for prog in gqa_to_nsclseq(q['semantic']):
            if prog.__contains__('concept'):
                if not prog.__contains__('attr'):
                    print('No attr!')
                    print(prog)
                if prog['attr'] not in train_cept_counters:
                    train_cept_counters[prog['attr']] = Counter()
                rm_not = prog['concept'].strip(')').split('(')[-1].strip()
                train_cept_counters[prog['attr']][rm_not] += 1                        
            if prog.__contains__('attr'):
                if prog['attr'] not in train_cept_counters:
                    train_cept_counters[prog['attr']] = Counter()
            if prog['op'] == 'query':
                if prog.__contains__('choices'):
                    for cept in prog['choices']:
                        train_cept_counters[prog['attr']][cept] += 1
            if prog.__contains__('rel'):
                train_rel_counters[prog['rel']] += 1
        #if i>10:
        #    break
    print('number of attributes: ', len(train_cept_counters))
    cnt = 0
    for k in train_cept_counters:
        print(k, len(train_cept_counters[k]))
        cnt += len(train_cept_counters[k])
    print('number of concepts: ', cnt)
    print(len(train_rel_counters))
    train_cept_dict = {}
    for k in train_cept_counters:
        train_cept_dict[k] = dict(train_cept_counters[k])
    train_rel_dict = {'spatial': {}, 'comparison': {}, 'semantic': {}}
    with open('iep/datasets/gqa_relationships.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[2] == 'spatial':
                train_rel_dict['spatial'][row[0]] = int(row[1])
            elif row[2] == 'comparison':
                train_rel_dict['comparison'][row[0]] = int(row[1])
            elif row[2] in ['semantic', '', 'semantic+spatial']:
                train_rel_dict['semantic'][row[0]] = int(row[1])
            else:
                raise ValueError('unknown relation type: ', row)
    with open('iep/datasets/define_prog_12.json', 'w') as f:
        json.dump({'cepts': train_cept_dict, 'rels': train_rel_dict}, f)


def build_vocab_new(train_jsons):
    train_cept_counters = {}
    train_rel_counters = Counter()
    for train_json in train_jsons:
        for i, q in enumerate(train_json.values()):
            for prog in gqa_to_nsclseq(q['semantic']):
                if prog.__contains__('concept'):
                    if not prog.__contains__('attr'):
                        print('No attr!')
                        print(prog)
                    if prog['attr'] not in train_cept_counters:
                        train_cept_counters[prog['attr']] = Counter()
                    rm_not = prog['concept'].strip(')').split('(')[-1].strip()
                    train_cept_counters[prog['attr']][rm_not] += 1                        
                if prog.__contains__('attr'):
                    if prog['attr'] not in train_cept_counters:
                        train_cept_counters[prog['attr']] = Counter()
                if prog['op'] == 'query':
                    if prog.__contains__('choices'):
                        for cept in prog['choices']:
                            train_cept_counters[prog['attr']][cept] += 1
                if prog.__contains__('rel'):
                    train_rel_counters[prog['rel']] += 1
            #if i>10:
            #    break
    print('number of attributes: ', len(train_cept_counters))
    cnt = 0
    for k in train_cept_counters:
        print(k, len(train_cept_counters[k]))
        cnt += len(train_cept_counters[k])
    print('number of concepts: ', cnt)
    print(len(train_rel_counters))
    train_cept_dict = {}
    for k in train_cept_counters:
        train_cept_dict[k] = dict(train_cept_counters[k])
    with open('iep/datasets/tmp_cept_prog_12.json', 'w') as f:
        json.dump(train_cept_dict, f)

class Gdef():
    def __init__(self, gdef_pth, sgg_vocab_pth):
        super().__init__()
        # op: [[input_type], [arguments], output_type] (_ means optional)
        self.op_def = {
            'scene': ([], [], 'object_set'),
            'select': [['object_set'], ['attribute', 'concept'], 'object_set'],
            'filter': [['object_set'], ['attribute', 'concept'], 'object_set'],
            'relate_o': [['object_set', 'object'], ['rel'], 'object_set'],
            'relate_s': [['object_set', 'object'], ['rel'], 'object_set'],
            'relate_ae': [['object_set', 'object'], ['attribute'], 'object_set'],
            'intersect': [['bool', 'bool'], [], 'bool'],
            'union': [['bool', 'bool'], [], 'bool'],
            #'intersect': [['object_set', 'object_set'], [], 'object_set'], #TODO: object or bool?
            #'union': [['object_set', 'object_set'], [], 'object_set'], #TODO: object or bool?
            'exist': [['object_set'], [], 'bool'],
            'query': [['object'], ['attribute', '_choices'], 'concept'],
            'query_ae': [['object', 'object'], ['attribute'], 'bool'],
            'query_rel_s': [['object', 'object'], ['choices'], 'relation'], 
            'query_rel_o': [['object', 'object'], ['choices'], 'relation'], 
            'verify_rel_s': [['object', 'object'], ['relation'], 'bool'], 
            'verify_rel_o': [['object', 'object'], ['relation'], 'bool'], 
            'verify': [['object'], ['attribute', 'concept'], 'bool'],
            'choose': [['object', 'object'], ['attribute', 'concept', 'choices'], 'concept'],
            'same': [['object_set'], ['attribute'], 'bool'],
            'common': [['object', 'object'], [], 'attribute'], 
            'negate': [['bool'], [], 'bool'],
            'unique': [['object_set'], [], 'object'],
        }
        self.gdef_pth = gdef_pth
        self.sgg_vocab_pth = sgg_vocab_pth
        
        sgg_vocab = json.load(open(self.sgg_vocab_pth, 'r'))
        self.sgg_vocab = {'label': sorted(sgg_vocab['label2idx'], key=lambda k: sgg_vocab['label2idx'][k]), 
            'attr': sorted(sgg_vocab['attr2idx'], key=lambda k: sgg_vocab['attr2idx'][k]),
             'rel': sorted(sgg_vocab['rel2idx'], key=lambda k: sgg_vocab['rel2idx'][k])}

        self.taxonomies = self.get_taxnomies()
        self.cept2attrs = self.map_cept2attrs()
        self.attributes_asked_in_common = ['color', 'material', 'shape']

    def qtype2atype(self, qtype):
        return self.op_def[qtype][2]

    def qtype2inputtype(self, qtype):
        return self.op_def[qtype][0]

    def rel_cept2attr(self, cept):
        return self.cept2attrs[1].get(cept, 'unk')

    def attr_cept2attr(self, cept):
        return self.cept2attrs[0].get(cept, 'unk')

    def get_taxnomies(self, unk_threshold=0):
        # return [attr_taxnomy, rel_taxnomy]
        cept_def = json.load(open(self.gdef_pth, 'r'))
        res = [dict(), dict()]
        # manully make attribute list
        for attr in ['void', 'color', 'name', 'height', 'size', 'material', 'vposition', 'hposition', 'pattern', 'activity', 'pose', 
        'place', 'weather', 'shape', 'tone', 'sport', 'location', 'company', 'state', 'age', 'cleanliness', 'length', 'width', 'gender', 
        'face', 'fatness', 'thickness', 'hardness', 'weight', 'depth', 'flavor', 'liquid', 'realism', 'room', 'race', 'opaqness', 'event', 
        'orientation', 'texture', 'brightness', 'face expression']:
            res[0][attr] = []
        for attr in ['spatial', 'comparison', 'semantic']:
            res[1][attr] = []
        for idx, tag in enumerate(['cepts', 'rels']):
            for attr_ in cept_def[tag]:
                attr = attr_
                # type and sportActivity can be merged into sport
                if attr_ == 'type' or attr_ == 'sportActivity':
                    attr = 'sport'
                elif attr_ == '':
                    attr = 'void'
                for cept, cnt in cept_def[tag][attr_].items():
                    try:
                        if cnt > unk_threshold:
                            res[idx][attr].append(cept)
                            if attr not in res[idx]:
                                raise NotImplementedError()
                    except:
                        pdb.set_trace()
        return res

    # def get_taxnomies(self, unk_threshold=0):
    #     tax = json.load(open(self.gdef_pth, 'r'))
    #     return [tax['attribute'], tax['relation']]

    def map_cept2attrs(self):
        res = [dict(), dict()]
        for idx, taxonomy in enumerate(self.taxonomies):
            for attr in taxonomy:
                for cept in taxonomy[attr]:
                    res[idx][cept] = attr
        return res



def load_gdef(gdef_pth='iep/datasets/define_prog_12.json', sgg_vocab_pth='/home/zhuowan/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/datasets/GQA/gqa_vocab_thres1.json'):
    gdef = Gdef(gdef_pth, sgg_vocab_pth)
    return gdef


if __name__ == "__main__":
    # for building defi_attrcept.json
    train_json = json.load(open('data/orig_data/questions1.2/train_balanced_questions.json', 'r'))
    val_json = json.load(open('data/orig_data/questions1.2/val_balanced_questions.json', 'r'))
    #build_vocab(train_json)
    build_vocab_new([train_json, val_json])

    # ver12: 40 attr, 2944 cept, 322 rel

    # test load_gdeg
    gdef = load_gdef()
    pdb.set_trace()
    print('finished')
