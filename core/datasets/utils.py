import json
import h5py
import os


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_h5(filename):
    return h5py.File(filename, 'r')


def make_partial_dataset():
    '''
    10398 images all together
    filter 10k training dataset out of 943000
    filter testdev images ( 12578 questions)
    (val: 132062)
    to experiment with different features
    148854 images
    '''
    img_list = []
    num_train = 10000
    for split, num_imgs in zip(['train', 'testdev'], [num_train, 1944]):
        questions_json = '/data/c/zhuowan/gqa/data/questions1.3/' + split + '_balanced_questions.json'
        questions = load_json(questions_json)
        question_ids = sorted(questions.keys())
        img_list_split = set()
        for index in range(len(questions)):
            qid = question_ids[index]
            img_id = questions[qid]['imageId']
            img_list_split.add('/data/c/zhuowan/gqa/data/images/'+str(img_id)+'.jpg')
            if split!='testdev' and len(img_list_split) >= num_imgs:
                break
        img_list.extend(list(img_list_split))
        print(len(img_list)) #10000, 10398
    with open('/data/c/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/datasets/GQA/train10k_testdev.json', 'w') as f:
        json.dump(img_list, f)


def filter_part_dataset():
    # train: 130862/943000, testdev: 12578/12578
    img_list = load_json('/data/c/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/datasets/GQA/train10k_testdev.json')
    img_set = set(img_list)
    questions_json = '/data/c/zhuowan/gqa/data/questions1.3/train_balanced_questions.json'
    questions = load_json(questions_json)
    question_ids = sorted(questions.keys())
    train_indices = []
    for index in range(len(questions)):
        qid = question_ids[index]
        img_id = questions[qid]['imageId']
        if '/data/c/zhuowan/gqa/data/images/'+str(img_id)+'.jpg' in img_set:
            train_indices.append(index)
            
    cache_path = '/home/zhuowan/zhuowan/gqa_project/gqa_cleaned/data/cache/'
    print('train num: ', len(train_indices))
    dump_json(cache_path + 'train-train10k.json', {'indices': train_indices, 'filt_name': 'train10k'})
    dump_json(cache_path + 'testdev-train10k.json', {'indices': list(range(12578)), 'filt_name': 'train10k'})

    
    def dump_cache(self, cache_path):
        filt_state_dict = {'indices': self.indices, 'filt_name': self._filt_name}
        dump_json(cache_path + self._filt_name+'.json', filt_state_dict)

    def load_cache(self, cache_path, filt_name):
        filt_state_dict = load_json(cache_path + filt_name+'.json')
        self.indices = filt_state_dict['indices']
        self._filt_name = filt_state_dict['filt_name']

if __name__ == "__main__":
    #make_partial_dataset()
    #filter_part_dataset()
    pass
