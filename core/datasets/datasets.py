import numpy as np
import pdb
import os.path as osp
import copy
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from jactorch.data.collate import VarLengthCollateV2
from core.datasets.program_translator import gqa_to_nsclseq
from core.datasets.utils import *
from core.datasets.preprocess import *
from core.utils import load_vocab

from tqdm import tqdm

class FilterableDatasetUnwrapped(Dataset):
    
    def __init__(self):
        super().__init__()


class FilterableDatasetView(FilterableDatasetUnwrapped):

    def __init__(self, dataset, indices=None, filt_name=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else range(len(dataset))
        self._filt_name = filt_name if filt_name is not None else 'origin'
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def filter(self, filt_func, filt_name='<anonymous>'):
        indices = list()
        for index in tqdm(self.indices):
            metainfo = self.dataset[index]
            if filt_func(metainfo):
                indices.append(index)
        if len(indices) == 0:
            raise ValueError('Filter results in an empty dataset.')
        self.indices = indices
        self._filt_name += ':{}'.format(filt_name)

    def split_trainval(self, split):
        assert(split < len(self))
        return (
            type(self)(self.dataset, indices=self.indices[:split], filt_name='train'),
            type(self)(self.dataset, indices=self.indices[split:], filt_name='validation')
        )

    @property
    def unwrapped(self):
        return self.dataset

    @property
    def filt_name(self):
        return self._filt_name

    def dump_cache(self, cache_path):
        filt_state_dict = {'indices': self.indices, 'filt_name': self._filt_name}
        dump_json(cache_path + self._filt_name+'.json', filt_state_dict)

    def load_cache(self, cache_path, filt_name):
        cache_name = cache_path + filt_name+'.json'
        if os.path.exists(cache_name):
            filt_state_dict = load_json(cache_name)
            self.indices = filt_state_dict['indices']
            self._filt_name = filt_state_dict['filt_name']
        else:
            print('Filtering cache: '+cache_name+ ' does not exists')


class NSCLDatasetUnwrapped(FilterableDatasetUnwrapped):

    def __init__(self, questions_json, info_json, objects_h5, scenegraphs_json, vocab_json, is_gtencode, data_num=None):
        self.questions_json = questions_json
        self.info_json = info_json
        self.objects_h5 = objects_h5
        self.vocab_json = vocab_json 
        self.is_gtencode = is_gtencode

        self.questions = {}
        for qf in questions_json.split(','):
            print('loading from json file: ', qf)
            self.questions.update(load_json(qf.strip()))
        if data_num is not None:
            if data_num < len(self.questions):
                selected_ids = list(self.questions.keys())[:data_num]
                self.questions = {idx:self.questions[idx] for idx in selected_ids}
        self.info = load_json(info_json) #a['2370799']: {'width': 500, 'objectsNum': 24, 'height': 333, 'index': 94550}
        self.vocab = load_vocab(vocab_json)
        # TODO
        self.scenegraphs = load_json(scenegraphs_json)
        self.sg_all_concepts = load_json('data/sg_all_concepts.json')
        self.concept2id = {c: i for i, c in enumerate(self.sg_all_concepts)}

        # key of question_json is question_id
        # key of info_json is image_id
        #self.question_ids = sorted(self.questions.keys())
        self.question_ids = list(self.questions.keys())
        print('Number of questions and scenegraphs: ', len(self.questions), len(self.scenegraphs))

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):        
        qid = self.question_ids[index]
        feeddict = self.gqa2meta(self.questions[qid], self.info, self.objects_h5)
        
        # add fields to feeddict
        feeddict['qid'] = qid
        feeddict['objects'], feeddict['boxes'] = self.annotate_objects(feeddict)
        feeddict['question_token'] = tokenize(feeddict['question_raw'], punct_to_keep=[';', ','], punct_to_remove=['?', '.'], is_stem=False)
        feeddict['question'] = np.array(encode(feeddict['question_token'], self.vocab['question_token_to_idx'], allow_unk=True, verbose_unk=False), dtype=np.int64)
        if feeddict['program_seq'][-1]['op'] != 'negate':
            feeddict['question_type'] = feeddict['program_seq'][-1]['op']
        else:
            feeddict['question_type'] = feeddict['program_seq'][-2]['op']
        feeddict['answer'] = np.array(encode(['_'.join(feeddict['answer_raw'].split())], self.vocab['answer_token_to_idx'], allow_unk=True, verbose_unk=False), dtype=np.int64)
        
        return feeddict

    def gqa2meta(self, q_raw, info_json, objects_h5):
        meta = dict()
        meta['imageId'] = q_raw['imageId']

        #if meta['imageId'] in info_json:
        info_raw = info_json[str(meta['imageId'])]
        meta['height'] = info_raw['height']
        meta['width'] = info_raw['width']
        # else:
        #     print(meta['imageId'])
        #     pdb.set_trace()
        #     pth = '/data/c/zhuowan/gqa/data/images/'+str(meta['imageId'])+'.jpg'
        #     img = Image.open(pth)
        #     meta['width'], meta['height'] = img.size

        meta['program_raw'] = q_raw['semantic']
        try:
            meta['program_seq'] = gqa_to_nsclseq(q_raw['semantic'])
        except Exception as e:
            print('dataset exception: ', e)
            meta['program_seq'] = [dict(op='select', attr='', concept='person', inputs=['_']), 
                dict(op='exist', inputs=[0])]
        meta['question_raw'] = q_raw['question']
        meta['answer_raw'] = q_raw['answer']

        if self.is_gtencode:
            sg = self.scenegraphs[q_raw['imageId']]
            objects = []
            boxes = []
            for v in sg['objects'].values():
                obj = np.zeros(len(self.sg_all_concepts), dtype=np.float32)
                box = np.zeros(4, dtype=np.float32)
                name = v['name']
                cid = self.concept2id.get(name, -1)
                if cid == -1:
                    print('name', name)
                else:
                    obj[cid] = 1
                for attr in v['attributes']:
                    cid = self.concept2id.get(attr, -1)
                    if cid == -1:
                        print('concept', attr)
                    else:
                        obj[cid] = 1
                objects.append(obj)
                box[0] = v['x']
                box[1] = v['y']
                box[2] = v['x'] + v['w']
                box[3] = v['y'] + v['h']
                boxes.append(box)
            if len(objects)==0:
                objects.append(np.zeros(len(self.sg_all_concepts), dtype=np.float32))
                boxes.append(np.zeros(4, dtype=np.float32))
                print('WARNING: no objects for this example!')
            objects = np.stack(objects, axis=0)
            boxes = np.stack(boxes, axis=0)
            meta['objects_raw'] = objects
            meta['boxes_raw'] = boxes
            meta['num_objects'] = objects.shape[0]
        else:
            meta['num_objects'] = info_raw['objectsNum']
            meta['object_idx'] = info_raw['index']
            with load_h5(objects_h5) as h5_raw:
                features = h5_raw['features'][meta['object_idx']][:meta['num_objects']].astype('float32')
                if 'attributes' in h5_raw:
                    attributes = h5_raw['attributes'][meta['object_idx']][:meta['num_objects']].astype('float32')
                    meta['attr_logits'] = attributes
                if 'all_scores' in h5_raw:
                    all_scores = h5_raw['all_scores'][meta['object_idx']][:meta['num_objects']].astype('float32')
                    meta['label_scores'] = all_scores
                #meta['objects_raw'] = np.concatenate((features, all_scores), axis=-1)
                meta['objects_raw'] = features
                meta['boxes_raw'] = h5_raw['bboxes'][meta['object_idx']][:meta['num_objects']]
        
        return meta

    def annotate_objects(self, feeddict):
        h, w = feeddict['height'], feeddict['width']
        # boxes_raw: [x1, y1, x2, y2]
        #ratio = min(1 / min(h, w) * 600., 
        #            1 / max(h, w) * 1000.)
        #boxes = copy.deepcopy(feeddict['boxes_raw']) * ratio
        # boxes = copy.deepcopy(feeddict['boxes_raw'])
        # boxes[:, 0] = 1.0 / w
        # boxes[:, 1] *= 1.0 / h
        # boxes[:, 2] *= 1.0 / w 
        # boxes[:, 3] *= 1.0 / h 
        shape = feeddict['boxes_raw'].shape
        boxes = np.zeros((shape[0], shape[1]+1), dtype=feeddict['boxes_raw'].dtype)
        boxes[:, 0] = feeddict['boxes_raw'][:, 0] / w
        boxes[:, 1] = feeddict['boxes_raw'][:, 1] / h
        boxes[:, 2] = feeddict['boxes_raw'][:, 2] / w - boxes[:, 0]
        boxes[:, 3] = feeddict['boxes_raw'][:, 3] / h - boxes[:, 1]
        boxes[:, 4] = boxes[:, 2] * boxes[:, 3]
        return feeddict['objects_raw'], boxes


class NSCLDatasetFilterableView(FilterableDatasetView):
    
    def filter_program_size(self, max_length):
        def filt(metainfo):
            return len(metainfo['program_seq']) <= max_length
        self.filter(filt, 'program-size-[{}]'.format(max_length))

    def filter_scene_size(self, max_scene_size):
        def filt(metainfo):
            return metainfo['objects'].shape[0] <= max_scene_size
        self.filter(filt, 'scene-size-[{}]'.format(max_scene_size))

    def filter_question_type(self, *, allowed=None, disallowed=None):
        def filt(metainfo):
            if allowed is not None:
                return metainfo['question_type'] in allowed
            if disallowed is not None:
                return metainfo['question_type'] not in disallowed
        if allowed is not None:
            self.filter(filt, 'question-type-[allowed={{{}}}]'.format(','.join(allowed)))
        elif disallowed is not None:
            self.filter(filt, 'question-type-[disallowed={{{}}}]'.format(','.join(disallowed)))
        else:
            raise ValueError('Either allowed or disallowed must be provided.')

    def filter_op_type(self, *, allowed=None, disallowed=None):
        def filt(metainfo):
            if allowed is not None:
                for block in metainfo['program_seq']:
                    if block['op'] not in allowed:
                        return False
                return True
            if disallowed is not None:
                for block in metainfo['program_seq']:
                    if block['op'] in disallowed:
                        return False
                return True
        if allowed is not None:
            self.filter(filt, 'op-type-[allowed={{{}}}]'.format(','.join(allowed)))
        elif disallowed is not None:
            self.filter(filt, 'op-type-[disallowed={{{}}}]'.format(','.join(disallowed)))
        else:
            raise ValueError('Either allowed or disallowed must be provided.')
        

def NSCLDataset(*args, **kwargs):
    return NSCLDatasetFilterableView(NSCLDatasetUnwrapped(*args, **kwargs))

def build_gqa_dataset(args):
    dataset = NSCLDataset(args.question_json_path, args.info_json_path, args.objects_h5_path, args.scenegraphs_json_path, args.vocab_json_path, args.is_gtencode, args.data_num)
    print('original dataset built')
    #dataset.filter_op_type(allowed=['select', 'filter', 'query'])
    #dataset.dump_cache('data/cache/'+args.mode+'-ver12-')
    if args.dataset_cache is not None:
        print('loading dataset cache: ', args.dataset_cache)
        dataset.load_cache('data/cache/'+args.mode+'-', args.dataset_cache)
        print('dataset_filtered', args.mode, len(dataset))
    return dataset

def build_gqa_dataloader(dataset, batch_size, shuffle, drop_last, num_workers):
    # mode: skip, concat, pad, pad2d, padimage
    collate_guide = {
        'imageId': 'skip',
        'height': 'skip',
        'width': 'skip',
        'program_seq': 'skip',
        'program_raw': 'skip',
        'question_raw': 'skip',
        'answer_raw': 'skip',
        'num_objects': 'skip',
        'object_idx': 'skip',
        'objects_raw': 'skip',
        'boxes_raw': 'skip',
        'question': 'pad',
        'qid': 'skip',
        'objects': 'concat',
        'boxes': 'concat',
        'question_type': 'skip',
        'answer': 'concat',
        'attr_logits': 'concat',
        'label_scores': 'concat'
    }
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, 
        pin_memory=True, collate_fn=VarLengthCollateV2(collate_guide))
