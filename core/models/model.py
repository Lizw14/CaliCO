import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import random
import os.path as osp
import pdb

from core.models.symbolic import DifferentiableReasoning
from core.utils import load_vocab

def my_print(*inp):
    #return print(inp)
    pass

class QALoss(nn.Module):
    def __init__(self, gdef):
        super().__init__()
        self.gdef = gdef
        self.vocab = load_vocab('data/gqa_vocab_13.json')
        post_process = {"to the right of": 'right', 'to the left of': 'left', 'in front of': 'front'} #'wood': 'wooden', 'metal': 'metallic'
        for k in post_process:
            self.vocab['answer_token_to_idx'][k.replace(' ','_')] = self.vocab['answer_token_to_idx'][post_process[k]]

    def forward(self, feed_dict, answers):
        """
        Args:
            feed_dict (dict): input feed dict.
            answers (list): answer derived from the reasoning module.
            return outputs: a dict containing list of {answer, loss, is_correct}
        """
        outputs = {'answer': [], 'loss':[], 'is_correct': [], 'loss_type': {'op': {k:[] for k in self.gdef.op_def.keys()}, 'losstype':{'bool':[], 'softmax':[]}}}
        answers, lstm_logits, mixing = answers

        for i, (query_type, answer) in enumerate(answers):

            gt_raw = feed_dict['answer_raw'][i]
            # fields: imageId, height, width, program_seq, question_raw, answer_raw, num_objects, object_idx, objects_raw, boxes_raw, 
            # additional fields: objects, boxes, question_token, question, question_type, answer 
            response_query_type = self.gdef.qtype2atype(query_type)

            try:
                if response_query_type == 'bool':
                    answer = answer[0]
                    argmax = int((answer > 0).item())
                    outputs['answer'].append(['no', 'yes'][argmax])
                    gt = {'yes': 1.0, 'no': 0.0}[gt_raw] #gt need to be float type
                    loss = self._bce_loss
                elif response_query_type in ['concept', 'relation', 'attribute']:
                    response_query_type = 'softmax'
                    answer, idx2word = answer[0], answer[-1]

                    post_process = {"to the right of": 'right', 'to the left of': 'left', 'in front of': 'front'} #'wood': 'wooden', 'metal': 'metallic'

                    argmax = answer.argmax(dim=-1).item()
                    outputs['answer'].append(idx2word[argmax])
                    word2idx = {v:k for k, v in idx2word.items()}
                    if outputs['answer'][-1] in post_process:
                        outputs['answer'][-1] = post_process[outputs['answer'][-1]]
                    if gt_raw=='wooden':
                        gt = word2idx['wood']
                    elif gt_raw=='metallic':
                        gt = word2idx['metal']
                    elif gt_raw in word2idx:
                        gt = word2idx[gt_raw]
                    #TODO: answer not in choices
                    else:
                        gt = None
                        for word in word2idx:
                            if gt_raw in word:
                                gt = word2idx[word]
                    
                    if gt is None:
                        my_print('answer not in choice, skipping ', feed_dict['qid'][i])
                        my_print(gt_raw, outputs['answer'][-1], word2idx.keys(), feed_dict['question_raw'][i])
                    else:
                        if answer.ndim==0:
                            answer = answer.unsqueeze(0)
                    loss = self._ce_loss
                else:
                    gt = None
                    print('Unknown query type: {}.'.format(query_type), feed_dict['qid'][i])
                    if len(outputs['answer']) < i+1:
                        outputs['answer'].append('yes')
                    answer = answer[0]

                outputs['is_correct'].append(int(outputs['answer'][-1] == gt_raw))

                if self.training and gt is not None:
                    l = loss(answer.unsqueeze(0), gt)
                    outputs['loss'].append(l)
                    l_detached = l.detach().item()
                    outputs['loss_type']['op'][query_type].append(l_detached)
                    outputs['loss_type']['losstype'][response_query_type].append(l_detached)
                    if torch.isnan(l):
                        pdb.set_trace()
            except Exception as e:
                print(e)
                print(feed_dict['qid'][i])
                print(i, len(outputs['is_correct']))
                if len(outputs['is_correct']) < i+1:
                    outputs['is_correct'].append(0)
                if len(outputs['answer']) < i+1:
                    outputs['answer'].append('yes')
                print(i, len(outputs['is_correct']))
                assert(len(outputs['is_correct'])==i+1)
                assert(len(outputs['answer'])==i+1)
                print('error here')
        return outputs

    def _bce_loss(self, pred, label):
        _label = torch.tensor(label, device=pred.device).unsqueeze(0)
        bce = F.binary_cross_entropy_with_logits(input = pred, target=_label)
        # label smoothing
        # smooth_loss = F.binary_cross_entropy_with_logits(input = pred, target=1-_label)
        # eps = 0.1
        # loss = (1-eps)*bce + eps*smooth_loss
        return bce

    def _ce_loss(self, pred, label):
        _label = torch.tensor(label, device=pred.device).unsqueeze(0)
        cross_entropy = F.cross_entropy(input=pred, target=_label)
        # # label smoothing
        # logsoftmax = F.log_softmax(pred, dim=-1)
        # n = pred.size()[-1]
        # smooth_loss = -logsoftmax.sum(dim=-1)
        # eps = 0.1
        # loss = (1-eps)*cross_entropy + eps*smooth_loss/n
        return cross_entropy

    def _bce_classify_loss(self, pred, label):
        _label = torch.zeros(pred.shape, device=pred.device)
        _label[0, label] = 1.0
        return F.binary_cross_entropy_with_logits(input = pred, target=_label)



class GQA_model(nn.Module):
    def __init__(self, model_configs, gdef):
        super().__init__()
        self.qa_loss = QALoss(gdef)
        self.reasoning = DifferentiableReasoning(input_dims=model_configs.input_dims, 
            hidden_dims = model_configs.hidden_dims, 
            gdef = gdef)
        self.hier_superviser = None
        if model_configs.if_hier:
            self.hier_superviser = Hier_superviser(attribute_taxonomy=self.reasoning.sym_attribute_taxonomy,
                data_hypernym_json_pth='data/vcml/gqa_hypernym.json',
                data_instance_json_pth='data/vcml/gqa_isinstanceof.json')

    def forward(self, feed_dict):
        programs, buffers, answers = self.reasoning(feed_dict) 
        assert(len(answers[0])==len(feed_dict['qid']))
        raw_loss = self.qa_loss(feed_dict, answers) # a dict containing list of {answer, loss, is_correct}
        loss = sum(raw_loss['loss'])/len(programs)
        if self.hier_superviser is not None:
            hier_loss, is_corrects = self.hier_superviser(num_sample = 25)
            loss = loss + hier_loss
        if self.training:
            for logname in raw_loss['loss_type']:
                for k in raw_loss['loss_type'][logname]:
                    num = len(raw_loss['loss_type'][logname][k])
                    if num > 0:
                        raw_loss['loss_type'][logname][k] = sum(raw_loss['loss_type'][logname][k]) / num
                    else:
                        raw_loss['loss_type'][logname][k] = None
            if self.hier_superviser is not None:
                raw_loss['loss_type']['hier'] = {'basic': loss.detach().item(), 'hier': hier_loss.detach().item()}
                raw_loss['is_correct_hier'] = is_corrects
        return loss, [raw_loss, buffers]

    def save_ckpt(self, checkpoint, ckpt_pth):
        torch.save(checkpoint, osp.join(ckpt_pth, 'ckpt.pt'))
        if checkpoint['best_val_iter']==checkpoint['iters']:
            torch.save(checkpoint, osp.join(ckpt_pth, 'best_ckpt.pt'))

    def load_ckpt(self, ckpt_pth, strict=True, is_best=True):
        print('loading checkpint from '+ ckpt_pth)
        if is_best:
            checkpoint = torch.load(osp.join(ckpt_pth, 'best_ckpt.pt'))
        else:
            checkpoint = torch.load(osp.join(ckpt_pth, 'ckpt.pt'))
        self.load_state_dict(checkpoint['model'], strict=strict)


def build_model(model_configs, gdef, ckpt_pth=None):
    model = GQA_model(model_configs, gdef)
    if ckpt_pth is not None:
        model.load_ckpt(ckpt_pth, strict=False)
    return model


class Hier_superviser(nn.Module):
    def __init__(self, attribute_taxonomy, data_hypernym_json_pth, data_instance_json_pth):
        super().__init__()
        self.attribute_taxonomy = attribute_taxonomy
        self.data_hypernym = json.load(open(data_hypernym_json_pth, 'r'))
        self.data_hypernym_keys = list(self.data_hypernym.keys())
        self.data_isinstanceof = json.load(open(data_instance_json_pth, 'r'))
        self.data_isinstanceof_keys = list(self.data_isinstanceof.keys())
    
    def meteconcept(self, concept1, concept2, operation):
        metaconcept_index = {
            'synonym': 0, 'hypernym': 2, 'samekind': 3, 'meronym': 4,
        }[operation]
        concept_embedding1 = self.attribute_taxonomy.get_concept_embedding(concept1).unsqueeze(0)
        concept_embedding2 = self.attribute_taxonomy.get_concept_embedding(concept2).unsqueeze(0)
        score = self.attribute_taxonomy.judge_relation(concept_embedding1, concept_embedding2).squeeze()[metaconcept_index]
        return score

    def forward(self, num_sample, pos_rate=0.5):
        scores, gts = [], []
        for _ in range(num_sample):
            is_pos = True if random.uniform(0,1)>0.5 else False
            hyper_or_instance = True if random.uniform(0,1)>1 else False
            if hyper_or_instance:
                concept1 = random.choice(self.data_hypernym_keys)
                concept1for2 = concept1 if is_pos else random.choice(self.data_hypernym_keys)
                concept2 = random.choice(self.data_hypernym[concept1for2])
                operation = 'hypernym'
            else:
                concept_class = random.choice(self.data_isinstanceof_keys)
                if is_pos:
                    concept1, concept2 = random.sample(self.data_isinstanceof[concept_class], 2)
                else:
                    concept1 = random.choice(self.data_isinstanceof[concept_class])
                    concept_class2 = random.choice(self.data_isinstanceof_keys)
                    concept2 = random.choice(self.data_isinstanceof[concept_class2])
                operation = 'samekind'
            #print(','.join([concept1, concept2, operation, str(is_pos)]))
            score = self.meteconcept(concept1, concept2, operation)
            scores.append(score)
            gts.append(float(is_pos))
        scores = torch.stack(scores)
        gts = torch.tensor(gts, device=scores.device)
        loss = F.binary_cross_entropy_with_logits(input = scores, target=gts)
        is_corrects = (scores>0).float() == gts
        # correct_rate = is_corrects.sum().item() / num_sample
        return loss, is_corrects