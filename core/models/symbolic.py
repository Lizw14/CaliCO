import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import concept_embedding
from core.models import utils

import numpy as np

import pdb
import jactorch.nn.functional as jacf

_apply_self_mask = {'relate': True, 'relate_ae': True}


def do_apply_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return m * (1 - self_mask) + (-10) * self_mask


def merge_fn(a, b, weight=None):
    if weight is not None:
        w = 1.4*torch.sigmoid(weight)
        b = w * b
    return torch.add(a, b)

def output_fn(a, weight=None):
    if weight is not None:
        a = a * weight.squeeze()
    return a


class Weight_Predictor(nn.Module):
    def __init__(self, hidden_size, cept2idx, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.NULL = 0

        self.module2idx = {k:i for i,k in enumerate(['<pad>', '<start>', '<end>', 'scene','select','filter','relate_o','relate_s','relate_ae',
            'intersect','union','intersect','exist','query','query_ae','query_rel_s','query_rel_o',
            'verify_rel_s','verify_rel_o','verify','choose','same','common','negate','unique'])}
        self.cept2idx = cept2idx        
        cept_embed = np.load('data/cept_glove_12.npy')
        rel_embed = np.load('data/rel_glove_12.npy')
        self.cept_embedding = nn.Embedding.from_pretrained(torch.tensor(np.concatenate((np.zeros((1,300)), cept_embed, rel_embed))))
        self.name_embedding = nn.Embedding(len(self.module2idx), rel_embed.shape[1])
        self.lstm = nn.LSTM(300, hidden_size, batch_first=True, bidirectional=True)

        self.predictor = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(True),
            nn.Linear(2*hidden_size, 1)
        )

        self.dict_order = ['concept', 'rel']

    def forward(self, input_programs):
        prog_lists = [self.list_to_prefix(input_program) for input_program in input_programs]
        name_seq, cept_seq = self.tokenize_prefix(prog_lists)
        embedded_cept = self.cept_embedding(cept_seq.cuda()) #[256, 11, 300]
        embedded_name = self.name_embedding(name_seq.cuda()) #[256, 11, 300]
        embedded = embedded_cept + embedded_name
        outputs, hidden = self.lstm(embedded.float()) #output: [256, 11, 600], hidden[0] hidden[1]: [2, 256, 300]
        logits = self.predictor(outputs) #[256, 11, 1]
        output = self.post_process(prog_lists, input_programs, logits)
        return output

    def post_process(self, prog_lists, input_programs, logits):
        output = []
        for i_batch, prog_list in enumerate(prog_lists):
            output.append([1]*len(input_programs[i_batch]))
            for idx, prog in enumerate(prog_list):
                if idx >= 0:
                    output[-1][prog['idx']] = logits[i_batch, idx+1] #+1: start
        return output
        

    def list_to_prefix(self, program_list):
        for idx, prog in enumerate(program_list):
            prog['idx'] = idx
        def build_subtree(cur):
            values = []
            for k in self.dict_order:
                if k in cur:
                    values.append(cur[k])
            return {
                'op': cur['op'],
                'values': values,
                'idx': cur['idx'],
                'inputs': [build_subtree(program_list[i]) if i!='_' else {'op': 'scene', 'values': [], 'idx':-1, 'inputs':[]} for i in cur['inputs']],
            }
        program_tree = build_subtree(program_list[-1])
        output = []
        def helper(cur):
            output.append({
                'op': cur['op'],
                'values': cur['values'],
                'idx': cur['idx'],
            })
            for node in cur['inputs']:
                helper(node)
        helper(program_tree)
        return output

    def tokenize_prefix(self, program_lists):
        #[{'op':'query, 'values':[]}]
        max_len = max([len(a) for a in program_lists])
        name_seq = torch.zeros((len(program_lists), max_len+2), dtype=int)
        cept_seq = torch.zeros((len(program_lists), max_len+2), dtype=int)
        name_seq[:, 0] = self.module2idx['<start>']
        for i, prog_list in enumerate(program_lists):
            for idx, prog in enumerate(prog_list):
                name_seq[i, idx+1] = self.module2idx[prog['op']]
                if len(prog['values'])>0:
                    identifier_rmnot = prog['values'][0].strip(')').split('(')[-1].strip()
                    cept_seq[i, idx+1] = self.cept2idx.get(identifier_rmnot, self.cept2idx['unk'])
            name_seq[i, idx+2] = self.module2idx['<end>']
        return name_seq, cept_seq


class ProgramExecutorContext(nn.Module):
    def __init__(self, attribute_taxonomy, relation_taxonomy, gdef, training=True):
        super().__init__()

        self.attribute_taxonomy = attribute_taxonomy
        self.relation_taxonomy = relation_taxonomy
        self.gdef = gdef

        self.train(training)


    def init_features(self, features, boxes):
        self.features = features
        self.boxes = boxes
        self.features_rel = self.features
        return self


    def select(self, selected, attribute, concept, weight=None):
        attribute_embeddings = self.attribute_taxonomy.embed(attribute, self.features, self.boxes)
        concept_embedding = self.attribute_taxonomy.get_concept_embedding(concept, weight=0) # weight=0
        mask = self.attribute_taxonomy.similarity(attribute_embeddings, concept_embedding, attribute)
        mask = merge_fn(selected, mask, weight)
        return mask, concept_embedding

    def filter(self, selected, attribute, concept, weight=None):
        attribute_embeddings = self.attribute_taxonomy.embed(attribute, self.features, self.boxes)
        concept_embedding = self.attribute_taxonomy.get_concept_embedding(concept, weight=1) # weight=1
        mask = self.attribute_taxonomy.similarity(attribute_embeddings, concept_embedding, attribute)
        mask_ = merge_fn(selected, mask, weight)
        return mask_, mask

    def relate_o(self, selected1, selected2, relconcept, weight=None):    # selected1 is the set where the target is in
        reltype = self.gdef.rel_cept2attr(relconcept)
        relation_embeddings = self.relation_taxonomy.pairwise_embed(reltype, self.features_rel, self.boxes)
        relconcept_embedding = self.relation_taxonomy.get_concept_embedding(relconcept, weight=0) #weight=0
        # mask: (num_objects, num_objects)
        mask = self.relation_taxonomy.similarity(relation_embeddings, relconcept_embedding, reltype)
        mask = do_apply_self_mask(mask)
        mask = (mask * selected2.unsqueeze(1)).sum(dim=0)
        mask_ = merge_fn(selected1, mask, weight)
        return mask_, mask

    def relate_s(self, selected1, selected2, relconcept, weight=None):    # selected1 is the set where the target is in
        # mask(i,j): score of obj_i(sbj) and obj_j(obj) being in relationship relconcept
        reltype = self.gdef.rel_cept2attr(relconcept)
        relation_embeddings = self.relation_taxonomy.pairwise_embed(reltype, self.features_rel, self.boxes)
        relconcept_embedding = self.relation_taxonomy.get_concept_embedding(relconcept, weight=0) #weight=0
        # mask: (num_objects, num_objects)
        mask = self.relation_taxonomy.similarity(relation_embeddings, relconcept_embedding, reltype)
        mask = do_apply_self_mask(mask)
        mask = (mask * selected2.unsqueeze(0)).sum(dim=1)
        mask_ = merge_fn(selected1, mask, weight)
        return mask_, mask

    def relate_ae(self, selected1, selected2, attribute, weight=None):    # selected1 is the set where the target is in
        attribute_embeddings = self.attribute_taxonomy.embed(attribute, self.features, self.boxes)
        mask = self.attribute_taxonomy.cross_similarity(attribute_embeddings, attribute)
        mask = do_apply_self_mask(mask)
        mask = (mask * selected2.unsqueeze(0)).sum(dim=1)
        mask = merge_fn(selected1, mask, weight)
        return mask

    def query(self, selected, attribute, choices=[], weight=None):
        attribute = 'output_' + attribute
        if len(choices) == 0:
            idx2word = self.attribute_taxonomy.idx2concepts
        else:
            idx2word = {i:k for i, k in enumerate(choices)}
        attribute_embeddings = self.attribute_taxonomy.embed(attribute, self.features, self.boxes)
        attribute_embedding = (selected.unsqueeze(-1) * attribute_embeddings).sum(dim=-2)
        concept_embeddings = self.attribute_taxonomy.locate_concept(choices, weight=2) #weight=2
        concept_score = self.attribute_taxonomy.similarity(attribute_embedding, concept_embeddings, attribute)
        concept_score = output_fn(concept_score, weight)
        return concept_score, idx2word

    def query_rel_s(self, selected1, selected2, choices, weight=None):
        # relation_embeddings: (num_objects, num_objects, embed_dim)
        reltype = self.gdef.rel_cept2attr(choices[0]) #TODO: reltype need to be refined
        reltype = 'output_' + reltype
        relation_embeddings = self.relation_taxonomy.pairwise_embed(reltype, self.features_rel, self.boxes)
        relation_embedding = (selected1.unsqueeze(-1).unsqueeze(-1) * relation_embeddings).sum(dim=-3)
        relation_embedding = (relation_embedding * selected2.unsqueeze(-1)).sum(dim=-2)
        relconcept_embeddings = self.relation_taxonomy.locate_concept(choices, weight=1) #weight=1
        relconcept_scores = self.relation_taxonomy.similarity(relation_embedding, relconcept_embeddings, reltype)
        relconcept_scores = output_fn(relconcept_scores, weight)
        return relconcept_scores, {i:k for i, k in enumerate(choices)}

    def query_rel_o(self, selected1, selected2, choices, weight=None):
        return self.query_rel_s(selected2, selected1, choices, weight)

    def verify(self, selected, attribute, concept, weight=None):
        attribute = 'output_' + attribute
        attribute_embeddings = self.attribute_taxonomy.embed(attribute, self.features, self.boxes)
        attribute_embedding = (selected.unsqueeze(-1) * attribute_embeddings).sum(dim=-2)
        concept_embedding = self.attribute_taxonomy.get_concept_embedding(concept, weight=3) #weight=3
        concept_score = self.attribute_taxonomy.similarity(attribute_embedding, concept_embedding, attribute)
        concept_score = output_fn(concept_score, weight)
        return concept_score

    def choose(self, selected1, selected2, attribute, concept, choices, weight=None):
        concept_score1 = self.verify(selected1, attribute, concept)
        concept_score2 = self.verify(selected2, attribute, concept)
        res = torch.stack([concept_score1, concept_score2])
        res = output_fn(res, weight)
        return res, {i:k for i, k in enumerate(choices)}

    def verify_rel_s(self, selected1, selected2, relconcept, weight=None):
        # relation_embeddings: (num_objects, num_objects, embed_dim)
        reltype = self.gdef.rel_cept2attr(relconcept)
        reltype = 'output_' + reltype
        relation_embeddings = self.relation_taxonomy.pairwise_embed(reltype, self.features_rel, self.boxes)
        relation_embedding = (selected1.unsqueeze(-1).unsqueeze(-1) * relation_embeddings).sum(dim=-3)
        relation_embedding = (relation_embedding * selected2.unsqueeze(-1)).sum(dim=-2)
        relconcept_embedding = self.relation_taxonomy.get_concept_embedding(relconcept, weight=2) #weight=2
        relconcept_score = self.relation_taxonomy.similarity(relation_embedding, relconcept_embedding, reltype)
        relconcept_score = output_fn(relconcept_score, weight)
        return relconcept_score

    def verify_rel_o(self, selected1, selected2, rel, weight=None):
        return self.verify_rel_s(selected2, selected1, rel, weight)

    def same(self, selected, attribute, weight=None):
        attribute = 'output_' + attribute
        attribute_embeddings = self.attribute_taxonomy.embed(attribute, self.features, self.boxes)
        selected_norm = F.softmax(selected, dim=-1)
        average_embedding = (selected_norm.unsqueeze(-1) * attribute_embeddings).sum(dim=-2)
        sims = self.attribute_taxonomy.embedding_similarity(attribute_embeddings, average_embedding, attribute)
        score = (selected_norm * sims).sum(dim=-1)
        score = output_fn(score, weight)
        return score

    def query_ae(self, selected1, selected2, attribute, weight=None):
        attribute = 'output_' + attribute
        attribute_embeddings = self.attribute_taxonomy.embed(attribute, self.features, self.boxes)
        attribute_embedding1 = (selected1.unsqueeze(-1) * attribute_embeddings).sum(dim=-2)
        attribute_embedding2 = (selected2.unsqueeze(-1) * attribute_embeddings).sum(dim=-2)
        score = self.attribute_taxonomy.embedding_similarity(attribute_embedding1, attribute_embedding2, attribute)
        score = output_fn(score, weight)
        return score

    def common(self, selected1, selected2, weight=None):
        all_attributes = self.gdef.attributes_asked_in_common
        attribute_scores = list()
        for attribute_ in all_attributes:
            attribute = 'output_' + attribute_
            attribute_embeddings = self.attribute_taxonomy.embed(attribute, self.features, self.boxes)
            attribute_embedding1 = (selected1.unsqueeze(-1) * attribute_embeddings).sum(dim=-2)
            attribute_embedding2 = (selected2.unsqueeze(-1) * attribute_embeddings).sum(dim=-2)
            attribute_score = self.attribute_taxonomy.embedding_similarity(attribute_embedding1, attribute_embedding2, attribute)
            attribute_scores.append(attribute_score)
        attribute_scores = torch.stack(attribute_scores)
        attribute_scores = output_fn(attribute_scores, weight)
        return attribute_scores, {i:k for i, k in enumerate(all_attributes)}

    def exist(self, selected, weight=None):
        res = selected.max(dim=-1)[0]
        res = output_fn(res, weight)
        return res

    def negate(self, score, weight=None):
        return -score

    def unique(self, selected, weight=None):
        return F.softmax(selected, dim=-1)

    def intersect(self, selected1, selected2, weight=None):
        return torch.min(selected1, selected2)

    def union(self, selected1, selected2, weight=None):
        return torch.max(selected1, selected2)


class DifferentiableReasoning(nn.Module):
    def __init__(self, input_dims, hidden_dims, gdef):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.gdef = gdef

        taxonomies = self.gdef.taxonomies
        all_cepts = []
        all_cepts.extend(gdef.sgg_vocab['attr'])
        all_cepts.extend(gdef.sgg_vocab['label'])
        glove_embed = utils.load_cept_embeddings(all_cepts, 'data/attrlabel_glove_taxo.npy')
        assert(glove_embed.shape[0]==len(all_cepts))
        self.attrlabel_embeddings = nn.Parameter(torch.Tensor(glove_embed))

        self.sym_attribute_taxonomy = concept_embedding.ConceptEmbedding(300, hidden_dims[0])
        self.sym_attribute_taxonomy.init_taxonomy(taxonomies[0], self.attrlabel_embeddings)
        self.sym_relation_taxonomy = concept_embedding.RelationConceptEmbedding(300, hidden_dims[1])
        self.sym_relation_taxonomy.init_taxonomy(taxonomies[1], self.attrlabel_embeddings, self.sym_attribute_taxonomy.preprocess)

        self.sym = ProgramExecutorContext(self.sym_attribute_taxonomy, self.sym_relation_taxonomy, 
            self.gdef, training=self.training)

        self.value2idx = {'<null>':0}
        self.value2idx.update({k:v+1 for k,v in self.sym_attribute_taxonomy.concepts2idx.items()})
        self.value2idx.update({k:v+1+len(self.sym_attribute_taxonomy.concepts2idx) for k,v in self.sym_relation_taxonomy.concepts2idx.items()})
        self.weight_predictor = Weight_Predictor(hidden_size=300, cept2idx=self.value2idx)

    def forward(self, feed_dict):

        programs = feed_dict['program_seq']
        buffers = []
        result = []
        lstm_logits, mixing = None, None

        weights = self.weight_predictor(feed_dict['program_seq'])


        batch_size = len(programs)
        current_obj = 0
        for idx in range(batch_size):
            buffer = []

            num_obj = feed_dict['num_objects'][idx]
            features = feed_dict['objects'][current_obj:current_obj+num_obj].cuda()
            boxes = feed_dict['boxes'][current_obj:current_obj+num_obj].cuda()
            attr_logits = feed_dict['attr_logits'][current_obj:current_obj+num_obj].cuda() if 'attr_logits' in feed_dict else None #num_obj*1313
            label_scores = feed_dict['label_scores'][current_obj:current_obj+num_obj].cuda() if 'label_scores' in feed_dict else None #num_obj*622
            prog = feed_dict['program_seq'][idx]
            current_obj += num_obj


            sym = None
            if attr_logits is not None or label_scores is not None:
                sym_feat = torch.cat((F.softmax(attr_logits, dim=-1), label_scores), dim=-1)
                ctx = self.sym.init_features(sym_feat, boxes)
            

            for block_id, block in enumerate(prog):
                op = block['op']

                # composing inputs for the module
                inputs = []
                # handling program error (with less inputs)
                if len(block['inputs']) < len(self.gdef.qtype2inputtype(op)):
                    print('padding inputs:', block)
                    if len(block['inputs']) == 0:
                        block['inputs'].extend(['_'] * len(self.gdef.qtype2inputtype(op)))
                    else:
                        block['inputs'].extend([block['inputs'][-1]] * (len(self.gdef.qtype2inputtype(op)) - len(block['inputs'])))
                for inp, inp_type in zip(block['inputs'], self.gdef.qtype2inputtype(op)):
                    if inp == '_':
                        inp = 0 + torch.zeros(features.shape[0], dtype=torch.float, device=features.device)
                    else:
                        inp = buffer[inp][0]
                    if inp_type == 'object':
                        inp = ctx.unique(inp)
                    inputs.append(inp)

                dict_order = ['attr', 'concept', 'rel', 'choices']
                for k in dict_order:
                    if k in block:
                        inputs.append(block[k])

                try:
                    inputs.append(weights[idx][block_id])
                    # run the module
                    assert hasattr(ctx, op)
                    
                    res = getattr(ctx, op)(*inputs)
                    if sym is not None:
                        res_sym = getattr(sym, op)(*inputs)
                        if type(res) != tuple:
                            res = (torch.min(res, res_sym), )
                        else:
                            res = (torch.min(res[0], res_sym[0]), res[1])
                    else:
                        if type(res) != tuple:
                            res = (res, )
                    buffer.append(res)
                except Exception as e:
                    print(e)
                    print('Error symbolic.py: ', prog)
                    buffer = buffers[-1]
                    op = result[-1][0]
                    break

                if op in ['query', 'common'] and block_id < (len(prog)-1):
                    print('query common not last', prog)
                    break

            if len(buffer)==0:
                buffer = buffers[-1]
                op = result[-1][0]

            result.append((op, buffer[-1]))
            buffers.append(buffer)

        return programs, buffers, (result, lstm_logits, mixing)
