import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import utils


import pdb


'''
For pick symbol
'''
class AttributeBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gate = nn.Parameter(10*torch.ones(1313+622))
        self.weight_loc = nn.Parameter(torch.zeros(2))
        self.map_loc = nn.Sequential(
            nn.Linear(5, output_dim)
        )
        self.map = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x, box, attrlabel_embeddings, preprocess):
        x = x * torch.sigmoid(self.gate)
        embedding = torch.matmul(x, attrlabel_embeddings)
        embedding = preprocess(embedding, box)
        semantic = embedding
        spatial = self.map_loc(box)
        weights = torch.softmax(self.weight_loc, dim=-1)
        output = self.map(weights[0]*semantic + weights[1]*spatial)
        return output


class RelationBlock(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.gate1 = nn.Parameter(10*torch.ones(1313+622))
        self.gate2 = nn.Parameter(10*torch.ones(1313+622))
        self.weight_loc = nn.Parameter(torch.zeros(2))
        self.fc1 = nn.Linear(input_dim1 + input_dim2, output_dim)
        self.fc2 = nn.Linear(5+5, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, feature1, box1, feature2, box2, attrlabel_embeddings, preprocess):
        # feature1: (n1, d1), feature2: (n2, d2)
        # semantic: (n1, n2, output_dim)
        feature1 = feature1 * torch.sigmoid(self.gate1)
        feature1 = torch.matmul(feature1, attrlabel_embeddings)
        feature1 = preprocess(feature1, box1)
        feature2 = feature2 * torch.sigmoid(self.gate2)
        feature2 = torch.matmul(feature2, attrlabel_embeddings)
        feature2 = preprocess(feature2, box2)
        x1, x2 = utils.meshgrid(feature1, feature2, dim=0)
        semantic = self.relu(self.fc1(torch.cat((x1, x2), dim=-1)))
        x1, x2 = utils.meshgrid(box1, box2, dim=0)
        spatial = self.relu(self.fc2(torch.cat((x1, x2), dim=-1)))
        weights = torch.softmax(self.weight_loc, dim=-1)
        embedding = self.fc3(weights[0]*semantic + weights[1]*spatial)
        return embedding


class PreprocessBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.map = nn.Sequential(
            nn.Linear(5+input_dim, output_dim),
            nn.ReLU(True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x, box):
        embedding = self.map(torch.cat((x, box), dim=-1))
        return embedding


class ConceptEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, intermediate_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        if intermediate_dim is None:
            intermediate_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.all_attributes = list()
        self.all_concepts = list()
        self.attribute_operators = nn.ModuleDict()
        self.concept_embeddings = None
        self.belong = dict()

        self.bilinear = nn.Parameter(torch.randn(self.embed_dim, self.embed_dim))
        self.bilinear_bias = nn.Parameter(torch.randn(1))

        ## for complex relationship supervision
        self.emb2complex_real = nn.Linear(self.embed_dim, self.embed_dim//2)
        self.emb2complex_img = nn.Linear(self.embed_dim, self.embed_dim//2)
        self.relation_emb_real = nn.Parameter(torch.randn(self.embed_dim//2, 4))
        self.relation_emb_img = nn.Parameter(torch.randn(self.embed_dim//2, 4))


    def init_taxonomy(self, taxonomy, attrlabel_embeddings):
        for attribute in taxonomy:
            self.all_attributes.append(attribute)
            self.attribute_operators['attr_'+attribute] = AttributeBlock(self.intermediate_dim, self.embed_dim)
            self.all_attributes.append('output_'+attribute)
            self.attribute_operators['attr_output_'+attribute] = AttributeBlock(self.intermediate_dim, self.embed_dim)
            for concept in taxonomy[attribute]:
                self.all_concepts.append(concept)
                # TODO: one concept contained in different attributes
                self.belong[concept] = attribute

        
        # add unk
        self.all_attributes.append('unk')
        self.attribute_operators['attr_unk'] = AttributeBlock(self.intermediate_dim, self.embed_dim)
        self.all_concepts.append('unk')
        self.belong['unk'] = 'unk'


        glove_embed = utils.load_cept_embeddings(self.all_concepts, 'data/cept_glove_12.npy')
        self.concept_embeddings = nn.Parameter(torch.Tensor(glove_embed))
        assert (self.concept_embeddings.shape[0] == len(self.all_concepts))
        self.concept_embeddings_weight = nn.Parameter(torch.ones(4, len(self.all_concepts)))
        self.concept_embeddings_bias = nn.Parameter(torch.zeros(4, len(self.all_concepts)))
        self.concept_embeddings_modify = nn.Parameter(torch.randn((len(self.all_concepts), self.embed_dim)))

        self.preprocess = PreprocessBlock(self.input_dim, self.intermediate_dim)

        # idx2concepts, idx2attributes
        self.idx2concepts = {i: k for i,k in enumerate(self.all_concepts)}
        self.concepts2idx = {k: i for i,k in enumerate(self.all_concepts)}
        self.idx2attributes = {i: k for i,k in enumerate(self.all_attributes)}
        self.idx2attributes = {k: i for i,k in enumerate(self.all_attributes)}

        self.attrlabel_embeddings = attrlabel_embeddings

    @property
    def num_attributes(self):
        return len(self.all_attributes)

    @property
    def num_concepts(self):
        return len(all_concepts)

    def get_attribute_operator(self, identifier):
        if identifier not in self.all_attributes:
            identifier = 'unk'
        attr_identifier = 'attr_' + identifier
        if hasattr(self, 'attribute_op'):
            return self.attribute_op, self.attribute_embeddings[attr_identifier]
        else:
            return self.attribute_operators[attr_identifier]

    def all_concepts_asked_in_common(self):
        return ['color', 'material', 'shape']

    def embed(self, attribute, features, boxes):
        mapping = self.get_attribute_operator(attribute)
        embedding = mapping(features, boxes, self.attrlabel_embeddings, self.preprocess)
        return embedding

    _tau = 0.25
    _bias = 0

    def similarity(self, embedding1, embedding2, attribute, is_norm_1=True, is_norm_2=False):
        # embedding1: (embed_dim,) or (n1, embed_dim) or (n11, n12, embed_dim)
        # embedding2: (embed_dim,) or (n2, embed_dim)
        # hacky way to add bias for each concept
        if type(embedding2) == tuple:
            embedding2, bilinear_b = embedding2
        if embedding1.ndim < 2:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.ndim < 2:
            embedding2 = embedding2.unsqueeze(0)
        if is_norm_1:
            embedding1 = embedding1 / embedding1.norm(2, dim=-1, keepdim=True)
        if is_norm_2:
            embedding2 = embedding2 / embedding2.norm(2, dim=-1, keepdim=True)
        logits = (torch.matmul(embedding1, embedding2.T).squeeze() - self._bias) / self._tau
        return logits

    def cross_similarity(self, embeddings, attribute):
        return self.similarity(embeddings, embeddings, attribute)

    def embedding_similarity(self, embedding1, embedding2, attribute):
        sim = torch.matmul(torch.matmul(embedding1, self.bilinear), embedding2.T).squeeze() + self.bilinear_bias.squeeze()
        return sim

    def get_concept_embedding(self, identifier, weight=-1):
        identifier_rmnot = identifier.strip(')').split('(')[-1].strip() #TODO: hacky to deal with not, eg. not(on)
        if identifier_rmnot not in self.all_concepts:
            identifier_rmnot = 'unk'
        cept_identifier = self.concepts2idx[identifier_rmnot]
        embedding = self.concept_embeddings[cept_identifier]
        if identifier.startswith('not('):
            embedding = -embedding
        if weight>=0:
            w = self.concept_embeddings_weight[weight, cept_identifier]
            embedding = w * embedding
            b = self.concept_embeddings_bias[weight, cept_identifier]
            embedding = b + embedding
        return embedding
        
    def locate_concept(self, choices, weight=2):
        if len(choices)>0:
            concept_embeddings = list()
            for concept in choices:
                cept_embedding = self.get_concept_embedding(concept, weight)
                concept_embeddings.append(cept_embedding)
            if type(concept_embeddings[0]) == tuple:
                concept_embeddings = (torch.stack([c[0] for c in concept_embeddings]), torch.stack([c[1] for c in concept_embeddings]))
            else:
                concept_embeddings = torch.stack(concept_embeddings)
            return concept_embeddings
        concept_embeddings = self.concept_embeddings
        if weight>=0:
            concept_embeddings = self.concept_embeddings_weight[weight].unsqueeze(-1) * concept_embeddings
            concept_embeddings = self.concept_embeddings_bias[weight].unsqueeze(-1) + concept_embeddings
        return concept_embeddings

    def judge_relation(self, concept_embedding1, concept_embedding2):
        # ## VCML
        # A_to_B = self.logit_fn(concept_embedding1, concept_embedding2)
        # B_to_A = self.logit_fn(concept_embedding2, concept_embedding1).t()
        # logit_lambda = logit_ln(self.logit_fn.ln_lambda(
        #     concept_embedding1, concept_embedding2
        # ))
        # tensor = torch.stack([A_to_B, B_to_A, logit_lambda], dim=-1)
        # output = self.metaconcept_subnet(tensor)

        ## Tongfei's complex supervision
        emb1_real = self.emb2complex_real(concept_embedding1)
        emb1_img = self.emb2complex_img(concept_embedding1)
        emb2_real = self.emb2complex_real(concept_embedding2)
        emb2_img = self.emb2complex_img(concept_embedding2)
        output = torch.matmul(emb1_real * emb2_real, self.relation_emb_real) \
            + torch.matmul(emb1_real * emb2_img, self.relation_emb_img) \
            + torch.matmul(emb1_img * emb2_img, self.relation_emb_real) \
            - torch.matmul(emb1_img * emb2_real, self.relation_emb_img)
        return output



class RelationConceptEmbedding(ConceptEmbedding):
    def __init__(self, input_dim, embed_dim, intermediate_dim=None):
        super().__init__(input_dim, embed_dim, intermediate_dim)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        if intermediate_dim is None:
            intermediate_dim = embed_dim
        self.intermediate_dim = intermediate_dim

    def init_taxonomy(self, taxonomy, attrlabel_embeddings, preprocess=None):
        for reltype in taxonomy:
            self.all_attributes.append(reltype)
            self.attribute_operators['attr_'+reltype] = RelationBlock(self.intermediate_dim, self.intermediate_dim, self.embed_dim)
            self.all_attributes.append('output_'+reltype)
            self.attribute_operators['attr_output_'+reltype] = RelationBlock(self.intermediate_dim, self.intermediate_dim, self.embed_dim)
            for concept in taxonomy[reltype]:
                self.all_concepts.append(concept)
                self.belong[concept] = reltype
        # add unk
        self.all_attributes.append('unk')
        self.attribute_operators['attr_unk'] = RelationBlock(self.intermediate_dim, self.intermediate_dim, self.embed_dim)
        self.all_concepts.append('unk')
        self.belong[concept] = 'unk'

        glove_embed = utils.load_cept_embeddings(self.all_concepts, 'data/rel_glove_12.npy')
        self.concept_embeddings = nn.Parameter(torch.Tensor(glove_embed))
        assert (self.concept_embeddings.shape[0]==len(self.all_concepts))
        self.concept_embeddings_weight = nn.Parameter(torch.ones(3, len(self.all_concepts)))
        self.concept_embeddings_bias = nn.Parameter(torch.zeros(3, len(self.all_concepts)))

        # idx2concepts, idx2attributes
        self.idx2concepts = {i: k for i,k in enumerate(self.all_concepts)}
        self.concepts2idx = {k: i for i,k in enumerate(self.all_concepts)}
        self.idx2attributes = {i: k for i,k in enumerate(self.all_attributes)}
        self.idx2attributes = {k: i for i,k in enumerate(self.all_attributes)}

        if preprocess is None:
            self.preprocess = PreprocessBlock(self.input_dim, self.intermediate_dim)
        else:
            self.preprocess = preprocess
        self.attrlabel_embeddings = attrlabel_embeddings

    def pairwise_embed(self, reltype, features, boxes):
        # This function shold only be used for embeddinig relation concepts
        mapping = self.get_attribute_operator(reltype)
        embedding = mapping(features, boxes, features, boxes, self.attrlabel_embeddings, self.preprocess)
        return embedding
    
