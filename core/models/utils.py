import torch
import numpy as np
import os.path as osp
import pdb

def broadcast(tensor, dim, size):
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    new_shape = shape[:dim] + (size,) + shape[dim+1:]
    return tensor.expand(new_shape)


def meshgrid(input1, input2=None, dim=1):
    """Perform np.meshgrid along given axis. It will generate a new dimension after dim."""
    if input2 is None:
        input2 = input1
    if dim < 0:
        dim += input1.dim()
    n, m = input1.size(dim), input2.size(dim)
    x = broadcast(input1.unsqueeze(dim + 1), dim + 1, m)
    y = broadcast(input2.unsqueeze(dim + 0), dim + 0, n)
    return x, y


def load_glove(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def load_cept_embeddings(concepts, output_file):
    if osp.exists(output_file):
        cept_emb = np.load(output_file)
    else:
        emb = load_glove('../glove/glove.6B.300d.txt')
        miss = 0
        cept_emb = np.zeros((len(concepts), 300), 'float32')
        for i, w in enumerate(concepts):
            flag = True
            for w_elem in w.lower().strip('2').split():
                if w_elem not in emb:
                    flag = False
                    cept_emb[i] += np.random.rand(300)
                else:
                    cept_emb[i] += emb[w_elem]
            cept_emb[i] = cept_emb[i] / len(w.split(' '))
            if not flag:
                miss += 1
                print(miss, ': ', w)

        print("miss = {}, miss_rate = {}".format(miss, float(miss)/len(concepts)))
        np.save(output_file, cept_emb)
    return cept_emb


from core.datasets.defi_attrcept import *
if __name__ == "__main__":
    tax = load_gdef(gdef_pth='iep/datasets/define_prog_12.json', 
        sgg_vocab_pth='/home/zhuowan/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/datasets/GQA/gqa_vocab_taxo.json').taxonomies
    # tax = load_gdef().taxonomies
    concepts = []
    for attribute in tax[0]:
        for concept in tax[0][attribute]:
            concepts.append(concept)
    cept_emb = load_cept_embeddings(concepts, 'data/cept_glove_12_taxo.npy')
    pdb.set_trace()
    print('Done')
