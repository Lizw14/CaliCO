#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:

<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""
import pdb
from nltk.stem import WordNetLemmatizer
import re

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}


def programstr_decompose(program_str):
  s = [p.split('_') for p in program_str.split()]
  progs = ['_'.join(a[:2]) for a in s]
  opers = ['_'.join(a[2:]) for a in s]
  return progs, opers

split_chrac = re.compile(r'[|,\(]\s*')
find_chrac = re.compile('\[(.*?)\]')
def valuestr_decompose(program_str,value_str):
  values = re.findall(find_chrac, value_str)
  values = [re.split(split_chrac, a.strip(')')) for a in values]
  for i,a in enumerate(values):
    if len(a)==3:
      # this is to handle the choose rel module, e.g: [man, to the left of|to the right of, s]
      # other modules should have num_arguments strictly less than 3
      values[i][1] = values[i][1]+'|'+values[i][2]
      values[i] = values[i][:2]
      #if a[1]=='to the right of' and a[2]=='to the left of':
      #  pass
      #elif a[2]=='to the right of' and a[1]=='to the left of':
      #  pass
      #elif a[1]=='behind' and a[2]=='in front of':
      #  pass
      #elif a[1]=='in front of' and a[2]=='behind':
      #  pass
      #else:
  for i,a in enumerate(values):
    if len(a)>3:
      print(program_str, value_str)
  #if 'choose_rel' in program_str:
  #    print(program_str, value_str)
  values_1 = [a[0].replace(' ','_') if len(a)>1 else '' for a in values]
  values_2 = [a[-1].replace(' ','_') for a in values]
  return values_1, values_2
  

def tokenize(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None,
      is_stem=False):
  """
  Tokenize a sequence, converting a string s into a list of (string) tokens by
  splitting on the specified delimiter. Optionally keep or remove certain
  punctuation marks and add start and end tokens.
  """
  if punct_to_keep is not None:
    for p in punct_to_keep:
      s = s.replace(p, '%s%s' % (delim, p))

  if punct_to_remove is not None:
    for p in punct_to_remove:
      s = s.replace(p, '')

  tokens = s.split(delim)
  if is_stem:
    tokens = [stem_word(a) for a in tokens]
  if add_start_token:
    tokens.insert(0, '<START>')
  if add_end_token:
    tokens.append('<END>')
  return tokens


wordnet_lemmatizer = WordNetLemmatizer()
stem_dict = {'men':'man','women':'woman','children':'child'}
def stem_word(word):
    #word = lemmatizer.lemmatize(word.strip().lower())
    word = wordnet_lemmatizer.lemmatize(word)
    if word in stem_dict:
        word = stem_dict[word]
    res = word.lower()
    return res


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, is_stem=False):
  token_to_count = {}
  tokenize_kwargs = {
    'delim': delim,
    'punct_to_keep': punct_to_keep,
    'punct_to_remove': punct_to_remove,
  }
  for seq in sequences:
    seq_tokens = tokenize(seq, add_start_token=False, add_end_token=False, is_stem=is_stem, **tokenize_kwargs)
    for token in seq_tokens:
      if token not in token_to_count:
        token_to_count[token] = 0
      token_to_count[token] += 1

  token_to_idx = {}
  for token, idx in SPECIAL_TOKENS.items():
    token_to_idx[token] = idx
  for token, count in sorted(token_to_count.items()):
    if count >= min_token_count:
      token_to_idx[token] = len(token_to_idx)

  return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False, verbose_unk=False):
  seq_idx = []
  for token in seq_tokens:
    if token not in token_to_idx:
      if allow_unk:
        if verbose_unk:
          print('UNK token:', token)
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seq_idx.append(token_to_idx[token])
  return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
  tokens = []
  if type(seq_idx) != list:
    seq_idx = seq_idx.squeeze().tolist()
  for idx in seq_idx:
    tokens.append(idx_to_token[idx])
    if stop_at_end and tokens[-1] == '<END>':
      break
  if delim is None:
    return tokens
  else:
    return delim.join(tokens)
