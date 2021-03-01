import tempfile
from typing import Dict, Iterable
from overrides import overrides
from commons import BOND_TOKEN
import torch
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from parameters import Params
import glob
import os
import random
import pdb
from tqdm import tqdm
import json
from tokenizer import CustomTokenizer
import numpy as np
import pdb


class NamedEntityResolutionReader(DatasetReader):
    def __init__(
        self,
        config,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_tokenizer_class = CustomTokenizer(config=config)
        self.token_indexers = self.custom_tokenizer_class.token_indexer_returner()
        self.max_tokens = max_tokens
        self.config = config
        self.train_mention_ids, self.dev_mention_ids, self.test_mention_ids, self.mention_id2data = \
            self._mention_ids_returner()

    @overrides
    def _read(self, train_dev_test_flag: str) -> list:
        '''
        :param train_dev_test_flag: 'train', 'dev', 'test'
        :return: list of instances
        '''
        mention_ids, instances = list(), list()
        if train_dev_test_flag == 'train':
            mention_ids += self.train_mention_ids
            # Because Iterator(shuffle=True) has bug, we forcefully shuffle train dataset here.
            random.shuffle(mention_ids)
        elif train_dev_test_flag == 'dev':
            mention_ids += self.dev_mention_ids
        elif train_dev_test_flag == 'test':
            mention_ids += self.test_mention_ids
        elif train_dev_test_flag == 'train_and_dev':
            mention_ids += self.train_mention_ids
            mention_ids += self.dev_mention_ids


        for idx, mention_uniq_id in tqdm(enumerate(mention_ids)):
            instances.append(self.text_to_instance(mention_uniq_id, data=self.mention_id2data[mention_uniq_id]))

        return instances

    @overrides
    def text_to_instance(self, mention_uniq_id, data=None) -> Instance:
        l_tokenized = [Token('[CLS]')]
        l_tokenized += [Token(split_token) for split_token in self.custom_tokenizer_class.tokenize(txt=data['l'])]
        l_tokenized.append(Token('[SEP]'))
        r_tokenized = [Token('[CLS]')]
        r_tokenized += [Token(split_token) for split_token in self.custom_tokenizer_class.tokenize(txt=data['r'])]
        r_tokenized.append(Token('[SEP]'))

        l_plus_r = [Token('[CLS]')]
        l_plus_r += [Token(split_token) for split_token in self.custom_tokenizer_class.tokenize(txt=data['l'])]
        l_plus_r += [Token(BOND_TOKEN)]
        l_plus_r += [Token(split_token) for split_token in self.custom_tokenizer_class.tokenize(txt=data['r'])]
        l_plus_r += [Token('[SEP]')]

        context_field = TextField(l_tokenized, self.token_indexers)
        fields = {"l": context_field}
        fields['r'] = TextField(r_tokenized, self.token_indexers)
        fields['l_plus_r'] = TextField(l_plus_r, self.token_indexers)
        fields['label'] = ArrayField(np.array(data['label']))
        fields['mention_uniq_id'] = ArrayField(np.array(mention_uniq_id))


        return Instance(fields)


    def _mention_ids_returner(self):
        mention_id2data = {}
        train_mention_ids, dev_mention_ids, test_mention_ids = [], [], []
        dataset_dir = self.config.dataset_dir

        for train_dev_test_flag in ['train', 'dev', 'test']:
            with open(dataset_dir+train_dev_test_flag+'.tsv', 'r') as f:
                for idx, line in enumerate(f):
                    if idx != 0 and line.strip() != '':
                        data = line.strip().split('\t')
                        l = data[0]
                        r = data[1]
                        label = int(data[2])

                        mention_id = len(mention_id2data)
                        mention_id2data.update({mention_id: {'l': l,
                                                             'r': r,
                                                             'label': label}})

                        if train_dev_test_flag == 'train':
                            train_mention_ids.append(mention_id)
                        elif train_dev_test_flag == 'dev':
                            dev_mention_ids.append(mention_id)
                        elif train_dev_test_flag == 'test':
                            test_mention_ids.append(mention_id)

                    if self.config.debug:
                        if idx == 10000:
                            break

        return train_mention_ids, dev_mention_ids, test_mention_ids, mention_id2data

if __name__ == '__main__':
    params = Params()
    config = params.opts
    dsr = NamedEntityResolutionReader(config=config)
    dsr._read('train')