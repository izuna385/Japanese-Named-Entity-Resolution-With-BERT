'''
Model classes
'''
import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import normalize
import torch.nn.functional as F
import copy
import pdb

class ResolutionLabelClassifier(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.accuracy = BooleanAccuracy()
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.mesloss = nn.MSELoss()
        self.istrainflag = 1

        self.linear_for_cat = nn.Linear(self.mention_encoder.get_output_dim()* 2, 1)
        self.linear_for_bond = nn.Linear(self.mention_encoder.get_output_dim(), 1)

    def forward(self, l, r, label, mention_uniq_id, l_plus_r):
        l_mention = self.mention_encoder(l)
        r_mention = self.mention_encoder(r)
        l_and_r_cross = self.mention_encoder(l_plus_r)

        if self.args.scoring_function_for_model == 'sep':
            # l_forcossim = normalize(l_mention, dim=1)
            # r_forcossim = normalize(r_mention, dim=1)
            # scores = l_forcossim.mm(r_forcossim.t())
            scores = self.linear_for_cat(torch.cat((l_mention, r_mention), dim=1))
        else:
            scores = self.linear_for_cat(torch.cat((l_and_r_cross, torch.abs(l_mention - r_mention)), dim=1))

        loss = self.BCEWloss(scores.view(-1), label.float())
        output = {'loss': loss}

        if self.istrainflag:
            binary_class = (torch.sigmoid(scores.view(-1)) > 0.5).int()
            self.accuracy(binary_class, label)
        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}


    def switch2eval(self):
        self.istrainflag = copy.copy(0)

    def calc_L2distance(self, h, t):
        diff = h - t
        return torch.norm(diff, dim=2)  # batch * cands