#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The standard way to train a model. After training, also computes validation
and test error.

The user must provide a model (with ``--model``) and a task (with ``--task`` or
``--pytorch-teacher-task``).

Examples
--------

.. code-block:: shell

  python -m parlai.scripts.train -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
  python -m parlai.scripts.train -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128
  python -m parlai.scripts.train -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
from tqdm import tqdm
from math import exp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import signal
import json
import argparse
import pickle as pkl
from dataset import dataset, CRSdataset
from model import CrossModel
import torch.nn as nn
from torch import optim
import torch
import math

try:
    import torch.version
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score as nltkbleu
import re
from collections import Counter
import rouge


def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()


def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    train.add_argument("-batch_size","--batch_size",type=int,default=32)
    train.add_argument("-max_count","--max_count",type=int,default=5)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
    train.add_argument("-load_dict","--load_dict",type=str,default=None)
    train.add_argument("-learningrate","--learningrate",type=float,default=1e-3)
    train.add_argument("-optimizer","--optimizer",type=str,default='adam')
    train.add_argument("-momentum","--momentum",type=float,default=0)
    train.add_argument("-is_finetune","--is_finetune",type=bool,default=True)
    train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
    train.add_argument("-epoch","--epoch",type=int,default=30)
    train.add_argument("-gpu","--gpu",type=str,default='0,1')
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
    train.add_argument("-embedding_size","--embedding_size",type=int,default=300)

    train.add_argument("-n_heads", "--n_heads", type=int, default=2)
    train.add_argument("-n_layers", "--n_layers", type=int, default=2)
    train.add_argument("-ffn_size", "--ffn_size", type=int, default=300)

    train.add_argument("-dropout", "--dropout", type=float, default=0.1)
    train.add_argument("-attention_dropout", "--attention_dropout", type=float, default=0.0)
    train.add_argument("-relu_dropout", "--relu_dropout", type=float, default=0.1)

    train.add_argument("-learn_positional_embeddings", "--learn_positional_embeddings", type=bool, default=False)
    train.add_argument("-embeddings_scale", "--embeddings_scale", type=bool, default=True)

    train.add_argument("-n_entity", "--n_entity", type=int, default=64369)
    train.add_argument("-n_relation", "--n_relation", type=int, default=214)
    train.add_argument("-n_concept", "--n_concept", type=int, default=29308)
    train.add_argument("-n_con_relation", "--n_con_relation", type=int, default=48)
    train.add_argument("-dim", "--dim", type=int, default=128)
    train.add_argument("-n_hop", "--n_hop", type=int, default=2)
    train.add_argument("-kge_weight", "--kge_weight", type=float, default=1)
    train.add_argument("-l2_weight", "--l2_weight", type=float, default=2.5e-6)
    train.add_argument("-n_memory", "--n_memory", type=float, default=32)
    train.add_argument("-item_update_mode", "--item_update_mode", type=str, default='0,1')
    train.add_argument("-using_all_hops", "--using_all_hops", type=bool, default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)

    return train


class TrainLoop_fusion_rec():
    def __init__(self, opt, is_finetune):
        self.opt = opt
        self.train_dataset = dataset('data/train_data.jsonl', opt)

        self.dict = self.train_dataset.word2index
        self.index2word = {self.dict[key]: key for key in self.dict}

        self.batch_size = self.opt['batch_size']
        self.epoch = self.opt['epoch']

        self.use_cuda = opt['use_cuda']
        if opt['load_dict'] != None:
            self.load_data = True
        else:
            self.load_data = False
        self.is_finetune = False

        self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec = {"recall@1": 0, "recall@10": 0, "recall@50": 0, "loss": 0, "count": 0, "ndcg@1": 0,
                            "ndcg@10": 0, "ndcg@50": 0, "mrr@1": 0, "mrr@10": 0, "mrr@50": 0}
        self.metrics_gen = {"ppl": 0, "intra-distinct-1": 0, "intra-distinct-2": 0, "intra-distinct-3": 0,
                            "intra-distinct-4": 0, "bleu-1": 0, "bleu-2": 0, "bleu-3": 0, "bleu-4": 0,
                            "inter-distinct-1": 0, "inter-distinct-2": 0, "inter-distinct-3": 0,
                            "inter-distinct-4": 0, "rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "bleu-count": 0, }

        self.build_model(is_finetune)

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self, is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        # self.model.load_model()
        losses = []
        best_val_rec = 0
        rec_stop = False
        for i in range(3):
            train_set = CRSdataset(self.train_dataset.data_process(), self.opt['n_entity'], self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                               batch_size=self.batch_size,
                                                               shuffle=False)
            num = 0
            for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                    train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _ = self.model(context.cuda(),
                                                                                                       response.cuda(),
                                                                                                       mask_response.cuda(),
                                                                                                       concept_mask,
                                                                                                       dbpedia_mask,
                                                                                                       seed_sets, movie,
                                                                                                       concept_vec,
                                                                                                       db_vec,
                                                                                                       entity_vector.cuda(),
                                                                                                       rec, test=False)

                joint_loss = info_db_loss  # +info_con_loss

                losses.append([info_db_loss])
                self.backward(joint_loss)
                self.update_params()
                if num % 50 == 0:
                    print('info db loss is %f' % (sum([l[0] for l in losses]) / len(losses)))
                    # print('info con loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    losses = []
                num += 1

        print("masked loss pre-trained")
        losses = []

        for i in range(self.epoch):
            train_set = CRSdataset(self.train_dataset.data_process(), self.opt['n_entity'], self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                               batch_size=self.batch_size,
                                                               shuffle=False)
            num = 0
            for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                    train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _ = self.model(context.cuda(),
                                                                                                       response.cuda(),
                                                                                                       mask_response.cuda(),
                                                                                                       concept_mask,
                                                                                                       dbpedia_mask,
                                                                                                       seed_sets, movie,
                                                                                                       concept_vec,
                                                                                                       db_vec,
                                                                                                       entity_vector.cuda(),
                                                                                                       rec, test=False)

                joint_loss = rec_loss + 0.025 * info_db_loss  # +0.0*info_con_loss#+mask_loss*0.05

                losses.append([rec_loss, info_db_loss])
                self.backward(joint_loss)
                self.update_params()
                if num % 50 == 0:
                    print('rec loss is %f' % (sum([l[0] for l in losses]) / len(losses)))
                    print('info db loss is %f' % (sum([l[1] for l in losses]) / len(losses)))
                    losses = []
                num += 1

            output_metrics_rec = self.val()

            if best_val_rec > output_metrics_rec["recall@50"] + output_metrics_rec["recall@1"]:
                rec_stop = True
            else:
                best_val_rec = output_metrics_rec["recall@50"] + output_metrics_rec["recall@1"]
                self.model.save_model()
                print("recommendation model saved once------------------------------------------------")

            if rec_stop:
                break

        _ = self.val(is_test=True)

    def metrics_cal_rec(self, rec_loss, scores, labels):
        batch_size = len(labels.view(-1).tolist())
        self.metrics_rec["loss"] += rec_loss
        outputs = scores.cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        for b in range(batch_size):
            if labels[b].item() == 0:
                continue
            target_idx = self.movie_ids.index(labels[b].item())
            # Recall
            self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())

            # MRR
            self.metrics_rec["mrr@1"] += TrainLoop_fusion_rec.mrr_calculation(target_idx, pred_idx[b], 1)
            self.metrics_rec["mrr@10"] += TrainLoop_fusion_rec.mrr_calculation(target_idx, pred_idx[b], 10)
            self.metrics_rec["mrr@50"] += TrainLoop_fusion_rec.mrr_calculation(target_idx, pred_idx[b], 50)

            # NDCG
            self.metrics_rec["ndcg@1"] += TrainLoop_fusion_rec.ndcg_calculation(target_idx, pred_idx[b], 1)
            self.metrics_rec["ndcg@10"] += TrainLoop_fusion_rec.ndcg_calculation(target_idx, pred_idx[b], 10)
            self.metrics_rec["ndcg@50"] += TrainLoop_fusion_rec.ndcg_calculation(target_idx, pred_idx[b], 50)
            self.metrics_rec["count"] += 1

    @staticmethod
    def mrr_calculation(target_idx, pred_idx, k):
        """Compute mean reciprocal rank (MRR)"""
        if target_idx in pred_idx[:k].tolist():
            rank = pred_idx.tolist().index(target_idx)
            return 1 / (rank + 1)
        return 0

    @staticmethod
    def ndcg_calculation(target_idx, pred_idx, k):
        """Compute NDCG"""
        if target_idx in pred_idx[:k].tolist():
            rank = pred_idx.tolist().index(target_idx)
            return 1 / math.log2(rank + 2)
        return 0

    def val(self, is_test=False):
        self.metrics_gen = {"ppl": 0, "intra-distinct-1": 0, "intra-distinct-2": 0, "intra-distinct-3": 0,
                            "intra-distinct-4": 0, "bleu-1": 0, "bleu-2": 0, "bleu-3": 0, "bleu-4": 0,
                            "inter-distinct-1": 0, "inter-distinct-2": 0, "inter-distinct-3": 0,
                            "inter-distinct-4": 0, "rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "count": 0, }
        self.metrics_rec = {"recall@1": 0, "recall@10": 0, "recall@50": 0, "loss": 0, "ndcg@1": 0, "ndcg@10": 0,
                            "ndcg@50": 0, "mrr@1": 0, "mrr@10": 0, "mrr@50": 0, "gate": 0, "count": 0, 'gate_count': 0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set = CRSdataset(val_dataset.data_process(), self.opt['n_entity'], self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                         batch_size=self.batch_size,
                                                         shuffle=False)
        recs = []
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(
                    context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,
                    concept_vec, db_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)

            recs.extend(rec.cpu())
            # print(losses)
            # exit()
            self.metrics_cal_rec(rec_loss, rec_scores, movie)

        output_dict_rec = {key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        print(output_dict_rec)

        return output_dict_rec

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr, 'amsgrad': True, 'betas': (0.9, 0.999)}

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()


class TrainLoop_fusion_gen():
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def __init__(self, opt, is_finetune):
        self.opt = opt
        self.train_dataset = dataset('data/train_data.jsonl', opt)

        self.dict = self.train_dataset.word2index
        self.index2word = {self.dict[key]: key for key in self.dict}

        self.batch_size = self.opt['batch_size']
        self.epoch = self.opt['epoch']

        self.use_cuda = opt['use_cuda']
        if opt['load_dict'] != None:
            self.load_data = True
        else:
            self.load_data = False
        self.is_finetune = False

        self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec = {"recall@1": 0, "recall@10": 0, "recall@50": 0, "loss": 0, "count": 0, "ndcg@1": 0,
                            "ndcg@10": 0, "ndcg@50": 0, "mrr@1": 0, "mrr@10": 0, "mrr@50": 0}
        self.metrics_gen = {"ppl": 0, "intra-distinct-1": 0, "intra-distinct-2": 0, "intra-distinct-3": 0,
                            "intra-distinct-4": 0, "bleu-1": 0, "bleu-2": 0, "bleu-3": 0, "bleu-4": 0,
                            "inter-distinct-1": 0, "inter-distinct-2": 0, "inter-distinct-3": 0,
                            "inter-distinct-4": 0, "rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "count": 0, }
        self.total_counts_1 = Counter()
        self.total_counts_2 = Counter()
        self.total_counts_3 = Counter()
        self.total_counts_4 = Counter()
        self.build_model(is_finetune=True)

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self, is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        self.model.load_model()
        losses = []
        best_val_gen = 1000
        gen_stop = False
        for i in range(self.epoch * 3):
            train_set = CRSdataset(self.train_dataset.data_process(True), self.opt['n_entity'], self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                               batch_size=self.batch_size,
                                                               shuffle=False)
            num = 0
            for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                    train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss = self.model(
                    context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,
                    concept_vec, db_vec, entity_vector.cuda(), rec, test=False)

                joint_loss = gen_loss

                losses.append([gen_loss])
                self.backward(joint_loss)
                self.update_params()
                if num % 50 == 0:
                    print('gen loss is %f' % (sum([l[0] for l in losses]) / len(losses)))
                    losses = []
                num += 1

            output_metrics_gen = self.val(True)
            if best_val_gen < output_metrics_gen["intra-distinct-4"]:
                pass
            else:
                best_val_gen = output_metrics_gen["intra-distinct-4"]
                self.model.save_model()
                print("generator model saved once------------------------------------------------")

        _ = self.val(is_test=True)

    def val(self, is_test=False):
        print(f"is_test: {is_test}")
        self.metrics_gen = {"ppl": 0, "intra-distinct-1": 0, "intra-distinct-2": 0, "intra-distinct-3": 0,
                            "intra-distinct-4": 0, "bleu-1": 0, "bleu-2": 0, "bleu-3": 0, "bleu-4": 0,
                            "inter-distinct-1": 0, "inter-distinct-2": 0, "inter-distinct-3": 0,
                            "inter-distinct-4": 0, "rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "count": 0, }
        self.metrics_rec = {"recall@1": 0, "recall@10": 0, "recall@50": 0, "loss": 0, "ndcg@1": 0, "ndcg@10": 0,
                            "ndcg@50": 0, "mrr@1": 0, "mrr@10": 0, "mrr@50": 0, "gate": 0, "count": 0,
                            'gate_count': 0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set = CRSdataset(val_dataset.data_process(True), self.opt['n_entity'], self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                         batch_size=self.batch_size,
                                                         shuffle=False)
        inference_sum = []
        golden_sum = []
        context_sum = []
        losses = []
        recs = []
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                _, _, _, _, gen_loss, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(),
                                                                                          response.cuda(),
                                                                                          mask_response.cuda(),
                                                                                          concept_mask, dbpedia_mask,
                                                                                          seed_sets, movie, concept_vec,
                                                                                          db_vec, entity_vector.cuda(),
                                                                                          rec, test=False)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(
                    context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,
                    concept_vec, db_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)

            golden_sum.extend(self.vector2sentence(response.cpu()))
            inference_sum.extend(self.vector2sentence(preds.cpu()))
            context_sum.extend(self.vector2sentence(context.cpu()))
            recs.extend(rec.cpu())
            losses.append(torch.mean(gen_loss))
            # print(losses)
            # exit()

        self.metrics_cal_gen(losses, inference_sum, golden_sum, recs)

        output_dict_gen = {}
        for key in self.metrics_gen:
            if any(item in key for item in ['intra-distinct', 'bleu', 'rouge']):
                output_dict_gen[key] = self.metrics_gen[key] / self.metrics_gen['count']
            elif key == 'inter-distinct-1':
                output_dict_gen[key] = len(self.total_counts_1) / sum(self.total_counts_1.values())
            elif key == 'inter-distinct-2':
                output_dict_gen[key] = len(self.total_counts_2) / sum(self.total_counts_2.values())
            elif key == 'inter-distinct-3':
                output_dict_gen[key] = len(self.total_counts_3) / sum(self.total_counts_3.values())
            elif key == 'inter-distinct-4':
                output_dict_gen[key] = len(self.total_counts_4) / sum(self.total_counts_4.values())
            else:
                output_dict_gen[key] = self.metrics_gen[key]
        print(output_dict_gen)

        f = open('context_test.txt', 'w', encoding='utf-8')
        f.writelines([' '.join(sen) + '\n' for sen in context_sum])
        f.close()

        f = open('output_test.txt', 'w', encoding='utf-8')
        f.writelines([' '.join(sen) + '\n' for sen in inference_sum])
        f.close()
        return output_dict_gen

    @staticmethod
    def _bleu(sen, tar, k):
        def correct_weights(k):
            """Correct bleu weights"""
            return [[1 / k for _ in range(k)]]

        bleu_score = sentence_bleu([tar], 
                                   sen, 
                                   weights=correct_weights(k), 
                                   smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1
                                   )
        return bleu_score

    def _distinct(self, outs, k):
        def ngram(seq, n):
            """Extract n-gram subsequences from a given sequence."""
            for i in range(len(seq) - n + 1):
                yield tuple(seq[i: i + n])

        print(f"Calculate distinct-{k}.")
        processed_lst = list()
        sample_intra = 0.0
        for i in tqdm(range(len(outs))):
            sen = outs[i]
            for token in sen:
                s = TrainLoop_fusion_gen._normalize_answer(token)
                if s != "":
                    processed_lst.append(s)
            counts = Counter(ngram(processed_lst, k))
            if k == 1:
                self.total_counts_1.update(counts)
            elif k == 2:
                self.total_counts_2.update(counts)
            elif k == 3:
                self.total_counts_3.update(counts)
            elif k == 4:
                self.total_counts_4.update(counts)
            sample_intra += max(len(counts), 1e-12) / max(sum(counts.values()), 1e-5)
        return sample_intra

    @staticmethod
    def _normalize_answer(s):
        """
        Lower text and remove punctuation, articles and extra whitespace.
        """
        s = s.lower()
        s = TrainLoop_fusion_gen.re_punc.sub('', s)
        s = TrainLoop_fusion_gen.re_art.sub('', s)
        # TODO: this could almost certainly be faster with a regex \s+ -> ' '
        return s

    @staticmethod
    def _rouge_score(guess, answers, measure='r'):
        """Compute ROUGE score."""
        evaluator = rouge.Rouge()
        try:
            scores = evaluator.get_scores(guess, answers)
        except LookupError:
            return [None, None, None]

        scores_rouge1 = max(score['rouge-1'][measure] for score in scores)
        scores_rouge2 = max(score['rouge-2'][measure] for score in scores)
        scores_rougeL = max(score['rouge-l'][measure] for score in scores)

        return [scores_rouge1, scores_rouge2, scores_rougeL]

    @staticmethod
    def _rouge(preds, responses):
        rouge1 = 0.0
        rouge2 = 0.0
        rougeL = 0.0
        for i in range(len(preds)):
            results = TrainLoop_fusion_gen._rouge_score(" ".join(preds[i]), " ".join(responses[i]))
            rouge1 += results[0]
            rouge2 += results[1]
            rougeL += results[2]
        return [rouge1, rouge2, rougeL]

    def metrics_cal_gen(self, rec_loss, preds, responses, recs):
        predict_s = preds
        golden_s = responses
        # print(rec_loss[0])
        self.metrics_gen["ppl"] += sum([exp(ppl) for ppl in rec_loss]) / len(rec_loss)
        generated = []

        for out, tar, rec in zip(predict_s, golden_s, recs):
            generated.append(out)
            self.metrics_gen['bleu-1'] += self._bleu(out, tar, 1)
            self.metrics_gen['bleu-2'] += self._bleu(out, tar, 2)
            self.metrics_gen['bleu-3'] += self._bleu(out, tar, 3)
            self.metrics_gen['bleu-4'] += self._bleu(out, tar, 4)
            self.metrics_gen['count'] += 1

        rouge_results = self._rouge(predict_s, golden_s)
        self.metrics_gen['rouge-1'] += rouge_results[0]
        self.metrics_gen['rouge-2'] += rouge_results[1]
        self.metrics_gen['rouge-l'] += rouge_results[2]

        self.metrics_gen['intra-distinct-1'] += self._distinct(generated, 1)
        self.metrics_gen['intra-distinct-2'] += self._distinct(generated, 2)
        self.metrics_gen['intra-distinct-3'] += self._distinct(generated, 3)
        self.metrics_gen['intra-distinct-4'] += self._distinct(generated, 4)

    def vector2sentence(self, batch_sen):
        sentences = []
        for sen in batch_sen.numpy().tolist():
            sentence = []
            for word in sen:
                if word > 3:
                    sentence.append(self.index2word[word])
                elif word == 3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()


if __name__ == '__main__':
    args = setup_args().parse_args()
    print(vars(args))
    if args.is_finetune == False:
        loop = TrainLoop_fusion_rec(vars(args), is_finetune=False)
        # loop.model.load_model()
        loop.train()
    else:
        loop = TrainLoop_fusion_gen(vars(args), is_finetune=True)
        # loop.train()
        loop.model.load_model()
        # met = loop.val(True)
        loop.train()
    met = loop.val(True)
    # print(met)
