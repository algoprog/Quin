import os
import random
import torch
import math
import argparse
import transformers
import logging

from torch import nn
from torch.nn import DataParallel
from data_utils import build_batch, LoggingHandler, get_examples
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoConfig, RobertaModel, AutoModel, AutoTokenizer, AdamW

import torch.nn.functional as F
import numpy as np


class NLI(nn.Module):
    """
    NLI model based on BERT (using the code from: https://github.com/yg211/bert_nli)
    """

    def __init__(self, model_path=None, device='cuda', parallel=False, debug=False, label_num=3, batch_size=16):
        super(NLI, self).__init__()

        lm = 'roberta-large'

        if model_path is not None:
            configuration = AutoConfig.from_pretrained(lm)
            self.bert = RobertaModel(configuration)
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.vdim = 1024
        self.max_length = 256

        self.nli_head = nn.Linear(self.vdim, label_num)
        self.batch_size = batch_size

        if parallel:
            self.bert = DataParallel(self.bert)

        # load trained model
        if model_path is not None:
            sdict = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(sdict, strict=False)

        self.to(device)
        self.device = device
        self.debug = debug

    def load_model(self, sdict):
        if self.gpu:
            self.load_state_dict(sdict)
            self.to('cuda')
        else:
            self.load_state_dict(sdict)

    def forward(self, sent_pair_list):
        all_probs = None
        iterator = range(0, len(sent_pair_list), self.batch_size)
        if self.debug:
            iterator = tqdm(iterator, desc='batch')
        for batch_idx in iterator:
            probs = self.ff(sent_pair_list[batch_idx:batch_idx + self.batch_size]).data.cpu().numpy()
            if all_probs is None:
                all_probs = probs
            else:
                all_probs = np.append(all_probs, probs, axis=0)
        labels = []
        for pp in all_probs:
            ll = np.argmax(pp)
            if ll == 0:
                labels.append('entailment')
            elif ll == 1:
                labels.append('contradiction')
            else:
                labels.append('neutral')
        return labels, all_probs

    def ff(self, sent_pair_list):
        ids, types, masks = build_batch(self.tokenizer, sent_pair_list, max_len=self.max_length)
        if ids is None:
            return None
        ids_tensor = torch.tensor(ids)
        #ypes_tensor = torch.tensor(types)
        masks_tensor = torch.tensor(masks)

        ids_tensor = ids_tensor.to(self.device)
        #types_tensor = types_tensor.to(self.device)
        masks_tensor = masks_tensor.to(self.device)
        # self.bert.to('cuda')
        # self.nli_head.to('cuda')

        cls_vecs = self.bert(input_ids=ids_tensor, attention_mask=masks_tensor)[1]
        logits = self.nli_head(cls_vecs)
        predict_probs = F.log_softmax(logits, dim=1)
        return predict_probs

    def save(self, output_path, config_dic=None, acc=None):
        if acc is None:
            model_name = 'nli_large_2.state_dict'
        else:
            model_name = 'nli_large_2_acc{}.state_dict'.format(acc)
        opath = os.path.join(output_path, model_name)
        if config_dic is None:
            torch.save(self.state_dict(), opath)
        else:
            torch.save(config_dic, opath)


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    scheduler = scheduler.lower()
    if scheduler == 'constantlr':
        return transformers.optimization.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.optimization.get_constant_schedule_with_warmup(optimizer, warmup_steps)
    elif scheduler == 'warmuplinear':
        return transformers.optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)
    elif scheduler == 'warmupcosine':
        return transformers.optimization.get_cosine_schedule_with_warmup(optimizer, warmup_steps, t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, warmup_steps,
                                                                                            t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))


def train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, gpu,
          max_grad_norm, best_acc, model_save_path):
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    step_cnt = 0
    for pointer in tqdm(range(0, len(train_data), batch_size), desc='training'):
        step_cnt += 1
        sent_pairs = []
        labels = []
        for i in range(pointer, pointer + batch_size):
            if i >= len(train_data):
                break
            sents = train_data[i].get_texts()
            sent_pairs.append(sents)
            labels.append(train_data[i].get_label())
        predicted_probs = model.ff(sent_pairs)
        if predicted_probs is None:
            continue
        true_labels = torch.LongTensor(labels)
        if gpu:
            true_labels = true_labels.to('cuda')
        loss = loss_fn(predicted_probs, true_labels)
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step_cnt % 5000 == 0:
            acc = evaluate(model, dev_data, mute=True)
            logging.info('==> step {} dev acc: {}'.format(step_cnt, acc))
            model.train()  # model was in eval mode in evaluate(); re-activate the train mode
            if acc > best_acc:
                best_acc = acc
                logging.info('Saving model...')
                model.save(model_save_path, model.state_dict())

    return best_acc


def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli training")
    ap.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('-ep', '--epoch_num', type=int, default=10, help='epoch num')
    ap.add_argument('--fp16', type=int, default=0, help='use apex mixed precision training (1) or not (0)')
    ap.add_argument('--gpu', type=int, default=1, help='use gpu (1) or not (0)')
    ap.add_argument('-ss', '--scheduler_setting', type=str, default='WarmupLinear',
                    choices=['WarmupLinear', 'ConstantLR', 'WarmupConstant', 'WarmupCosine',
                             'WarmupCosineWithHardRestarts'])
    ap.add_argument('-mg', '--max_grad_norm', type=float, default=1., help='maximum gradient norm')
    ap.add_argument('-wp', '--warmup_percent', type=float, default=0.1,
                    help='how many percentage of steps are used for warmup')

    args = ap.parse_args()
    return args.batch_size, args.epoch_num, args.fp16, args.gpu, args.scheduler_setting, args.max_grad_norm, args.warmup_percent


def evaluate(model, test_data, mute=False):
    model.eval()
    sent_pairs = [test_data[i].get_texts() for i in range(len(test_data))]
    all_labels = [test_data[i].get_label() for i in range(len(test_data))]
    _, probs = model(sent_pairs)
    all_predict = [np.argmax(pp) for pp in probs]
    assert len(all_predict) == len(all_labels)

    acc = len([i for i in range(len(all_labels)) if all_predict[i] == all_labels[i]]) * 1. / len(all_labels)
    prf = precision_recall_fscore_support(all_labels, all_predict, average=None, labels=[0, 1])

    if not mute:
        print('==>acc<==', acc)
        print('==>precision-recall-f1<==\n', prf)

    return acc


if __name__ == '__main__':

    batch_size, epoch_num, fp16, gpu, scheduler_setting, max_grad_norm, warmup_percent = parse_args()
    fp16 = bool(fp16)
    gpu = bool(gpu)

    print('=====Arguments=====')
    print('batch size:\t{}'.format(batch_size))
    print('epoch num:\t{}'.format(epoch_num))
    print('fp16:\t{}'.format(fp16))
    print('gpu:\t{}'.format(gpu))
    print('scheduler setting:\t{}'.format(scheduler_setting))
    print('max grad norm:\t{}'.format(max_grad_norm))
    print('warmup percent:\t{}'.format(warmup_percent))
    print('=====Arguments=====')

    label_num = 3
    model_save_path = 'weights/entailment'

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # Read the dataset
    train_data = get_examples('../data/allnli/train.jsonl')
    dev_data = get_examples('../data/allnli/dev.jsonl')

    logging.info('train data size {}'.format(len(train_data)))
    logging.info('dev data size {}'.format(len(dev_data)))
    total_steps = math.ceil(epoch_num * len(train_data) * 1. / batch_size)
    warmup_steps = int(total_steps * warmup_percent)

    model = NLI(batch_size=batch_size, parallel=True)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-6, correct_bias=False)
    scheduler = get_scheduler(optimizer, scheduler_setting, warmup_steps=warmup_steps, t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    best_acc = -1.
    for ep in range(epoch_num):
        random.shuffle(train_data)
        logging.info('\n=====epoch {}/{}====='.format(ep, epoch_num))
        best_acc = train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, gpu,
                         max_grad_norm, best_acc, model_save_path)
