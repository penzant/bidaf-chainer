import argparse
import os
import gzip
import pickle
import json
import string
from collections import Counter

import chainer
import numpy as np
from chainer import cuda, function, reporter, training, serializers
from chainer import reporter as reporter_module
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions

from net import BiDAF
from dataset import SQuADDataset, update_config
from adadelta import AdaDeltaWithLearningRate

def data_filter(data, config, data_type):
    org_len = len(data['y'])
    del_idx = []
    for i, y in enumerate(data['y']):
        for start, stop in y:
            flag = True
            if stop[0] >= config.num_sents_th:
                flag = False
            if start[0] != stop[0]:
                flag = False
            if stop[1] >= config.sent_size_th:
                flag = False
            if not flag:
                del_idx.append(i)

    data_keys = data.keys()
    for i in sorted(del_idx, reverse=True):
        for k in data_keys:
            del data[k][i]

    print('{0}/{1} examples in {2} are filtered'.format(len(del_idx), org_len, data_type))
    return data

def load_data(config, data_type, vocab=None):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    if not vocab:
        vocab = {}
        vocab['word2vec'] = shared['lower_word2vec'] # known words
        word_counter = shared['lower_word_counter'] # all words with counter
        char_counter = shared['char_counter']
        vocab['word2idx'] = {word: idx for idx, word in
                             enumerate(word for word, count in word_counter.items()
                                       if count > config.word_count_th and word in vocab['word2vec'])}
        vocab['unk_word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th and word not in vocab['word2vec'])}
        vocab['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        NULL = "-NULL-"
        UNK = "-UNK-"
        vocab['unk_word2idx'][NULL] = 0
        vocab['unk_word2idx'][UNK] = 1
        vocab['char2idx'][NULL] = 0
        vocab['char2idx'][UNK] = 1

    # elif vocab:
    #     # word dict update
    #     word2vec_dict = shared['lower_word2vec']
    #     word_counter = shared['lower_word_counter']

    #     unk_words = [word for word, count in word_counter.items()
    #                  if count > config.word_count_th
    #                  and word not in vocab['word2idx']
    #                  and word not in vocab['unk_word2idx']
    #                  and word not in word2vec_dict]
    #     for w in unk_words:
    #         vocab['unk_word2idx'][w] = len(vocab['unk_word2idx'])

    #     part_words = [word for word, count in word_counter.items()
    #                   if count > config.word_count_th
    #                   and word not in vocab['word2idx']
    #                   and word not in vocab['unk_word2idx']
    #                   and word in word2vec_dict]
    #     for pw in part_words:
    #         vocab['word2idx'][pw] = len(vocab['word2idx'])
    #         vocab['word2vec'][pw] = word2vec_dict[pw]

    #     # char dict update
    #     char_counter = shared['char_counter']
    #     unk_chars = [char for char, count in char_counter.items()
    #                  if count > config.char_count_th
    #                  and char not in vocab['char2idx']]
    #     for c in unk_chars:
    #         vocab['char2idx'][c] = len(vocab['char2idx'])

    data = data_filter(data, config, data_type)
    if config.debug_mode:
        data = {k: data[k][:600] for k in data.keys()}
    return data, shared, vocab

def load_dataset(config):
    vocab_path = os.path.join(config.data_dir, 'vocabulary.json')
    vocab = json.load(open(vocab_path, 'r')) if os.path.exists(vocab_path) else None

    train_data, train_shared, vocab = load_data(config, 'train', vocab)
    test_data, test_shared, vocab = load_data(config, 'test', vocab)

    if not os.path.exists(vocab_path):
        json.dump(vocab, open(vocab_path, 'w'))

    # create merged word2idx and emb_mat into vocab
    unk_len = len(vocab['unk_word2idx'])
    idx2vec = {idx+unk_len: vocab['word2vec'][word]
               for word, idx in vocab['word2idx'].items()}

    unk_emb = np.random.normal(loc=0.0, scale=1.0,
                               size=(len(vocab['unk_word2idx']), config.word_emb_dim))
    known_emb = np.array([idx2vec[idx] for idx in sorted(idx2vec.keys())], dtype='float32')
    vocab['emb_mat'] = np.concatenate((unk_emb, known_emb), axis=0)

    word2idx = {word: idx+unk_len for word, idx in vocab['word2idx'].items()}
    word2idx.update(vocab['unk_word2idx'])
    vocab['word2idx'] = word2idx
    vocab['idx2word'] = {idx: word for word, idx in word2idx.items()}

    train_dataset = SQuADDataset(train_data, train_shared, vocab, config)
    test_dataset = SQuADDataset(test_data, test_shared, vocab, config)

    return train_dataset, test_dataset, vocab

def squad_converter(batch, device=None):
    if device >= 0:
        xp = cuda.cupy
        xp.cuda.Device(device).use()
    else:
        xp = np
    return {'x': xp.asarray([b[0] for b in batch], dtype=xp.int32),
            'cx': xp.asarray([b[1] for b in batch], dtype=xp.int32),
            'x_mask': xp.asarray([b[2] for b in batch], dtype=xp.bool_),
            'q': xp.asarray([b[3] for b in batch], dtype=xp.int32),
            'cq': xp.asarray([b[4] for b in batch], dtype=xp.int32),
            'q_mask': xp.asarray([b[5] for b in batch], dtype=xp.bool_),
            'y': xp.asarray([b[6] for b in batch], dtype=xp.bool_),
            'y2': xp.asarray([b[7] for b in batch], dtype=xp.bool_)}

class BiDAFEvaluator(training.extensions.Evaluator):

    def __init__(self, iterator, target, config, converter=squad_converter, device=None):
        super(BiDAFEvaluator, self).__init__(iterator, target, converter, device)
        self.log_path = config.log_path
        self.idx2word = config.idx2word
        self.word2idx = config.word2idx
        self.skip_word_in_result = [self.word2idx[w] for w in ['a', 'an', 'the', '']]
        self.eval_step = config.eval_step
        self.steps = 0

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']
        self.steps += self.eval_step

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        predicts, ids = [], []
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    loss, acc, f1, pred = eval_func(**in_arrays)
                    acc, f1 = self.calc_result([b[0] for b in batch], [b[9] for b in batch], pred)
                    predicts.extend(pred)
                    ids.extend([b[8] for b in batch])
                    reporter_module.report({'val/main/loss': loss.data, 'val/main/f1': f1, 'val/main/acc': acc})

            summary.add(observation)

        self.write_predict(predicts, ids, self.steps)

        return summary.compute_mean()

    def write_predict(self, preds, ids, step=0):
        out_path = os.path.join(self.log_path, 'predict_{0:06d}.json'.format(step))
        dump_pred = {id_: ' '.join([self.idx2word[wid] for wid in pred])
                     for pred, id_ in zip(preds, ids)}
        with open(out_path, 'w') as f:
            json.dump(dump_pred, f)

    def normalize_answer(self, words):
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        return remove_punc(' '.join(words)).split()

    def calc_result(self, x, ys, yps):
        match, f1 = [], []
        for xi, yi, ypi in zip(x, ys, yps):
            mi, f1i = 0.0, 0.0
            for yyi in yi:
                y_words = [int(w) for w in xi[0][yyi[0]:yyi[1]+1]]
                y_words = [w for w in y_words if w not in self.skip_word_in_result]
                yp_words = ypi
                y_tokens = [self.idx2word[w] for w in y_words]
                yp_tokens = [self.idx2word[w] for w in yp_words]
                y_words = self.normalize_answer(y_tokens)
                yp_words = self.normalize_answer(yp_tokens)
                if y_words == yp_words:
                    mi = 1.0
                common = Counter(y_words) & Counter(yp_words)
                num_same = sum(common.values())
                if num_same > 0:
                    precision = 1.0 * num_same / len(yp_words)
                    recall = 1.0 * num_same / len(y_words)
                    f1i = max(f1i, (2 * precision * recall) / (precision + recall))
            f1.append(f1i)
            match.append(mi)
        return (np.mean(match), np.mean(f1))


def main():
    parser = argparse.ArgumentParser()
    pa = parser.add_argument    

    pa('--gpu', type=str, default='0')
    pa('--epoch', type=int, default=20)
    pa('--debug_mode', action='store_true')
    pa('--resume', type=str, default='')

    pa('--data_dir', type=str, default='../bi-att-flow/data/squad_nonsplit')
    pa('--ckpt_path', type=str, default='logs')
    pa('--log_path', type=str, default='logs')

    pa('--batch_size', type=int, default=60)
    pa('--display_step', type=int, default=50)
    pa('--eval_step', type=int, default=500)

    pa('--init_lr', type=float, default=0.5)
    pa('--optimizer', type=str, default='adadelta')
    pa('--decay_rate', type=float, default=0.999)
    pa('--dropout_rate', type=float, default=0.2)
    pa('--no_ema', action='store_true')

    pa('--hidden_size', type=int, default=100)
    pa('--word_emb_dim', type=int, default=100)
    pa('--char_emb_dim', type=int, default=8)
    pa('--char_conv_n_kernel', type=int, default=100)
    pa('--char_conv_height', type=int, default=5)
    pa('--char_out_dim', type=int, default=100)

    pa('--highway_n_layer', type=int, default=2)

    pa('--word_count_th', type=int, default=10)
    pa('--char_count_th', type=int, default=50)
    pa('--sent_size_th', type=int, default=195) # 400
    pa('--para_size_th', type=int, default=256)
    pa('--num_sents_th', type=int, default=8)
    pa('--ques_size_th', type=int, default=30)
    pa('--word_size_th', type=int, default=16)

    config = parser.parse_args()
    print(json.dumps(config.__dict__, indent=4))

    train_data, test_data, vocab = load_dataset(config)
    config = update_config(config, [train_data, test_data], vocab)

    config.gpu = [int(g) for g in config.gpu.split(',')]
    config.enc_dim = config.word_emb_dim + config.char_out_dim

    model = BiDAF(config)
    if config.resume:
        serializers.load_npz(config.resume, model)

    # optimizer
    if config.optimizer == 'adam':
        optimizer = chainer.optimizers.Adam(0.001)
    else:
        optimizer = AdaDeltaWithLearningRate(lr=config.init_lr, eps=1e-08)
    optimizer.setup(model)
    model.word_emb.W.update_rule.enabled = False

    # iterator
    train_iter = MultiprocessIterator(train_data, config.batch_size, repeat=True, shuffle=True)
    test_iter = MultiprocessIterator(test_data, config.batch_size, repeat=False, shuffle=False)

    # updater, trainer
    if len(config.gpu) == 2:
        multi_devices = {'main': int(config.gpu[0]), 'second': int(config.gpu[1])}
        updater = training.ParallelUpdater(train_iter, optimizer,
                                           converter=squad_converter, devices=multi_devices)
    elif len(config.gpu) == 1:
        if config.gpu[0] >= 0:
            model.to_gpu(config.gpu[0])
        updater = training.StandardUpdater(train_iter, optimizer,
                                           converter=squad_converter, device=config.gpu[0])
    trainer = training.Trainer(updater, (config.epoch, 'epoch'), out=config.log_path)

    evaluator = BiDAFEvaluator(test_iter, model, config,
                               converter=squad_converter, device=config.gpu[0])
    evaluator.name = 'val'

    iter_per_epoch = len(train_data) // config.batch_size
    print('Iter/epoch =', iter_per_epoch)

    log_trigger = (min(config.display_step, iter_per_epoch // 2), 'iteration')
    eval_trigger = (config.eval_step, 'iteration') if iter_per_epoch > config.eval_step else (1, 'epoch')
    record_trigger = training.triggers.MaxValueTrigger('val/main/f1', eval_trigger)

    trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}.npz'),
                   trigger=record_trigger)
    trainer.extend(evaluator, trigger=eval_trigger)
    trainer.extend(extensions.LogReport(trigger=log_trigger, log_name='iteration.log'))
    trainer.extend(extensions.LogReport(trigger=eval_trigger, log_name='epoch.log'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/match', 'main/f1',
         'val/main/loss', 'val/main/match', 'val/main/f1', 'elapsed_time']))

    trainer.run()
    

if __name__ == '__main__':
    main()

