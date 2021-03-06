import random
import string

import chainer
import numpy as np


class SQuADDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data, shared_data, vocab, config):
        self.data = data
        self.shared_data = shared_data
        self.vocab = vocab
        self.config = config

    def __len__(self):
        return len(self.data['*x'])

    def get_example(self, i):
        config = self.config
        data_item = {k: self.data[k][i] for k in self.data.keys()}
        for k in self.data.keys():
            if k.startswith('*'):
                shared_key = k[1:]
                data_item[shared_key] = self.shared_idx(self.shared_data[shared_key], data_item[k])

        M = config.max_num_sents
        JX = config.max_sent_size
        JQ = config.max_ques_size
        W = config.max_word_size
        
        x = np.zeros([M, JX], dtype='int32')
        cx = np.zeros([M, JX, W], dtype='int32')
        x_mask = np.zeros([M, JX], dtype='bool')
        q = np.zeros([JQ], dtype='int32')
        cq = np.zeros([JQ, W], dtype='int32')
        q_mask = np.zeros([JQ], dtype='bool')
        y = np.zeros([M, JX], dtype='bool')
        y2 = np.zeros([M, JX], dtype='bool')

        ids = data_item['ids'] # for logging of results

        start_idx, stop_idx = data_item['y'][0] # random.choice(data_item['y'])
        j, k = start_idx
        j2, k2 = stop_idx
        y[j, k] = True
        y2[j2, k2-1] = True

        X = data_item['x']
        CX = data_item['cx']

        def _get_word(word):
            d = self.vocab['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            return 1

        def _get_char(char):
            d = self.vocab['char2idx']
            if char in d:
                return d[char]
            return 1

        for j, xij in enumerate(X):
            if j == config.max_num_sents:
                break
            for k, xijk in enumerate(xij):
                if k == config.max_sent_size:
                    break
                each = _get_word(xijk)
                assert isinstance(each, int), each
                x[j, k] = each
                x_mask[j, k] = True

        for j, cxij in enumerate(CX):
            if j == config.max_num_sents:
                break
            for k, cxijk in enumerate(cxij):
                if k == config.max_sent_size:
                    break
                for l, cxijkl in enumerate(cxijk):
                    if l == config.max_word_size:
                        break
                    cx[j, k, l] = _get_char(cxijkl)

        for j, qij in enumerate(data_item['q']):
            q[j] = _get_word(qij)
            q_mask[j] = True

        for j, cqij in enumerate(data_item['cq']):
            for k, cqijk in enumerate(cqij):
                cq[j, k] = _get_char(cqijk)
                if k + 1 == config.max_word_size:
                    break

        ys = [[y[0][1], y[1][1]] for y in data_item['y']]

        return (x, cx, x_mask, q, cq, q_mask, y, y2, ids, ys)

    def shared_idx(self, l, i):
        return self.shared_idx(l[i[0]], i[1:]) if len(i) > 1 else l[i[0]]


def update_config(config, datasets, vocab):
    config.max_num_sents = 0
    config.max_sent_size = 0
    config.max_ques_size = 0
    config.max_word_size = 0
    config.max_para_size = 0
    for data_set in datasets:
        data = data_set.data
        shared = data_set.shared_data
        for idx in range(len(data_set.data['*x'])):
            rx = data['*x'][idx]
            q = data['q'][idx]
            sents = shared['x'][rx[0]][rx[1]]
            config.max_para_size = max(config.max_para_size, sum(map(len, sents)))
            config.max_num_sents = max(config.max_num_sents, len(sents))
            config.max_sent_size = max(config.max_sent_size, max(map(len, sents)))
            config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
            if len(q) > 0:
                config.max_ques_size = max(config.max_ques_size, len(q))
                config.max_word_size = max(config.max_word_size, max(len(word) for word in q))

    config.max_num_sents = min(config.max_num_sents, config.num_sents_th)
    config.max_sent_size = min(config.max_sent_size, config.sent_size_th)
    config.max_para_size = min(config.max_para_size, config.para_size_th)

    config.max_word_size = min(config.max_word_size, config.word_size_th)

    config.char_vocab_size = len(vocab['char2idx'])
    config.word_emb_size = len(next(iter(vocab['word2vec'].values())))
    config.word_vocab_size = len(vocab['word2idx'])

    config.idx2word = vocab['idx2word']
    config.word2idx = vocab['word2idx']
    config.word_emb = vocab['emb_mat']

    skip_word = list(string.punctuation) + ['a', 'an', 'the', '']
    skip_word_in_result = [vocab['word2idx'][w]
                           for w in skip_word if w in vocab['word2idx']]
    config.skip_word_in_result = skip_word_in_result


    return config

