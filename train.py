import argparse
import os
import gzip
import pickle
import json

import chainer
import numpy as np
from chainer import cuda, reporter, training
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

def load_data(config, data_type):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    vocab_path = os.path.join(config.data_dir, 'vocabulary.json')
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    if not os.path.exists(vocab_path):
        import pdb
        pdb.set_trace()
        word2vec_dict = shared['lower_word2vec']
        word_counter = shared['lower_word_counter']
        char_counter = shared['char_counter']
        shared['word2idx'] = {word: idx + 2 for idx, word in
                              enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th and word not in word2vec_dict)}
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(vocab_path, 'w'))

    else:
        new_shared = json.load(open(vocab_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    word2vec_dict = shared['lower_word2vec']
    new_word2idx_dict = {word: idx for idx, word in
                         enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
    shared['new_word2idx'] = new_word2idx_dict
    idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
    new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
    shared['new_emb_mat'] = new_emb_mat
            
    data = data_filter(data, config, data_type)
    if config.debug_mode:
        data = {k: data[k][:120] for k in data.keys()}
    dataset = SQuADDataset(data, shared, config)
    return dataset

def load_dataset(config):
    return load_data(config, 'train'), load_data(config, 'test')

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

def main():
    parser = argparse.ArgumentParser()
    pa = parser.add_argument    

    pa('--gpu', type=str, default='0')
    pa('--epoch', type=int, default=12)
    pa('--debug_mode', action='store_true')

    pa('--data_dir', type=str, default='../bi-att-flow/data/squad_nonsplit')
    pa('--ckpt_path', type=str, default='logs')
    pa('--log_path', type=str, default='logs')

    pa('--batch_size', type=int, default=60)
    pa('--display_step', type=int, default=50)
    pa('--eval_step', type=int, default=1000)

    pa('--init_lr', type=float, default=0.5)
    pa('--keep_prob', type=float, default=0.8)
    pa('--decay_rate', type=float, default=0.999)
    pa('--dropout_rate', type=float, default=0.2)

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

    pa('--use_glove_for_unk', action='store_false')
    
    config = parser.parse_args()
    print(json.dumps(config.__dict__, indent=4))

    train_data, test_data = load_dataset(config)
    config = update_config(config, [train_data, test_data])

    config.gpu = [int(g) for g in config.gpu.split(',')]
    config.enc_dim = config.word_emb_dim + config.char_out_dim

    unk_emb = np.random.normal(loc=0.0, scale=1.0,
                               size=(len(train_data.shared_data['word2idx']), config.word_emb_size))
    config.word_emb = np.concatenate((unk_emb, train_data.shared_data['new_emb_mat']), axis=0)
    config.word_vocab_size = len(config.word_emb)

    model = BiDAF(config)
    if config.gpu[0] >= 0:
        model.to_gpu(config.gpu[0])

    # chainer.config.use_cudnn = 'never'

    # optimizer
    optimizer = AdaDeltaWithLearningRate(lr=config.init_lr, eps=1e-08)
    optimizer.setup(model)

    # iterator
    train_iter = MultiprocessIterator(train_data, config.batch_size, repeat=True, shuffle=True)
    test_iter = MultiprocessIterator(test_data, config.batch_size, repeat=False, shuffle=False)

    # updater, trainer
    updater = training.StandardUpdater(train_iter, optimizer,
                                       converter=squad_converter, device=config.gpu[0])
    trainer = training.Trainer(updater, (config.epoch, 'epoch'), out=config.log_path)

    evaluator = extensions.Evaluator(test_iter, model,
                                     converter=squad_converter, device=config.gpu[0]) #, eval_hook=generate_graph)
    evaluator.name = 'val'

    iter_per_epoch = len(train_data) // config.batch_size
    print('Iter/epoch =', iter_per_epoch)

    log_trigger = (min(10, iter_per_epoch // 2), 'iteration')
    eval_trigger = (1000, 'iteration')
    record_trigger = training.triggers.MaxValueTrigger(
        'val/main/f1', eval_trigger)

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

