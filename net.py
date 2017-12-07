from collections import Counter

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import reporter

from utils import softsel, get_logits
from ema import ExponentialMovingAverage

class CharacterConvolution(chainer.Chain):

    def __init__(self, config):
        super(CharacterConvolution, self).__init__()
        with self.init_scope():
            k = (1, config.char_conv_height)
            s = 1
            out_channel = config.char_conv_n_kernel
            self.conv_layer = L.Convolution2D(None, out_channel, k, s)
        self.dropout_rate = config.dropout_rate

    def __call__(self, h):
        h = h.transpose(0, 3, 1, 2) # NHWC -> NCHW
        h = F.dropout(h, self.dropout_rate)
        h = F.relu(self.conv_layer(h))
        h = F.max(h.transpose(0, 2, 3, 1), 2) # NCHW -> NHWC

        return h


class HighwayLayer(chainer.Chain):

    def __init__(self, in_out_size, nobias=False, activate=F.relu,
                 init_Wh=None, init_Wt=None, init_bh=None, init_bt=-1):
        super(HighwayLayer, self).__init__()
        self.activate = activate

        with self.init_scope():
            self.plain = L.Linear(
                None, in_out_size, nobias=nobias,
                initialW=init_Wh, initial_bias=init_bh)
            self.transform = L.Linear(
                None, in_out_size, nobias=nobias,
                initialW=init_Wt, initial_bias=init_bt)
            
    def __call__(self, x):
        out_plain = self.activate(self.plain(x))
        out_transform = F.sigmoid(self.transform(x))
        y = out_plain * out_transform + x * (1 - out_transform)
        return y
                
                
class HighwayNetwork(chainer.Chain):

    def __init__(self, config):
        super(HighwayNetwork, self).__init__()
        self.enc_dim = config.enc_dim

        with self.init_scope():
            self.highway_layer_1 = HighwayLayer(self.enc_dim)
            self.highway_layer_2 = HighwayLayer(self.enc_dim)
        
        if config.gpu[0] >= 0:
            self.highway_layer_1.to_gpu(config.gpu[0])
            self.highway_layer_2.to_gpu(config.gpu[0])

    def __call__(self, h):
        org_shape = h.shape
        h = h.reshape((-1, self.enc_dim))
        hx = self.highway_layer_1(h)
        hy = self.highway_layer_2(hx)
        hz = hy.reshape(list(org_shape[:-1]) + [self.enc_dim])

        return hz


class BiLSTM(L.NStepBiLSTM):

    def __init__(self, in_size, out_size, dropout_rate, config):
        with self.init_scope():
            super(BiLSTM, self).__init__(1, in_size, out_size, dropout_rate)
        if config.gpu[0] >= 0:
            self.to_gpu(config.gpu[0])

    def __call__(self, x, x_len, dropout=None):
        flat_x = x.reshape([-1] + list(x.shape[-2:]))
        xs = [xx[:int(xl.data),:] for xx, xl in zip(flat_x, x_len.reshape(-1))]
        if dropout is not None:
            org_dropout = self.dropout
            self.dropout = dropout
        hs, cs, ys = super(BiLSTM, self).__call__(None, None, xs)
        if dropout is not None:
            self.dropout = org_dropout
        ys = F.pad_sequence(ys, x.shape[-2], padding=-0.0)
        ys = ys.reshape(list(x.shape[:-1]) + [-1])

        return ys # [..., out_size * 2]


class AttentionFlow(chainer.Chain):

    def __init__(self, config):
        super(AttentionFlow, self).__init__()
        with self.init_scope():
            self.u_logit_layer = L.Linear(None, 1)

    def bi_attention(self, h, u, h_mask, u_mask):
        h_sent_num = h.shape[1]
        h_len = h.shape[2]
        u_len = u.shape[1]
        h_aug = F.tile(F.expand_dims(h, 3), (1, 1, 1, u_len, 1))
        u_aug = F.tile(F.expand_dims(F.expand_dims(u, 1), 1), (1, h_sent_num, h_len, 1, 1))

        if h_mask is None:
            hu_mask = None
        else:
            h_aug_mask = F.tile(F.expand_dims(h_mask, 3), (1, 1, 1, u_len))
            u_aug_mask = F.tile(F.expand_dims(F.expand_dims(u_mask, 1), 1), (1, h_sent_num, h_len, 1))

            hu_mask = h_aug_mask.data & u_aug_mask.data

        u_logits = get_logits(self.u_logit_layer, [h_aug, u_aug], hu_mask)
        u_a = softsel(u_aug, u_logits)
        h_a = softsel(h, F.max(u_logits, 3))
        h_a = F.tile(F.expand_dims(h_a, 2), (1, 1, h_len, 1))

        return u_a, h_a

    def __call__(self, h, u, h_mask, u_mask):
        u_a, h_a = self.bi_attention(h, u, h_mask, u_mask)
        p0 = F.concat((h, u_a, h * u_a, h * h_a), 3)
        return p0
        
        
class BiDAF(chainer.Chain):

    def __init__(self, config):
        super(BiDAF, self).__init__()
        with self.init_scope():
            self.word_emb = L.EmbedID(config.word_vocab_size, config.word_emb_dim,
                                      initialW=config.word_emb, ignore_label=-1)
            self.char_emb = L.EmbedID(config.char_vocab_size,
                                      config.char_emb_dim, ignore_label=-1)
            self.char_conv = CharacterConvolution(config)

            self.highway_network = HighwayNetwork(config)

            self.word_enc_dim = config.word_emb_dim + config.char_out_dim
            self.dropout_rate = config.dropout_rate

            self.context_bilstm = BiLSTM(self.word_enc_dim,
                                         config.hidden_size, self.dropout_rate, config) #in=200

            self.attention_layer = AttentionFlow(config)
            
            self.modeling_bilstm_g0 = BiLSTM(self.word_enc_dim * 4,
                                             config.hidden_size, self.dropout_rate, config) #in=800
            self.modeling_bilstm_g1 = BiLSTM(config.hidden_size * 2,
                                             config.hidden_size, self.dropout_rate, config) #in=200
            self.modeling_bilstm_g2 = BiLSTM(self.word_enc_dim * 7,
                                             config.hidden_size, self.dropout_rate, config) #in=1400

            self.y_logits_layer = L.Linear(None, 1)
            self.y2_logits_layer = L.Linear(None, 1)

        self.char_out_dim = config.char_out_dim
        self.skip_word_in_result = config.skip_word_in_result

        self.no_ema = config.no_ema
        if not self.no_ema:
            self.ema = ExponentialMovingAverage(config.decay_rate)
            self.ema_init = True

    def __call__(self, x, cx, x_mask, q, cq, q_mask, y, y2):
        # exponential moving average
        if not self.no_ema and not self.ema_init:
            self.ema(self)

        # embedding
        cx_emb = self.char_emb(cx)
        cq_emb = self.char_emb(cq)
        cx_emb = cx_emb.reshape([-1] + list(cx_emb.shape[2:]))

        xx = self.char_conv(cx_emb)
        qq = self.char_conv(cq_emb)

        xx = xx.reshape([-1, cx.shape[1], cx.shape[2], self.char_out_dim])
        qq = qq.reshape([-1, cq.shape[1], self.char_out_dim])

        x_emb = self.word_emb(x)
        q_emb = self.word_emb(q)

        xx = F.concat((x_emb, xx), 3)
        qq = F.concat((q_emb, qq), 2)
        
        xx = self.highway_network(xx)
        qq = self.highway_network(qq)

        # contextual
        x_len = F.sum(x_mask * 1.0, 2) # bool to int
        q_len = F.cast(F.sum(q_mask * 1.0, 1), 'f')

        h = self.context_bilstm(xx, x_len, 0.0)
        u = self.context_bilstm(qq, q_len)

        # attention flow
        p0 = self.attention_layer(h, u, h_mask=x_mask, u_mask=q_mask)

        # modeling and output
        g0 = self.modeling_bilstm_g0(p0, x_len)
        g1 = self.modeling_bilstm_g1(g0, x_len)

        logits = get_logits(self.y_logits_layer, [g1, p0], x_mask, 'linear', self.dropout_rate)
        g1s = g1.shape
        a1i = softsel(g1.reshape((g1s[0], g1s[1] * g1s[2], g1s[3])), logits.reshape((logits.shape[0], -1)))
        a1i = F.tile(F.expand_dims(F.expand_dims(a1i, 1), 1), (1, g1s[1], g1s[2], 1))

        g2 = self.modeling_bilstm_g2(F.concat((p0, g1, a1i, g1 * a1i), 3), x_len)
        logits2 = get_logits(self.y2_logits_layer, [g2, p0], x_mask, 'linear', self.dropout_rate)

        flat_logits = logits.reshape((-1, g1s[1] * g1s[2]))
        flat_yp = F.softmax(flat_logits)
        yp = flat_yp.reshape((-1, g1s[1], g1s[2]))
        flat_logits2 = logits2.reshape((-1, g1s[1] * g1s[2]))
        flat_yp2 = F.softmax(flat_logits2)
        yp2 = flat_yp2.reshape((-1, g1s[1], g1s[2]))

        # loss
        loss1 = F.softmax_cross_entropy(flat_logits, F.argmax(y.reshape((-1, g1s[1] * g1s[2])) * 1.0, axis=1), reduce='no')
        loss_mask = F.max(F.cast(q_mask * 1.0, 'f'), axis=1)
        loss1 = F.mean(loss_mask * loss1)
        loss2 = F.softmax_cross_entropy(flat_logits2, F.argmax(y2.reshape((-1, g1s[1] * g1s[2])) * 1.0, axis=1), reduce='no')
        loss2 = F.mean(loss_mask * loss2)
        loss = loss1 + loss2

        match, f1 = self.calc_result(x.reshape((x.shape[0], -1)),
                                     y.reshape((y.shape[0], -1)),
                                     y2.reshape((y2.shape[0], -1)),
                                     yp.reshape((yp.shape[0], -1)),
                                     yp2.reshape((yp2.shape[0], -1)))

        reporter.report({'loss': loss, 'match': match, 'f1': f1}, self)

        if not self.no_ema and self.ema_init:
            self.ema(self)
            self.ema_init = False
        
        return loss
        

    def calc_result(self, x, y, y2, yp, yp2):
        y_idx = F.argmax(y * 1.0, axis=1).data
        y2_idx = F.argmax(y2 * 1.0, axis=1).data
        yp_idx = F.argmax(yp, axis=1).data
        yp2_idx = F.argmax(yp2, axis=1).data
        match = np.mean([1 if yi == ypi and y2i == yp2i else 0
                        for yi, y2i, ypi, yp2i in zip(y_idx, y2_idx, yp_idx, yp2_idx)])

        f1 = []
        for idx, (yi, y2i, ypi, yp2i) in enumerate(zip(y_idx, y2_idx, yp_idx, yp2_idx)):
            y_words = [int(w) for w in x[idx][yi:y2i+1]]
            yp_words = [int(w) for w in x[idx][ypi:yp2i+1]]
            y_words = [w for w in y_words if w not in self.skip_word_in_result]
            yp_words = [w for w in yp_words if w not in self.skip_word_in_result]
            common = Counter(y_words) & Counter(yp_words)
            num_same = sum(common.values())
            if num_same == 0:
                f1i = 0
            else:
                precision = 1.0 * num_same / len(yp_words)
                recall = 1.0 * num_same / len(y_words)
                f1i = (2 * precision * recall) / (precision + recall)
            f1.append(f1i)
        return (match, np.mean(f1))

