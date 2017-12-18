import numpy
import copy

import chainer
from chainer import cuda
from chainer import optimizer

class ExponentialMovingAverage(object):

    def __init__(self, decay_rate=0.999):
        self.decay = decay_rate
        self.avg_dict = {}
        self.org_dict = {}
        self.is_test = False
        self.init = False

    def __call__(self, target, device=None):
        if device:
            import cupy as xp
            xp.cuda.Device(device).use()
        if chainer.config.train:
            if self.is_test:
                self.reset(target)
                self.is_test = False
            for name, param in target.namedparams():
                if not param.update_rule.enabled: continue
                p = param.data
                n = name
                if not n in self.avg_dict.keys() or self.avg_dict[n] is None:
                    self.avg_dict[n] = p
                else:
                    avg_p = self.avg_dict[n]
                    self.avg_dict[n] = self.decay * avg_p + (1 - self.decay) * p
        elif not self.is_test:
            self.is_test = True
            self.average(target)

    def average(self, target):
        for name, param in target.namedparams():
            if not param.update_rule.enabled: continue
            self.org_dict[name] = param.data
            param.data = self.avg_dict[name]

    def reset(self, target):
        for name, param in target.namedparams():
            if not param.update_rule.enabled: continue
            param.data = self.org_dict[name]

