import numpy

from chainer import cuda
from chainer import optimizer

# from ema import UpdateRuleWithEMA

_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 1.0
_default_hyperparam.rho = 0.95
_default_hyperparam.eps = 1e-6


class AdaDeltaRuleWithLearningRate(optimizer.UpdateRule): #UpdateRuleWithEMA):

    def __init__(self, parent_hyperparam=None, lr=None, rho=None, eps=None):
        super(AdaDeltaRuleWithLearningRate, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if rho is not None:
            self.hyperparam.rho = rho
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['msg'] = xp.zeros_like(param.data)
            self.state['msdx'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        msg, msdx = self.state['msg'], self.state['msdx']
        lr = self.hyperparam.lr
        rho = self.hyperparam.rho
        eps = self.hyperparam.eps

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = numpy.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx * lr

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T one_minus_rho, T eps',
            'T param, T msg, T msdx',
            '''msg   = msg + one_minus_rho * (grad * grad - msg);
               T dx  = sqrt((msdx + eps) / (msg + eps)) * grad;
               msdx  += one_minus_rho * (dx * dx - msdx);
               param -= dx * lr;''',
            'adadelta')(grad, self.hyperparam.lr,
                        1 - self.hyperparam.rho,
                        self.hyperparam.eps, param.data,
                        self.state['msg'], self.state['msdx'])


class AdaDeltaWithLearningRate(optimizer.GradientMethod):

    """Zeiler's ADADELTA.

    See: http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

    Args:
        rho (float): Exponential decay rate of the first and second order
            moments.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 rho=_default_hyperparam.rho, eps=_default_hyperparam.eps):
        super(AdaDeltaWithLearningRate, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.rho = rho
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    rho = optimizer.HyperparameterProxy('rho')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return AdaDeltaRuleWithLearningRate(self.hyperparam)
