
import chainer
import chainer.functions as F
from chainer import cuda

xp = cuda.cupy

def exp_mask(val, mask):
    xp = cuda.get_array_module(val)
    return val + (1 - xp.asarray(mask, xp.float32)) * -1e30
    
def softmax(logits, mask=None):
    if mask is not None:
        logits = exp_mask(logits, mask)
    flat_logits = flatten(logits)
    flat_out = F.softmax(flat_logits)
    out = flat_out.reshape(list(logits.shape[:-1]) + [-1])
    return out

def softsel(target, logits, mask=None, scope=None):
    a = softmax(logits, mask=mask)
    target_rank = len(target.shape)
    out = F.sum(F.tile(F.expand_dims(a, -1), target.shape[-1]) * target, target_rank - 2)
    return out

def flatten(args):
    if type(args) is not list:
        args = [args]
    flat_arg = F.concat([arg.reshape((-1,arg.shape[-1])) for arg in args], axis=1)
    return flat_arg

def flat_linear(layer, args, squeeze=False, drop_rate=0.0):
    flat_arg = flatten(args)
    flat_arg = F.dropout(flat_arg, drop_rate)
    flat_out = layer(flat_arg)
    if type(args) is not list:
        reconst_shape = list(args.shape[:-1])
    else:
        reconst_shape = list(args[0].shape[:-1])
    out = flat_out.reshape(reconst_shape + [-1])
    if squeeze:
        out = F.squeeze(out, len(out.shape)-1)
    return out

def linear_logits(layer, args, mask=None, drop_rate=0.0):
    logits = flat_linear(layer, args, squeeze=True, drop_rate=drop_rate)
    if mask is not None:
        logits = exp_mask(logits, mask)
    return logits

def get_logits(layer, args, mask=None, func='tri_linear', drop_rate=0.0):
    if func == 'tri_linear':
        new_arg = args[0] * args[1]
        args.append(new_arg)
    return linear_logits(layer, args, mask=mask, drop_rate=drop_rate)

