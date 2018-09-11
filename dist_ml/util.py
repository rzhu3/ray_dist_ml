from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def proxgrad_l1l2(var, grad, lr, l1, l2):
    # Try to perform proximal gradient step:
    # x_{t+1} = prox_{lr * h}(var - lr * grad)
    # where h(x) = l1 * ||x||_1 + 0.5 * l2 * ||x||_F^2.
    prox_var = var - grad * lr
    if l1 > 0:
        prox_var = np.sign(prox_var) * np.maximum(np.abs(prox_var) - lr*l1, 0.0)
    prox_var /= 1.0 + l2 * lr
    return prox_var


def prox_posl2ball(x, l2):
    tmp = np.maximum(x, 0.0)
    norm = np.linalg.norm(x)
    if norm <= l2:
        return tmp
    else:
        return tmp * l2 / norm


def proxgrad_posl2ball(var, grad, lr, l2=1.0):
    return prox_posl2ball(var - lr * grad, l2)