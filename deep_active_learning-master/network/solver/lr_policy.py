"""
This code is implemented as a part of the following paper and it is only meant to reproduce the results of the paper:
    "Active Learning for Deep Detection Neural Networks,
    "Hamed H. Aghdam, Abel Gonzalez-Garcia, Joost van de Weijer, Antonio M. Lopez", ICCV 2019
_____________________________________________________

Developer/Maintainer:  Hamed H. Aghdam
Year:                  2018-2019
License:               BSD
_____________________________________________________

"""

import numpy as np


def exp_weight_decay(base_lr, gamma, cur_iter):
    """
    This implementation is similar to the implementation in the Caffe library
    :param base_lr:
    :param gamma:
    :param cur_iter:
    :return:
    """
    decays = base_lr * np.power(gamma, cur_iter)
    return decays


def step_weight_decay(base_lr, gamma, step_size, cur_iter):
    decays = np.power(gamma, np.floor_divide(cur_iter, step_size))*base_lr
    return decays


def get_gamma_exp_policy(desired_lr, at_iteration, base_lr):
    return np.power(desired_lr/base_lr, 1/float(at_iteration))


def get_gamma_step_policy(desired_lr, at_iteration, based_lr, step_size):
    return np.power(desired_lr/based_lr, step_size/float(at_iteration))


def get_stepsize_step_policy(desired_lr, at_iteration, based_lr, gamma):
    return int(at_iteration*np.log(gamma)/np.log(desired_lr/based_lr))

