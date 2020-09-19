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

import tensorflow as tf


class SolverParameter:
    """
    This class stores information about the training procedure.
    """
    def __init__(self):
        self.update_rule = 'rms'
        self.weight_decay = 0.00001
        self.momentum = 0.9
        self.base_lr = 0.02
        self.final_lr = 0.0002
        self.tf_lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.lr_policy = 'exp'
        self.gamma = 0.98
        self.power = 0.98
        self.step_size = 1000
        self.snapshot_interval = -1
        self.snapshot_prefix = ''
        self.display_interval = 200
        self.loss_average_hist = 200
        self.max_iter = 600000
        self.test_interval = 5e15
        self.test_iter = 1
        self.regularization_type = 'l2'

    def __str__(self):
        s = 'Solver Parameters:\n'
        for att in sorted(self.__dict__.keys()):
            if not callable(att):
                s += '- {}: {}\n'.format(att, self.__dict__[att])
        return s
