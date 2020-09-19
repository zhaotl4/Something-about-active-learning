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


class GraphKeysExtended (tf.GraphKeys):
    ENCODER_VARIABLE = 'encoder_varialbe'
    DECODER_VARIABLE = 'decoder_variable'
    LOGITS_VARIABLE = 'logits_variable'