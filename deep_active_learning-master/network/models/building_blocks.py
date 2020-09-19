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
import tensorflow.contrib.slim as tf_slim


def dropout_spatial(incoming, keep_prob, layer_suffix='', is_training=True):
    if keep_prob == 1 or keep_prob is None:
        return incoming
    with tf.name_scope('Dropout' + layer_suffix):
        shp = incoming.shape.as_list()
        shp[1] = 1
        shp[2] = 1
        node = tf_slim.dropout(incoming, keep_prob, noise_shape=shp, is_training=is_training, scope='spatial_dropout')
    return node


def logits(incoming, num_classes, layer_suffix='', variables_collections=None):
    with tf.variable_scope('Logits' + layer_suffix, reuse=tf.AUTO_REUSE):
        with tf.name_scope('Logits' + layer_suffix):
            with tf_slim.arg_scope([tf_slim.layers.conv2d],
                                   padding='SAME',
                                   activation_fn=None,
                                   weights_regularizer=tf.nn.l2_loss,
                                   variables_collections=variables_collections,
                                   normalizer_fn=None):
                logit = tf_slim.layers.conv2d(incoming, num_classes, [1, 1], scope='logits')
    return logit


def downsample(incoming, num_filters, ksize_conv=3, ksize_pool=2, st=2, layer_suffix='', reuse=None, activation_fn=tf.nn.leaky_relu, variables_collections=None, keep_prob=1, is_training=True):
    with tf.variable_scope('Downsampling' + layer_suffix, reuse=reuse):
        with tf.name_scope('Downsampling' + layer_suffix):
            with tf_slim.arg_scope([tf_slim.conv2d],
                                   activation_fn=activation_fn,
                                   weights_regularizer=tf.nn.l2_loss,
                                   padding='SAME',
                                   variables_collections=variables_collections):
                c = tf_slim.layers.conv2d(incoming, num_filters, ksize_conv, st, scope='ds_conv' + layer_suffix)
                p = tf_slim.layers.max_pool2d(incoming, ksize_pool, stride=st, scope='ds_pool' + layer_suffix)
                cat = tf.concat([c, p], axis=-1, name='ds_concat' + layer_suffix)
    cat = dropout_spatial(cat, keep_prob, layer_suffix, is_training)
    return cat


def conv(incoming, num_filters, dilation=1, layer_suffix='', ksize=3, reuse=None, activation_fn=tf.nn.leaky_relu, variables_collections=None, keep_prob=1, is_training=True):
    with tf.variable_scope('Conv' + layer_suffix, reuse=reuse):
        with tf.name_scope('Conv' + layer_suffix):
            with tf_slim.arg_scope([tf_slim.layers.conv2d],
                                   padding='SAME',
                                   activation_fn=activation_fn,
                                   weights_regularizer=tf.nn.l2_loss,
                                   variables_collections=variables_collections):
                c = tf_slim.layers.conv2d(incoming, num_filters, ksize, rate=dilation, scope='conv' + layer_suffix)
    c = dropout_spatial(c, keep_prob, layer_suffix, is_training)
    return c


def fire_vertical(incoming, num_filters_sq, num_filters_ex, dilation=1, layer_suffix='', ksize=3, reuse=None, activation_fn=tf.nn.leaky_relu, variables_collections=None, keep_prob=1, is_training=True):
    with tf.variable_scope('Fire' + layer_suffix, reuse=reuse):
        with tf.name_scope('Fire' + layer_suffix):
            with tf_slim.arg_scope([tf_slim.layers.conv2d],
                                   padding='SAME',
                                   activation_fn=activation_fn,
                                   weights_regularizer=tf.nn.l2_loss,
                                   variables_collections=variables_collections):
                c_sq = tf_slim.layers.conv2d(incoming, num_filters_sq, [3, 1], scope='3x1_sq' + layer_suffix)
                c_ex3 = tf_slim.layers.conv2d(c_sq, num_filters_ex, ksize, rate=dilation, scope='{0}x{0}_ex'.format(ksize) + layer_suffix)
                c_ex1 = tf_slim.layers.conv2d(c_sq, num_filters_ex, [1, 1], scope='1x1_ex' + layer_suffix)
                cat = tf.concat([c_ex3, c_ex1], axis=-1, name='concat' + layer_suffix)
    cat = dropout_spatial(cat, keep_prob, layer_suffix, is_training)
    return cat


def fire_residual_vertical(incoming, num_filters_ex, num_filters_sq, dilation=1, layer_suffix='', ksize_fire=3, ksize_residual=0, reuse=None, activation_fn=tf.nn.leaky_relu, variables_collections=None, keep_prob=1, is_training=True):
    with tf.variable_scope('Fire_residual' + layer_suffix, reuse=reuse):
        with tf.name_scope('Fire_Residual' + layer_suffix):
            f = fire_vertical(incoming, num_filters_ex, num_filters_sq, dilation, layer_suffix, ksize_fire, reuse, activation_fn, variables_collections)
            if ksize_residual > 0:
                with tf_slim.arg_scope([tf_slim.layers.conv2d],
                                       padding='SAME',
                                       activation_fn=activation_fn,
                                       weights_regularizer=tf.nn.l2_loss,
                                       variables_collections=variables_collections):
                    c_fuse = tf_slim.layers.conv2d(incoming,
                                                   f.shape.as_list()[-1],
                                                   ksize_residual,
                                                   scope='{0}x{0}_fuse'.format(ksize_residual) + layer_suffix)
            else:
                c_fuse = incoming
            res = tf.add(c_fuse, f)
    res = dropout_spatial(res, keep_prob, layer_suffix, is_training)
    return res


def skip_connection(incoming, merge_with, num_output, ksize=3, layer_suffix='', activation_fn=tf.nn.relu, variables_collections=None, skip_type='add'):
    assert skip_type in ['add', 'concat']

    with tf.variable_scope('Skip' + layer_suffix):
        with tf.name_scope('Skip' + layer_suffix):
            with tf_slim.arg_scope([tf_slim.conv2d],
                                   activation_fn=activation_fn,
                                   weights_regularizer=tf.nn.l2_loss,
                                   padding='SAME',
                                   variables_collections=variables_collections):
                c = tf_slim.layers.conv2d(merge_with, num_output, ksize, scope='gate' + layer_suffix)
                m = tf.add(c, incoming)
    return m



