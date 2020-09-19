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

from model_base import ModelBase
from building_blocks import *
from network.solver.lr_policy import get_gamma_exp_policy
from extended_graphkey import GraphKeysExtended as GK
from network.metrics.metrics import *
import scipy.ndimage as sp_im


class FireResidualDetection(ModelBase):
    """
    Implements a FPN style network
    """
    def __init__(self, input_size=(480, 960), num_classes=5, batch_size=1, uncertainty_method='mutual_info', mcdropout_kwargs=None):
        super(FireResidualDetection, self).__init__(input_size, num_classes, batch_size)
        if mcdropout_kwargs is None:
            mcdropout_kwargs = {'mcdrop_keep_prob': 0.9}

        if 'mcdrop_keep_prob' in mcdropout_kwargs:
            self._mcdrop_keep_prob = mcdropout_kwargs['mcdrop_keep_prob']
        self._solver.base_lr = 0.001
        self._solver.final_lr = 0.0001
        self._solver.max_iter = 1200000
        self._solver.gamma = get_gamma_exp_policy(self._solver.final_lr, self._solver.max_iter, self._solver.base_lr)
        self._solver.display_interval = 50
        self._solver.test_interval = 10000
        self._solver.test_iter = 2000000
        self._solver.max_iter_phase1 = 0
        self._solver.max_iter_phase2 = 0
        self._solver.snapshot_interval = 20000
        self._solver.weight_decay = 2e-6
        self._name = 'Pedestrian_FireResidual'
        self._uncertainty_method = uncertainty_method

    '''
    我们可以使用@property装饰器来创建只读属性
    @property装饰器会将方法转换为相同名称的只读属性,可以与所定义的属性配合使用，这样可以防止属性被修改。
    '''
    @property
    def uncertainty_method(self):
        return self._uncertainty_method

    def build_train(self, trainer=None):
        if self._solver.update_rule == 'rms':
            trainer = tf.train.RMSPropOptimizer(self._solver.tf_lr, 0.9)
        elif self._solver.update_rule == 'momentum':
            trainer = tf.train.MomentumOptimizer(self._solver.tf_lr, 0.9)

        super(FireResidualDetection, self).build_train(trainer)

    def build_network(self):
        with tf_slim.arg_scope([tf_slim.conv2d], normalizer_fn=tf_slim.batch_norm, normalizer_params={'is_training': self._tf_is_training}):
            kwargs = dict(activation_fn=tf.nn.elu,
                          variables_collections=[GK.ENCODER_VARIABLE],
                          is_training=self._tf_is_training,
                          keep_prob=1)

            keep_prob = self._tf_keep_prob
            kwargs['keep_prob'] = 1
            node = self._add_graph_node('ds1', downsample(self._tf_input, 61, 3, layer_suffix='_1', **kwargs))
            node = self._add_graph_node('f11', fire_residual_vertical(node, 32, 32, layer_suffix='_11', **kwargs))
            node = self._add_graph_node('f12', fire_residual_vertical(node, 32, 32, layer_suffix='_12', **kwargs))
            kwargs['keep_prob'] = 1
            node = self._add_graph_node('ds2', downsample(node, 64, 3, layer_suffix='_2', **kwargs))
            node = self._add_graph_node('f21', fire_residual_vertical(node, 64, 64, layer_suffix='_21', **kwargs))
            node = self._add_graph_node('f22', fire_residual_vertical(node, 64, 64, layer_suffix='_22', **kwargs))   # 55x31
            node = self._add_graph_node('f23', fire_residual_vertical(node, 64, 64, layer_suffix='_23', **kwargs))   # 71x39
            kwargs['keep_prob'] = (9 + keep_prob) / 10
            node = self._add_graph_node('ds3', downsample(node, 128, 3, layer_suffix='_3', **kwargs))                # 79x47
            node = self._add_graph_node('f31', fire_residual_vertical(node, 128, 128, layer_suffix='_31', **kwargs)) # 111x63
            node = self._add_graph_node('f32', fire_residual_vertical(node, 128, 128, layer_suffix='_32', **kwargs)) # 143x79
            node = self._add_graph_node('f33', fire_residual_vertical(node, 128, 128, layer_suffix='_33', **kwargs)) # 175x95
            kwargs['keep_prob'] = (4 + keep_prob) / 5

            kwargs['dilation'] = 2
            node = self._add_graph_node('f41', fire_residual_vertical(node, 128, 128, layer_suffix='_41', **kwargs)) # 223x127
            node = self._add_graph_node('f42', fire_residual_vertical(node, 128, 128, layer_suffix='_42', **kwargs)) # 271x159

            kwargs['keep_prob'] = keep_prob

            node_dec1 = self._add_graph_node('dec1', conv(node, 128, ksize=1, layer_suffix='_dec1', **kwargs))

            node_dec2 = self._add_graph_node('dec2', conv(node_dec1, 128, layer_suffix='dec2'))
            node_dec2 = skip_connection(node_dec2, self._graph_nodes['f41'], 128)

            node_dec3 = self._add_graph_node('dec3', conv(node_dec2, 128, layer_suffix='dec3'))
            node_dec3 = skip_connection(node_dec3, self._graph_nodes['f32'], 128, layer_suffix='_dec3')

            node_dec4 = self._add_graph_node('dec4', conv(node_dec3, 128, layer_suffix='dec4'))
            node_dec4 = skip_connection(node_dec4, self._graph_nodes['ds3'], 128, layer_suffix='_dec4')

            shp = self._graph_nodes['f22'].shape.as_list() if self._input_size[0] is not None else tf.shape(self._graph_nodes['f22'])
            node_dec5 = self._add_graph_node('up5', tf.image.resize_bilinear(node_dec4, [shp[1], shp[2]]))
            node_dec5 = self._add_graph_node('dec5', conv(node_dec5, 128, layer_suffix='dec5'))
            node_dec5 = skip_connection(node_dec5, self._graph_nodes['f22'], 128, layer_suffix='_dec5')

            kwargs['variables_collections'] = [GK.DECODER_VARIABLE]
            del kwargs['dilation']
            del kwargs['is_training']
            del kwargs['keep_prob']

            shp = self._tf_input.shape.as_list() if self._input_size[0] is not None else tf.shape(self._tf_input)
            logits1 = self._add_graph_node('logits1', logits(node_dec1, 1, variables_collections=[GK.DECODER_VARIABLE]))
            logits2 = self._add_graph_node('logits2', logits(node_dec2, 1, variables_collections=[GK.DECODER_VARIABLE]))
            logits3 = self._add_graph_node('logits3', logits(node_dec3, 1, variables_collections=[GK.DECODER_VARIABLE]))
            logits4 = self._add_graph_node('logits4', logits(node_dec4, 1, variables_collections=[GK.DECODER_VARIABLE]))
            logits5 = self._add_graph_node('logits5', logits(node_dec5, 1, variables_collections=[GK.DECODER_VARIABLE]))

            self._logits = [logits1, logits2, logits3, logits4, logits5]

            # ##########################################################
            # Resizing followed by sigmoid
            # ##########################################################
            with tf.name_scope('Contact_Logits'):
                list_temp = []
                with tf.name_scope('Resize_Prob'):
                    for lg in self._logits:
                        lg_resize = tf.image.resize_bilinear(lg, shp[1:3])
                        list_temp.append(lg_resize)
                logits_resized = tf.concat(list_temp, axis=-1)
                prob = tf.nn.sigmoid(logits_resized)

                self._add_graph_node('prob', prob)

            # ##########################################################
            # Prediction
            # ##########################################################
            with tf.name_scope('Prediction'):
                with tf.name_scope('smooth_probs'):
                    prob_smooth = tf.nn.avg_pool(prob, (1, 3, 3, 1), [1] * 4, 'SAME')
                    self._add_graph_node('probs_smooth', prob_smooth)

                if self._uncertainty_method == 'entropy':
                    # # ##########################################################
                    # # Entropy (spatial mean, smoothed)
                    # # ##########################################################
                    # with tf.name_scope('Uncertainty_Entropy'):
                    #     logits_resized_unc = tf.where(tf.less(logits_resized, tf.zeros_like(logits_resized)), logits_resized / 2., logits_resized)
                    #     prob = tf.nn.sigmoid(logits_resized_unc)
                    #     p = tf.nn.avg_pool(prob, (1, 9, 9, 1), [1] * 4, 'SAME')
                    #     p_m1 = 1-p
                    #     ent1 = -(p * tf.log(p+1e-12) + (p_m1) * tf.log(p_m1+1e-12))
                    #     ent1 = tf.reduce_sum(ent1, axis=-1)
                    #     self._add_graph_node('uncertainty', ent1)
                    # ##########################################################
                    # Entropy
                    # ##########################################################
                    with tf.name_scope('Uncertainty_Entropy'):
                        p = prob
                        p_m1 = 1 - p
                        ent1 = -(p * tf.log(p + 1e-12) + (p_m1) * tf.log(p_m1 + 1e-12))
                        ent1 = tf.reduce_sum(ent1, axis=-1)
                        self._add_graph_node('uncertainty', ent1)
                elif self._uncertainty_method in ['mutual_info', 'random']:
                    # ##########################################################
                    # Mutual Information
                    # ##########################################################
                    with tf.name_scope('Uncertainty_MI'):
                        prob_pixel = prob
                        prob_pixel_m1 = 1 - prob_pixel
                        ent_pixel = -(prob_pixel * tf.log(prob_pixel + 1e-12) + prob_pixel_m1 * tf.log(prob_pixel_m1 + 1e-12))
                        ent_pixel = tf.nn.avg_pool(ent_pixel, (1, 9, 9, 1), [1] * 4, 'SAME')
                        mean_of_entropy = tf.reduce_sum(ent_pixel, axis=-1)

                        prob_local = tf.nn.avg_pool(prob_pixel, (1, 9, 9, 1), [1] * 4, 'SAME')
                        prob_local_m1 = 1 - prob_local
                        entropy_of_mean = -(prob_local * tf.log(prob_local + 1e-12) + prob_local_m1 * tf.log(prob_local_m1 + 1e-12))
                        entropy_of_mean = tf.reduce_sum(entropy_of_mean, axis=-1)

                        self._add_graph_node('uncertainty', entropy_of_mean - mean_of_entropy)
                elif self._uncertainty_method == 'mcdropout':
                    # ##########################################################
                    # Generating T predictions using MCDropout
                    # ##########################################################
                    T = 15
                    mcdrop_keep_prob = self._mcdrop_keep_prob

                    mc_drop_probs = []
                    for i in xrange(T):
                        with tf.name_scope('MCDrop_Pred_T{}'.format(i)):
                            drp = dropout_spatial(node_dec1, mcdrop_keep_prob, '_mcdrop_1_t{}'.format(i), True)
                            logits1 = self._add_graph_node('logits_mcdrop_1_t{}'.format(i), logits(drp, 1, variables_collections=[GK.DECODER_VARIABLE]))

                            drp = dropout_spatial(node_dec2, mcdrop_keep_prob, '_mcdrop_2_t{}'.format(i), True)
                            logits2 = self._add_graph_node('logits_mcdrop_2_t{}'.format(i), logits(drp, 1, variables_collections=[GK.DECODER_VARIABLE]))

                            drp = dropout_spatial(node_dec3, mcdrop_keep_prob, '_mcdrop_3_t{}'.format(i), True)
                            logits3 = self._add_graph_node('logits_mcdrop_3_t{}'.format(i), logits(drp, 1, variables_collections=[GK.DECODER_VARIABLE]))

                            drp = dropout_spatial(node_dec4, mcdrop_keep_prob, '_mcdrop_4_t{}'.format(i), True)
                            logits4 = self._add_graph_node('logits_mcdrop_4_t{}'.format(i), logits(drp, 1, variables_collections=[GK.DECODER_VARIABLE]))

                            drp = dropout_spatial(node_dec5, mcdrop_keep_prob, '_mcdrop_5_t{}'.format(i), True)
                            logits5 = self._add_graph_node('logits_mcdrop_5_t{}'.format(i), logits(drp, 1, variables_collections=[GK.DECODER_VARIABLE]))

                            mc_logits = [logits1, logits2, logits3, logits4, logits5]

                            # ##########################################################
                            # Resizing followed by sigmoid
                            # ##########################################################
                            with tf.name_scope('Contact_Logits_mcdrop_t{}'.format(i)):
                                list_temp = []
                                with tf.name_scope('Resize_Prob_mcdrop_t{}'.format(i)):
                                    for lg in mc_logits:
                                        lg_resize = tf.image.resize_bilinear(lg, shp[1:3])
                                        list_temp.append(lg_resize)
                                logits_resized = tf.concat(list_temp, axis=-1)
                                prob = tf.nn.sigmoid(logits_resized)
                                # prob = tf.Print(prob, [tf.reduce_mean(prob), 'Mean Prob T{}:'.format(i)])
                                # prob = tf.Print(prob, [prob[0, 239, 319, :], 'Prob [0,239, 319, :] as T{}:'.format(i)], first_n=10)
                                mc_drop_probs.append(prob)

                    # ##########################################################
                    # Mutual Information
                    # ##########################################################
                    with tf.name_scope('Uncertainty_MI'):
                        mean_of_entropy_list = []
                        with tf.name_scope('Mean_of_entropy'):
                            for prob in mc_drop_probs:
                                prob_pixel = prob
                                prob_pixel_m1 = 1 - prob_pixel
                                ent_pixel = -(prob_pixel * tf.log(prob_pixel + 1e-12) + prob_pixel_m1 * tf.log(prob_pixel_m1 + 1e-12))
                                mean_of_entropy_list.append(ent_pixel)

                            mean_of_entropy = tf.add_n(mean_of_entropy_list)/len(mean_of_entropy_list)
                            mean_of_entropy = tf.reduce_sum(mean_of_entropy, axis=-1)

                        with tf.name_scope('Entropy_of_mean'):
                            prob_local = tf.add_n(mc_drop_probs)/len(mean_of_entropy_list)
                            prob_local_m1 = 1 - prob_local
                            entropy_of_mean = -(prob_local * tf.log(prob_local + 1e-12) + prob_local_m1 * tf.log(prob_local_m1 + 1e-12))
                            entropy_of_mean = tf.reduce_sum(entropy_of_mean, axis=-1)

                        self._add_graph_node('uncertainty', entropy_of_mean - mean_of_entropy)

                with tf.name_scope('non_max_supress'):
                    sz = 19
                    ones = tf.ones_like(prob_smooth)
                    zeros = tf.zeros_like(prob_smooth)

                    prob_smooth_locally_max = tf.nn.max_pool(prob_smooth,
                                                             (1, sz, sz, 1),
                                                             (1, 1, 1, 1),
                                                             padding='SAME',
                                                             name='spatial_maximum')
                    local_max = tf.where(tf.less_equal(prob_smooth_locally_max - prob_smooth, 1e-16), ones, zeros)

                with tf.name_scope('binary_prediction'):
                    pred_binary = tf.where(
                        tf.greater_equal(prob_smooth,
                                         tf.fill(tf.shape(prob_smooth),
                                                 self._tf_detection_prob_threshold)),
                        ones,
                        zeros)

                with tf.name_scope('num_of_local_predictions'):
                    shp = tf.shape(pred_binary)
                    sz = 19
                    num_detections = tf.nn.depthwise_conv2d(pred_binary,
                                                            tf.ones((sz, sz, shp[-1], 1)),
                                                            (1, 1, 1, 1),
                                                            padding='SAME',
                                                            name='number_of_detection')
                    min_detection = tf.fill(tf.shape(prob_smooth), self._tf_detection_min_detected_bbs)
                    min_detection_supress = tf.where(tf.greater_equal(num_detections, min_detection), ones, zeros)

                with tf.name_scope('detection_map'):
                    detection_map = tf.multiply(tf.multiply(local_max, min_detection_supress), pred_binary)

        self._add_graph_node('detection_map', detection_map)
        self._add_graph_node('detection_map1', tf.multiply(local_max, pred_binary))
        self._add_graph_node('pred_binary', pred_binary)
        self._add_graph_node('num_detections_', num_detections)
        self._add_graph_node('prob_smooth_locally_max_', prob_smooth_locally_max)

    def evaluate(self, sess=None, dataset=None, ckpt_file=None, max_batches=None, save_images=False, save_video=False,
                         name_prefix='', detection_threshold=0.5, scales=(1.,), apply_flipping=False, uncertainty_aggregation_method='max_pool'):
            import cv2
            from utils.tic_toc import TicToc
            import numpy as np
            ttc = TicToc()

            assert self._batch_size == 1

            dataset.shuffle = False
            tf_detection_map = self._graph_nodes['detection_map']
            tf_prob_map = self._graph_nodes['prob']

            if sess is None:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                tf.train.Saver().restore(sess, ckpt_file)

            if dataset is None:
                dataset = self._ds_test.clone()

            dataset.seek_ind(0)

            if max_batches is None:
                max_batches = len(dataset.data_table) // self._batch_size

            max_iter = min(self._solver.test_iter, max_batches)
            conf_mat = np.zeros((2, 2, max_iter*self._batch_size), dtype=np.float32)

            list_num_detected = []
            list_unc = [None] * max_iter

            max_unc = None
            min_unc = None
            for i in xrange(max_iter):
                cur_ind = dataset.cur_ind
                x_batch, y_score, y_reg, x_path, x_bbs = dataset.read_batch(self._batch_size)
                # y_score = np.where(1-y_score[..., -1] > 0, 1., 0.)
                if i == 0:
                    print 'Shape of validation image: {}'.format(x_batch.shape)
                ttc.tic()
                list_detected_bbs = [list() for _ in range(self._batch_size)]
                for s in scales:
                    sw = int(x_batch.shape[2] * s)
                    sh = int(x_batch.shape[1] * s)

                    x_batch_resized = np.zeros_like(x_batch)
                    for ii in xrange(x_batch.shape[0]):
                        x_batch_resized[ii, :sh, :sw, ...] = cv2.resize(x_batch[ii, ...], (sw, sh))
                    detection_map, prob_map, unc = sess.run([tf_detection_map, tf_prob_map, self._graph_nodes['uncertainty']],
                                             {self._tf_input: x_batch_resized,
                                              self._tf_detection_prob_threshold: detection_threshold,
                                              self._tf_detection_min_detected_bbs: 10 * 10,
                                              self._tf_keep_prob: 1.0,
                                              self._tf_is_training: True
                                              })

                    if apply_flipping:
                        detection_map_flip, prob_map_flip, unc_flip = sess.run([tf_detection_map, tf_prob_map, self._graph_nodes['uncertainty']],
                                                      {self._tf_input: x_batch_resized[:, ::, -1::-1, :],
                                                       self._tf_detection_prob_threshold: detection_threshold,
                                                       self._tf_detection_min_detected_bbs: 10 * 10,
                                                       self._tf_keep_prob: 1.0,
                                                       self._tf_is_training: True
                                                       })
                        detection_map_flip = detection_map_flip[..., -1::-1, :]
                        unc_flip = unc_flip[..., -1::-1, :]
                        unc = (unc + unc_flip) / 2.0
                        detection_map = np.clip(detection_map + detection_map_flip, 0, 1)

                    if max_unc is None:
                        max_unc = unc.max()
                    if min_unc is None:
                        min_unc = unc.min()

                    if uncertainty_aggregation_method == 'max_pool':
                        list_unc[cur_ind:cur_ind + self._batch_size] = np.mean(sp_im.maximum_filter(unc, size=(1, 30, 30))[..., ::30, ::30], axis=(1, 2))
                    elif uncertainty_aggregation_method == 'sum':
                        list_unc[cur_ind:cur_ind+self._batch_size] = np.sum(unc, axis=(1, 2))
                    elif uncertainty_aggregation_method == 'mean':
                        list_unc[cur_ind:cur_ind+self._batch_size] = np.mean(unc, axis=(1, 2))
                    elif uncertainty_aggregation_method == 'percentile':
                        list_unc[cur_ind:cur_ind+self._batch_size] = np.percentile(unc, 90, axis=(1, 2))
                    elif uncertainty_aggregation_method == 'top_mean':
                        __ii = int(640*480*0.9)
                        list_unc[cur_ind:cur_ind+self._batch_size] = np.mean(np.sort(np.reshape(unc, [self._batch_size, -1]), axis=-1)[:, -__ii:], axis=-1)
                    else:
                        raise ValueError()

                    detection_map[:, sh:, sw:, :] = 0
                    assert detection_map.shape[-1] == self.num_classes
                    list_num_detected.append(detection_map.sum())
                    ttc.lap('net')

                    for ii in xrange(detection_map.shape[0]):
                        ttc.lap('argmax')
                        y, x, c = np.where(detection_map[ii] == 1)
                        if self._reverse_default_bbs:
                            c = detection_map.shape[-1]-1 - c
                        ttc.toc('where')
                        for _i in xrange(x.shape[0]):
                            d_bb_h_half = int(dataset.default_bbs[c[_i]][1] / (2 * s))
                            d_bb_w_half = int(dataset.default_bbs[c[_i]][0] / (2 * s))
                            k = x_batch_resized.shape[1] / float(detection_map.shape[1])
                            px = int(x[_i] * k / s)
                            py = int(y[_i] * k / s)
                            d_bb = [(px - d_bb_w_half),
                                    (py - d_bb_h_half),
                                    (px + d_bb_w_half),
                                    (py + d_bb_h_half)]
                            list_detected_bbs[ii].append(d_bb)

                for ii in xrange(self._batch_size):
                    iou_mat = boundingbox_iou_matrix(x_bbs[ii], list_detected_bbs[ii])
                    conf_mat[..., cur_ind+ii] = iou_matrix_to_confusion_matrix(iou_mat, 0.3)
                if i % 100 == 0:
                    print '{}: {}'.format(i, ttc.time_elapsed(True))
            return conf_mat, list_unc


if __name__ == '__main__':
    net = FireResidualDetection()
    net.build_network()

