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

from network.solver.solverparam import SolverParameter
from image_datasets.datasources import DetectionSource
import tensorflow as tf
from network.solver.lr_policy import exp_weight_decay, step_weight_decay
from utils.logger import Logger
from image_datasets.datareader import DataReader
import numpy as np
from os import path, makedirs
from datetime import datetime
from network.metrics.metrics import compute_pre_recal
from network.solver.lr_policy import get_gamma_exp_policy
import cPickle
import os


class TrainingPhase:
    def __init__(self):
        self.train_on_source_dataset = 0
        self.evaluate_active_set = 1
        self.train_on_source_and_active_set = 2


class ModelBase (object):
    def __init__(self, input_size=(2048, 2048), num_classes=5, batch_size=1):
        if batch_size != 1:
            print '''Current code is tested for batch size 1. The loss function and the evaluation function might need
            to be changed for larger batch sizes.'''
            raise ValueError('Current code is tested for batch size 1.')
        self._solver = SolverParameter()
        self._ds_train = None
        self._ds_test = None
        self._ds_active_unlabeled = None
        self._ds_active_labeled = None
        self._graph_nodes = {}
        self._input_size = input_size
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._tf_input = tf.placeholder(tf.float32,
                                        (self._batch_size, self._input_size[0], self._input_size[1], 3),
                                        name='input')
        self._tf_label = tf.placeholder(tf.float32,
                                        (self._batch_size, self._input_size[0], self._input_size[1],
                                         self._num_classes),
                                        name='actual_labels')
        self._tf_sample_weight = tf.placeholder(tf.float32,
                                                (self._batch_size, self._input_size[0], self._input_size[1], self._num_classes),
                                                name='weight')
        self._tf_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
        self._tf_detection_prob_threshold = tf.placeholder_with_default(0.5, ())
        self._tf_detection_min_detected_bbs = tf.placeholder_with_default(1.0, ())

        self._tf_is_training = tf.placeholder_with_default(True, (), name='is_training')
        self._tf_global_step = tf.placeholder(tf.int32, shape=(), name='global_step')

        self._tf_last_precision_average = tf.placeholder('float32', ())
        self._tf_last_recall_average = tf.placeholder('float32', ())
        self._tf_last_precision_global = tf.placeholder('float32', ())
        self._tf_last_recall_global = tf.placeholder('float32', ())

        self._last_precision_average = 0
        self._last_recall_average = 0
        self._last_precision_global = 0
        self._last_recall_global = 0
        # #######################################################################################################
        # It is better to set _reverse_default_bbs to True. However, you can reproduce our results by setting it
        # to False
        # #######################################################################################################
        self._reverse_default_bbs = False

        self._logits = None
        self._name = 'network-base'

    def __getitem__(self, key_or_ind):
        if isinstance(key_or_ind, basestring):
            return self._graph_nodes[key_or_ind]

    def _add_graph_node(self, key, value):
        if key in self._graph_nodes:
            raise ValueError('{} is already in the list of nodes.'.format(key))

        self._graph_nodes[key] = value
        return value

    @staticmethod
    def compute_metrics(conf_mat):
        tp = np.diag(conf_mat)
        class_iou = tp / (np.sum(conf_mat, axis=1) + np.sum(conf_mat, axis=0) - tp + 1e-10)
        acc = tp.sum() / (conf_mat.sum() + 1e-10)
        return class_iou, acc

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def graph_nodes(self):
        return self._graph_nodes

    @property
    def input_tensor(self):
        return self._tf_input

    @property
    def solver(self):
        return self._solver

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, s):
        self._name = s

    def __get_ds_train(self):
        return self._ds_train

    def __set_ds_train(self, ds):
        if not isinstance(ds, DetectionSource) and ds is not None:
            raise ValueError("ds must be an instance of DetectionSource")
        self._ds_train = ds
    data_train = property(__get_ds_train, __set_ds_train)

    def __get_ds_test(self):
        return self._ds_test

    def __set_ds_test(self, ds):
        if not isinstance(ds, DetectionSource) and ds is not None:
            raise ValueError("ds must be an instance of DetectionSource")
        self._ds_test = ds

    @property
    def data_active_unlabeled(self):
        return self._ds_active_unlabeled

    @data_active_unlabeled.setter
    def data_active_unlabeled(self, value):
        self._ds_active_unlabeled = value

    @property
    def data_active_labeled(self):
        return self._ds_active_labeled

    @data_active_labeled.setter
    def data_active_labeled(self, value):
        self._ds_active_labeled = value

    data_test = property(__get_ds_test, __set_ds_test)

    def build_network(self):
        raise NotImplementedError()

    def build_summary(self, add_image_summaries=False):
        with tf.name_scope('summaries'):
            if add_image_summaries:
                tf.summary.image('input', self._tf_input, max_outputs=2)
                tf.summary.image('detection_map', tf.reduce_max(self._graph_nodes['detection_map'], axis=-1, keep_dims=True))
                tf.summary.image('detection_map1', tf.reduce_max(self._graph_nodes['detection_map1'], axis=-1, keep_dims=True))
                tf.summary.image('prediction_binary', tf.reduce_max(self._graph_nodes['pred_binary'], axis=-1, keep_dims=True))

                tf.summary.image('Prob012', self._graph_nodes['prob'][..., 0:3], max_outputs=2)
                tf.summary.image('Prob234', self._graph_nodes['prob'][..., 2:6], max_outputs=2)

                tf.summary.image('Act012', self._tf_label[..., 0:3], max_outputs=2)
                tf.summary.image('Act234', self._tf_label[..., 2:6], max_outputs=2)
                tf.summary.image('Act_All', tf.reduce_sum(self._tf_label, axis=-1, keep_dims=True), max_outputs=2)

                tf.summary.image('Weight_mask', tf.reduce_sum(self._tf_sample_weight, axis=-1, keep_dims=True), max_outputs=2)

            tf.summary.scalar('loss_total', self._graph_nodes['loss_total'])
            tf.summary.scalar('loss_cross', self._graph_nodes['loss_cross'])
            tf.summary.scalar('loss_reg', self._graph_nodes['loss_reg'])

            tf.summary.scalar('last_precision_average', self._tf_last_precision_average)
            tf.summary.scalar('last_recall_average', self._tf_last_recall_average)
            tf.summary.scalar('last_precision_global', self._tf_last_precision_global)
            tf.summary.scalar('last_recall_global', self._tf_last_recall_global)

            tf.summary.scalar('learning_rate', self._solver.tf_lr)

            self._add_graph_node('summary_whole', tf.summary.merge_all())

    def build_train(self, trainer=None):
        """
        We DO NOT apply UPDATE_OPS for batch normalization layers INTENTIONALLY! For this reason, we always set
        is_training to True. This way, each layer is always normalized using its own weighted mean and variance.
        Applying UPDATE_OPS is unlikely to affect the performance though.
        Thus, you may collect UPDATE_OPS here, group them with the training op and save it to the dictionary of
        endpoints with the key 'train_op_whole'.
        """
        loss = self._graph_nodes['loss_total']
        if trainer is None:
            trainer = tf.train.AdamOptimizer(self._solver.tf_lr, 0.9)
        self._add_graph_node('train_op_whole', trainer.minimize(loss))

    def build_metrics(self):
        list_conf_mat = []
        list_update_op = []
        list_init_conft_mat = []

        for i, lg in enumerate(self._logits):
            var = tf.get_variable('conf_mat{}'.format(i), (2, 2), tf.float32, tf.zeros_initializer, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            w = self._graph_nodes['tf_label_weights{}'.format(i)]
            w = tf.reshape(tf.ceil(w), [-1])
            
            cont_mat = tf.confusion_matrix(tf.reshape(self._graph_nodes['tf_label_resize{}'.format(i)], [-1]),
                                           tf.reshape(tf.where(tf.greater(lg, 0), tf.ones_like(lg), tf.zeros_like(lg)), [-1]),
                                           2, name='', dtype='float32', weights=w)

            op = tf.assign_add(var, cont_mat, name='update_confmat{}'.format(i))
            op_init = tf.variables_initializer([var], 'mean_iou_init{}'.format(i))
            list_conf_mat.append(var)
            list_init_conft_mat.append(op_init)
            list_update_op.append(op)
            self._add_graph_node('conf_mat{}'.format(i), var)

        conf_mat_op = tf.group(*list_update_op)
        cont_mat_init = tf.group(*list_init_conft_mat)

        self._add_graph_node('conf_mat_op', conf_mat_op)
        self._add_graph_node('reset_metrics', cont_mat_init)

    def build_loss(self):
        dic_loss = {}
        for i, lg in enumerate(self._logits):
            with tf.name_scope('cross_entropy{}'.format(i)):
                shp = tf.shape(lg)
                tf_label_resize = tf.image.resize_nearest_neighbor(self._tf_label[..., i:i+1], shp[1:3])
                self._add_graph_node('tf_label_resize{}'.format(i), tf_label_resize)

                # #####################################################################################################
                # Here, we first normalize the weights and then downsample it. This basically reduces the weight of
                # the cross entropy loss function since some of samples are discarded during downsampling.
                # To compensate that, we set the N2P to a high value.
                # Alternatively, you can first downsample the weights and then normalize them. This way, you may need
                # a smaller N2P. To that, comment the two following lines and uncomment the next 3rd and 4th lines.
                # #####################################################################################################
                tf_sample_weigth_reisze = self._tf_sample_weight[..., i:i + 1] / (tf.reduce_sum(self._tf_sample_weight[..., i:i + 1]) + 1e-9)
                tf_sample_weigth_reisze = tf.image.resize_nearest_neighbor(tf_sample_weigth_reisze, shp[1:3])

                # #####################################################################################################
                # The two following lines must be uncommented if you comment the above two lines
                # #####################################################################################################
                # tf_sample_weigth_reisze = tf.image.resize_nearest_neighbor(self._tf_sample_weight, shp[1:3])
                # tf_sample_weigth_reisze = tf_sample_weigth_reisze[..., i:i + 1] / (tf.reduce_sum(tf_sample_weigth_reisze[..., i:i + 1]) + 1e-9)

                self._add_graph_node('tf_label_weights{}'.format(i), tf_sample_weigth_reisze)

                loss_cross = tf.losses.sigmoid_cross_entropy(tf_label_resize,
                                                             lg,
                                                             tf_sample_weigth_reisze,
                                                             reduction=tf.losses.Reduction.SUM)
                dic_loss[lg] = loss_cross

        with tf.name_scope('regularization'):
            loss_reg = self.solver.weight_decay * tf.add_n([w for w in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
        with tf.name_scope('total_cross_entropy'):
            loss_cross = tf.add_n(dic_loss.values())
        loss = loss_cross + loss_reg
        self._add_graph_node('loss_cross', loss_cross)
        self._add_graph_node('loss_reg', loss_reg)
        self._add_graph_node('loss_total', loss)

    def build_all(self):
        self.build_network()
        self.build_loss()
        self.build_train()
        self.build_metrics()
        self.build_summary()

    def train(self, log_dir=None, restore_from=None, al_method='entropy', selection_method='top_best', uncertainty_aggregation='max_pool', init_at_each_cylce=True, use_temporal_coherence=True, apply_flipping=True, apply_selection_trick=True, **kwargs):

        dr_train = DataReader(self._ds_train, self._batch_size, 3)
        dr_val = DataReader(self._ds_test, self._batch_size, 10)
        dr_al = DataReader(self._ds_active_labeled, self._batch_size, 10)
        if self._ds_active_unlabeled is not None and self._ds_active_labeled is None:
            self._ds_active_labeled = self._ds_active_unlabeled.clone()
            self._ds_active_labeled.data_table = []
            self._ds_active_labeled.augment_data = True
            self._ds_active_labeled.shuffle = True
            self._ds_active_labeled.shuffle_data()

        dr_train.start()
        dr_val.start()

        if self._ds_train.augment_data:
            print 'List of augmentation params:'
            for k, v in self._ds_train.aug_dic:
                print k, v

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        if log_dir is None:
            log = Logger(None)
            fw = None
            save = None
        else:
            if not path.exists(log_dir):
                makedirs(log_dir)
            log = Logger(path.join(log_dir, 'log{}.txt'.format(datetime.now())))
            fw = tf.summary.FileWriter(log_dir, sess.graph)
            save = tf.train.Saver(max_to_keep=30)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print log << 'Training batch shape: {}'.format(self._ds_train.get_output_shape())
        print log << 'Test batch shape: {}'.format(self._ds_test.get_output_shape())
        print log << 'AL labeled batch shape: {}'.format(self._ds_active_labeled.get_output_shape() if self._ds_active_labeled is not None else None)
        print log << 'AL unlabeled batch shape: {}'.format(self._ds_active_unlabeled.get_output_shape() if self._ds_active_unlabeled is not None else None)

        print log << 'cuda visible devices: {}'.format(os.environ['CUDA_VISIBLE_DEVICES'])
        print log << 'uncertainty_aggregation: {}'.format(uncertainty_aggregation)
        print log << 'al method: {}'.format(al_method)
        print log << 'selection method: {}'.format(selection_method)
        print log << 'use temporal coherence: {}'.format(use_temporal_coherence)
        print log << 'apply selection  trick: {}'.format(apply_selection_trick)
        print log << 'reverse default boxes: {}'.format(self._reverse_default_bbs)
        for k, v in kwargs.items():
            print log << '{}: {}'.format(k, v)

        print log << 'Solver summary:'
        print log << self._solver
        print log << '*'*40
        print log << 'Training started....'

        lr = self._solver.base_lr
        cur_iter = 0
        train_phase = TrainingPhase()
        phase = train_phase.train_on_source_dataset
        phase_iter = 0

        def restore_weights(restore_flags=True):
            print log << 'Restoring weights from {}'.format(restore_from)
            var_list = [v for v in tf.global_variables() if all([v.name.find('Momentum') < 0, v.name.find('RMS') < 0])]

            rest = tf.train.Saver(var_list=var_list)
            if isinstance(restore_from, basestring):
                filename = restore_from
            elif isinstance(restore_from, list):
                filename = restore_from[0]
                if restore_flags:
                    with open(restore_from[1], 'rb') as file_id:
                        flags, list_selected_inds = cPickle.load(file_id)
                        self._ds_active_unlabeled.flags = flags
                        data_table = self._ds_active_labeled.data_table
                        ped_count = 0
                        count = 0
                        for _ii in xrange(len(flags)):
                            if flags[_ii]:
                                data_table.append(self._ds_active_unlabeled.data_table[_ii])
                                ped_count += len(data_table[-1][1])
                            count += 1
                        self._ds_active_labeled.data_table = data_table

            rest.restore(sess, filename)

        if restore_from is not None and restore_from != '':
            restore_weights()
            cur_iter = kwargs['cur_iter']
            phase = kwargs['phase'] if 'phase' in kwargs else 0
            phase_iter = kwargs['phase_iter'] if 'phase_iter' in kwargs else 0

        run_meta_data_flag = True
        budget = 500 if 'budget' not in kwargs else kwargs['budget']
        al_iter = 0
        last_al_iter = 0
        neg_to_pos_ratio = 10 if 'neg2pos' not in kwargs else kwargs['neg2pos']
        print log << 'neg_to_pos_ratio: {}'.format(neg_to_pos_ratio)
        print log << 'budget: {}'.format(budget)
        print log << 'network TF input size: {}'.format(self._tf_input.shape.as_list())
        # ##########################################################
        # Training/Active Learning LOOP
        # ##########################################################
        print 'Training started from phase {}'.format(phase)
        while al_iter < 40:
            lr_mult = 1
            if phase == train_phase.train_on_source_dataset:
                x_batch, y_batch, y_reg, path_batch, x_bbs = dr_train.dequeue()
                if phase_iter == 0:
                    print log << 'Shape of training image: {}'.format(x_batch.shape)

                if phase_iter >= self._solver.max_iter:
                    phase = train_phase.evaluate_active_set
                    print 'Changed the phase to {}'.format(phase)
                    phase_iter = 0
                else:
                    phase_iter += 1
            elif phase == train_phase.evaluate_active_set:
                if al_method == 'entropy' or al_method == 'mutual_info' or al_method == 'mcdropout':
                    conf_mat, list_unc = self.evaluate(sess=sess, dataset=self._ds_active_unlabeled, save_video=False, apply_flipping=apply_flipping, uncertainty_aggregation_method=uncertainty_aggregation)
                    with open(path.join(fw.get_logdir(), 'uncertainty_{}_{}.pkl'.format(uncertainty_aggregation, cur_iter)), 'wb') as file_id:
                        cPickle.dump(list_unc, file_id)
                    list_unc = np.asarray(list_unc)
                    if use_temporal_coherence:
                        filt = np.exp(-np.power(np.linspace(-5, 5, 11, dtype='float32'), 2)/(2*3**2))
                        filt = filt / filt.sum()
                        list_unc_smooth = np.convolve(list_unc, filt, 'same')
                        with open(path.join(fw.get_logdir(), 'uncertainty_smoothed_{}_{}.pkl'.format(uncertainty_aggregation, cur_iter)), 'wb') as file_id:
                            cPickle.dump(list_unc_smooth, file_id)
                        list_unc = list_unc_smooth

                elif al_method == 'random':
                    list_unc = np.random.uniform(0, 1, (len(self._ds_active_unlabeled), ))
                else:
                    raise ValueError('Invalid al_method!')

                if selection_method == 'top_best':
                    inds_sorted = np.argsort(list_unc)[::-1]
                elif selection_method == 'tournament':
                    inds_sorted = np.zeros((list_unc.shape[0]), dtype='int32')
                    for _ii in xrange(inds_sorted.shape[0]):
                        tour_inds = np.random.randint(0, inds_sorted.shape[0], 5)
                        max_ind = np.argmax(list_unc[tour_inds])
                        inds_sorted[_ii] = tour_inds[max_ind]
                else:
                    raise ValueError('Invalid selection method')

                if hasattr(self._ds_active_unlabeled, 'flags'):
                    flags = getattr(self._ds_active_unlabeled, 'flags')
                else:
                    flags = np.zeros((list_unc.shape[0], ))

                data_table = self._ds_active_labeled.data_table
                count = 0
                ped_count = 0
                list_selected_inds = []
                for _tt in xrange(3):
                    flags_selected = np.zeros((list_unc.shape[0],))
                    _ii = 0
                    while count < budget and _ii < list_unc.shape[0]:
                        if flags[inds_sorted[_ii]] == 1 or (flags_selected[inds_sorted[_ii]] == 1 and _tt <= 1):
                            _ii += 1
                            continue
                        list_selected_inds.append(_ii)

                        flags[inds_sorted[_ii]] = True
                        if apply_selection_trick:
                            flags_selected[inds_sorted[_ii] - 15:inds_sorted[_ii] + 15] = 1
                            flags[inds_sorted[_ii] - 2:inds_sorted[_ii] + 3] = 1
                        else:
                            flags_selected[inds_sorted[_ii]] = 1
                            flags[inds_sorted[_ii]] = 1

                        assert len(flags) == list_unc.shape[0]
                        data_table.append(self._ds_active_unlabeled.data_table[inds_sorted[_ii]])
                        ped_count += len(data_table[-1][1])
                        count += 1
                self._ds_active_unlabeled.flags = flags
                self._ds_active_labeled.data_table = data_table
                print log << 'Current: There are {} instances of pedestrian in the selected {} frames'.format(ped_count, count)
                print log << 'Total: There are {} instances of pedestrian in the {} labeld frames'.format(sum([len(r[1]) for r in self._ds_active_labeled.data_table]), len(self._ds_active_labeled))
                print log << 'Total: There are {} selectable frames left'.format(flags.shape[0]-flags.sum())
                print log << '{} out of {} frames contain pedestrian'.format(sum([min(1, len(r[1])) for r in self._ds_active_labeled.data_table]), len(self._ds_active_labeled))

                dr_al.ds = self._ds_active_labeled
                if not dr_al.is_alive():
                    dr_al.start()

                with open(path.join(fw.get_logdir(), 'al_state_{}.pkl'.format(cur_iter)), 'wb') as file_id:
                    cPickle.dump([flags, list_selected_inds], file_id)

                phase = train_phase.train_on_source_and_active_set
                print log << 'Changed the phase to {}'.format(phase)

                self._solver.max_iter = len(self._ds_active_labeled) * 50 # 250000
                print log << 'The network will be trained for {} iterations in the next cycle'.format(self._solver.max_iter)

                if al_iter * budget < 5000 or True:
                    print log << 'Initializing using the pretrained network'
                    init_at_each_cylce = True
                else:
                    print log << 'Continue training from the current weights'
                    init_at_each_cylce = False

                if init_at_each_cylce:
                    print log << 'Initializing the network...'
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    restore_weights(restore_flags=False)
                else:
                    print log << 'Continue training from the current weights'

                al_iter += 1

                if init_at_each_cylce:
                    last_al_iter = cur_iter
                    self._solver.gamma = get_gamma_exp_policy(self._solver.final_lr, self._solver.max_iter, self._solver.base_lr)
                    self._solver.display_interval = 100
                    self._solver.test_interval = 50000

                print log << 'Training the network for {} iterations..'.format(self._solver.max_iter)
                continue
            elif phase == train_phase.train_on_source_and_active_set:
                x_batch, y_batch, y_reg, path_batch, x_bbs = dr_al.dequeue()
                if phase_iter == 0:
                    print log << 'Shape of training image: {}'.format(x_batch.shape)
                lr_mult = 1

                fname = '/tmp/change_phase.txt'
                if path.exists(fname):
                    with open(fname, 'r') as _f:
                        flag = _f.read(1) == '1'
                    del _f
                else:
                    with open(fname, 'w') as _f:
                        _f.write('0')
                    flag = False

                if phase_iter > self._solver.max_iter or flag:
                    save.save(sess, fw.get_logdir(), cur_iter)
                    phase = train_phase.evaluate_active_set
                    print 'Changed the phase to {}'.format(phase)
                    phase_iter = 0
                    cm = self.__evaluate(log, sess)
                    del cm

                    log << '{}: Changing the state to evaluation.'.format(cur_iter)
                    save.save(sess, fw.get_logdir(), cur_iter)
                else:
                    phase_iter += 1

            assert y_batch.shape[-1] == self._num_classes + 1
            y_score = y_batch[..., :self.num_classes]
            if self._reverse_default_bbs:
                y_score = y_score[..., ::-1]

            valid_anchors = 1-y_score
            valid_anchors[:, 50:-50, 50:-50, ...] = 1
            p_binomial = max(100., y_score.sum()) / y_score.size * neg_to_pos_ratio
            if p_binomial > 1:
                print y_score.sum(), y_score.size, neg_to_pos_ratio
                print path_batch
                print x_bbs

            drop_mask = np.random.binomial(1, p_binomial, y_score.shape) * valid_anchors
            # ##########################################################################################
            # Uncomment this to reproduce our results. Comment it to improve the results.
            # ##########################################################################################
            np.random.seed(y_score.sum().astype('int64'))
            # ##########################################################################################
            weight_mask = np.clip(drop_mask + y_score, 0, 1)

            if cur_iter % 50000 == 0 or cur_iter == 10:
                run_meta_data_flag = True

            fetches = [self._graph_nodes['train_op_whole'],
                       self._graph_nodes['loss_cross'],
                       self._graph_nodes['loss_reg'],
                       self._graph_nodes['loss_total'],
                       self._graph_nodes['conf_mat1'],
                       self._graph_nodes['conf_mat2'],
                       self._graph_nodes['conf_mat3'],
                       self._graph_nodes['conf_mat4'],
                       self._graph_nodes['conf_mat0'],
                       self._graph_nodes['conf_mat_op'],
                       self._graph_nodes['logits1'],
                       self._graph_nodes['logits2'],
                       self._graph_nodes['logits3'],
                       self._graph_nodes['logits4'],
                       self._graph_nodes['logits5'],
                       ]

            if cur_iter % self._solver.display_interval == 0:
                fetches.append(self._graph_nodes['summary_whole'])

                if run_meta_data_flag:
                    options = tf.RunOptions()
                    options.output_partition_graphs = True
                    options.trace_level = tf.RunOptions.FULL_TRACE
                    metadata = tf.RunMetadata()
                    kwargs = {'run_metadata': metadata, 'options': options}
                else:
                    kwargs = {}

                _, loss_cross, loss_reg, loss_tot, cm1, cm2, cm3, cm4, cm0, _, lg1, lg2, lg3, lg4, lg5, summary = sess.run(
                    fetches, {
                        self._tf_input: x_batch,
                        self._tf_label: y_score,
                        self._tf_sample_weight: weight_mask,
                        self._solver.tf_lr: lr * lr_mult,
                        self._tf_keep_prob: 0.8,
                        self._tf_last_recall_global: self._last_recall_global,
                        self._tf_last_precision_global: self._last_precision_global,
                        self._tf_last_precision_average: self._last_precision_average,
                        self._tf_last_recall_average: self._last_recall_average,
                        self._tf_global_step: cur_iter}, **kwargs)
                class_iou1, acc1 = self.compute_metrics(cm1)
                class_iou2, acc2 = self.compute_metrics(cm2)
                class_iou3, acc3 = self.compute_metrics(cm3)
                class_iou4, acc4 = self.compute_metrics(cm4)
                class_iou0, acc0 = self.compute_metrics(cm0)

                print log << 'Iter:{0}, Cross:{1:.5f}, Reg:{2:.5f}, Tot:{3:.3f}, lr:{4:.5f}, Acc0:{5:.3f}, Acc1:{6:.3f}, Acc2:{7:.3f}, Acc3:{8:.3f}, ACC4:{9:.3f}'.format(
                    cur_iter,
                    loss_cross,
                    loss_reg,
                    loss_tot,
                    lr,
                    acc0,
                    acc1,
                    acc2,
                    acc3,
                    acc4)
                np.set_printoptions(precision=3, linewidth=120)
                fw.add_summary(summary, cur_iter)
                if run_meta_data_flag:
                    fw.add_run_metadata(metadata, 'step {}'.format(cur_iter), cur_iter)
                    run_meta_data_flag = False
                fw.flush()

            else:
                _, loss_cross, loss_reg, loss_tot, cm1, cm2, cm3, cm4, cm0, _, lg1, lg2, lg3, lg4, lg5 = sess.run(
                    fetches,
                    {self._tf_input: x_batch,
                     self._tf_label: y_score,
                     self._tf_sample_weight: weight_mask,
                     self._solver.tf_lr: lr,
                     self._tf_keep_prob: 0.8,
                     self._tf_global_step: cur_iter})

            if cur_iter % self._solver.test_interval == 0 and cur_iter > 0:
                cm = self.__evaluate(log, sess)
                del cm

            if cur_iter % self._solver.snapshot_interval == 0 and cur_iter > 0 and save is not None:
                save.save(sess, fw.get_logdir(), cur_iter)

            lr = self.anneal_lr(max(0, cur_iter - last_al_iter))

            cur_iter += 1

    def __evaluate(self, log, sess):
        cm, list_unc = self.evaluate(sess, self._ds_test, save_images=False, apply_flipping=False)
        print log << 'Validation error on {}:'.format(self._ds_test.name)
        pre, rec = compute_pre_recal(cm, 'global')
        self._last_precision_global = pre
        self._last_recall_global = rec

        print log << '-- Global: pre={}, rec={}'.format(pre, rec)
        pre, rec = compute_pre_recal(cm, 'average')
        self._last_precision_average = pre
        self._last_recall_average = rec
        print log << '-- Average: pre={}, rec={}'.format(pre, rec)
        return cm

    def anneal_lr(self, current_iter):
        if self._solver.lr_policy == 'exp':
            lr = exp_weight_decay(self._solver.base_lr,
                                  self._solver.gamma,
                                  current_iter)
        elif self._solver.lr_policy == 'step':
            lr = step_weight_decay(self._solver.base_lr,
                                   self._solver.gamma,
                                   self._solver.step_size,
                                   current_iter)
        else:
            raise ValueError('Incorrect lr_policy')

        lr = max(self._solver.final_lr, lr)

        return lr

    def evaluate(self, sess=None, dataset=None, ckpt_file=None, max_batches=None, save_images=False, save_video=False,
                         name_prefix='', detection_threshold=0.5, scales=(1.,), apply_flipping=False, uncertainty_aggregation_method='max_pool'):
        raise NotImplementedError()

