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

from network.models.fireresidual_detection import FireResidualDetection
from image_datasets.datasources import CityPerson, CaltechPedestrian, CaltechPedestrianh5Py, BDD100KPedestrianImages
import os
from network.solver.lr_policy import get_gamma_exp_policy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="Dataset used for active learning. Default='caltech'", default='caltech', choices=['caltech', 'bdd'])
parser.add_argument('--budget', help="The budget of active learning at each cycle. Default=500", default=500, type=int)
parser.add_argument('--uncertainty_method', help="The method used for computing the uncertainty. Default: mutual_info", default="mutual_info", choices=['mutual_info', 'entropy', 'mcdropout'])
parser.add_argument('--uncertainty_aggregation', help="The method used for aggregating uncertainty scores. This could be only max_pool", default="max_pool", choices=['max_pool'])
parser.add_argument('--selection_method', help="Sample selection method. Default: top_best", default="top_best", choices=['top_best', 'tournament'])
parser.add_argument('--update_rule', help="Training method. Default: momentum", default="momentum", choices=['momentum', 'rms'])
parser.add_argument('--restore_from', help="Path to pretrained weights.", default='network/pretrained_weights/cityperson/cityperson', type=str)
parser.add_argument('--gpu', help="The index of GPU for running the code. Default: '0'", default='0', type=str)
parser.add_argument('--use_temporal_coherence', help="Weather or not to use temporal coherence. Default: True", default=True, type=bool)
parser.add_argument('--apply_selection_trick', help="Weather or not to apply selection rules. Default: True", default=True, type=bool)
parser.add_argument('--dropout_keep_prob', help="Dropout keep probability. Default=0.8", default=0.8, type=float)
parser = parser.parse_args()

s = 480./1024
input_size = (int(2048*s), int(1024.*s))

ds_train = CityPerson('train')
ds_train.resize_shape = input_size
ds_train.mean_image = 127.
ds_train.scale = 1/128.
ds_train.shuffle = True
ds_train.augment_data = False

if parser.dataset == 'caltech':
    # Uncomment/comment the following lines accordingly if you have not still created the h5 file.
    # ds_val = CaltechPedestrian(['set06', 'set07', 'set08', 'set09', 'set10'], step=10, skip_empty_frames=True, acceptable_labels=['person', 'people'])
    # ds_al = CaltechPedestrian(['set01', 'set02', 'set03', 'set04', 'set05'], step=2, acceptable_labels=['person', 'people'])

    ds_val = CaltechPedestrianh5Py('val.h5py')
    ds_al_unlabeled = CaltechPedestrianh5Py('al.h5py')
elif parser.dataset == 'bdd':
    ds_val = BDD100KPedestrianImages('val')
    ds_al_unlabeled = BDD100KPedestrianImages('train')
else:
    raise ValueError('Invalid dataset!')

ds_val.mean_image = 127.
ds_val.scale = 1/128.
ds_val.shuffle = False
ds_val.augment_data = False
ds_val.randomly_expland_bbs = False
ds_val.resize_along_x_prob = False

ds_al_unlabeled.mean_image = 127.
ds_al_unlabeled.scale = 1 / 128.
ds_al_unlabeled.shuffle = False
ds_al_unlabeled.augment_data = False
ds_al_unlabeled.randomly_expland_bbs = False
ds_al_unlabeled.resize_along_x_prob = False

ds_al_labeled = ds_al_unlabeled.clone()
ds_al_labeled.data_table = []
ds_al_labeled.augment_data = True
ds_al_labeled.shuffle = True
ds_al_labeled.shuffle_data()
ds_al_labeled.crop_shape = (640, 480) if isinstance(ds_al_unlabeled, BDD100KPedestrianImages) else None

use_temporal_coherence = True if not isinstance(ds_al_unlabeled, BDD100KPedestrianImages) else parser.use_temporal_coherence
apply_flipping = False
apply_selection_trick = True if not isinstance(ds_al_unlabeled, BDD100KPedestrianImages) else parser.apply_selection_trick

uncertainty_method = parser.uncertainty_method
uncertainty_aggregation = parser.uncertainty_aggregation
selection_method = parser.selection_method
update_rule = parser.update_rule

print 'uncertainty_method: ', uncertainty_method
print 'uncertainty_aggregation: ', uncertainty_aggregation
print 'selection_method: ', selection_method
print 'update_rule: ', update_rule
print 'use_temporal_coherence: ', use_temporal_coherence
print 'apply_flipping: ', apply_flipping
print 'apply_selection_trick:', apply_selection_trick
print raw_input('Press any key to continue ...')

net = FireResidualDetection(input_size=(480, 640) if isinstance(ds_al_labeled, CaltechPedestrianh5Py) or isinstance(ds_al_labeled, CaltechPedestrian) else (None, None),
                            batch_size=1,
                            uncertainty_method=uncertainty_method )
net.name = 'FireResidual_AL'
net.data_train = ds_train
net.data_test = ds_val
net.data_active_unlabeled = ds_al_unlabeled
net.data_active_labeled = ds_al_labeled

net.solver.base_lr = 0.01 if update_rule == 'momentum' else 0.001
net.solver.final_lr = 0.0001
net.solver.update_rule = update_rule
net.solver.gamma = get_gamma_exp_policy(net.solver.final_lr, net.solver.max_iter, net.solver.base_lr)

net.solver.test_interval = 50000
net.solver.snapshot_interval = 20000000
net.solver.max_iter = 12000000
net.solver.display_interval = 100

net.build_all()

neg2pos = 15
budget = parser.budget
mcdrop_keep_prob = parser.dropout_keep_prob
al_method = net.uncertainty_method
os.environ['CUDA_VISIBLE_DEVICES'] = parser.gpu

# #######################################################################################################
# We pick a descriptive name for the log directory. You may change it to your desired name.
# #######################################################################################################
log_dir = '{}/Desktop/{}{}_{}_{}_{}_{}b{}_{}{}{}{}/'.format(os.environ['HOME'],
                                                            ds_al_unlabeled.name if isinstance(ds_al_unlabeled,                                                                                                BDD100KPedestrianImages) else '',
                                                            net.name, al_method if al_method != 'mcdropout' else al_method + 'p{}'.format(mcdrop_keep_prob),
                                                            uncertainty_aggregation,
                                                            selection_method,
                                                            neg2pos,
                                                            budget,
                                                            net.solver.update_rule,
                                                            '_no_temporal' if not use_temporal_coherence else '',
                                                            '_no_flip' if not apply_flipping else '',
                                                            '_no_selection_trick' if not apply_selection_trick else '')

net.train(log_dir,
          al_method=al_method,
          init_at_each_cylce=None,
          uncertainty_aggregation=uncertainty_aggregation,
          selection_method=selection_method,
          restore_from=parser.restore_from,
          use_temporal_coherence=use_temporal_coherence,
          apply_flipping=apply_flipping,
          apply_selection_trick=apply_selection_trick,
          **{'cur_iter': 0, 'phase': 1, 'neg2pos': neg2pos, 'budget': budget})
exit()

