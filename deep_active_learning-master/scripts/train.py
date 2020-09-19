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

from network.models.fireresidual_detection import FireResidualDetection, get_gamma_exp_policy
from image_datasets.datasources import CityPerson, CaltechPedestrian, CaltechPedestrianh5Py, BDD100KPedestrianImages
import os
import argparse

s = 480. / 1024
input_size = (int(2048 * s), int(1024. * s))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="Dataset used for active learning. Default='caltech'", default='caltech', choices=['caltech', 'bdd', 'cityperson'])
parser.add_argument('--neg2pos', help="The negative to positive ratio. Default: 15", default=15, type=int)
parser.add_argument('--logdir', help="Directory to store log files", default='/home/{}/Desktop/'.format(os.environ['USER']))
parser.add_argument('--gpu', help="The index of GPU for running the code. Default: '0'", default='0', type=str)
parser.add_argument('--restore_from', help="Path to pretrained weights.", default='network/pretrained_weights/cityperson/cityperson', type=str)
parser.add_argument('--current_iter', help="Current iteration. This is used to continue training form a checkpoint", default=0, type=int)
parser = parser.parse_args()

if parser.dataset == 'caltech':
    ds_train = CaltechPedestrianh5Py('train.h5py')
    ds_val = CaltechPedestrianh5Py('val.h5py')
elif parser.dataset == 'bdd':
    ds_train = BDD100KPedestrianImages('train', skip_empty_frames=True)
    ds_val = BDD100KPedestrianImages('val', skip_empty_frames=True)
elif parser.dataset == 'cityperson':
    ds_train = CityPerson('train')
    ds_val = CityPerson('val')
    ds_val.resize_shape = input_size
    ds_train.resize_shape = input_size
else:
    raise ValueError('Invalid dataset!')

ds_train.mean_image = 127.
ds_train.scale = 1/128.
ds_train.shuffle = True
# ########################################################################################
# Results might be slightly improved by setting augment_data to True
# ########################################################################################
ds_train.augment_data = False
# ########################################################################################
ds_train.crop_shape = (640, 480) if isinstance(ds_train, BDD100KPedestrianImages) else None

ds_val.mean_image = 127.
ds_val.scale = 1 / 128.
ds_val.shuffle = False
ds_val.augment_data = False

os.environ['CUDA_VISIBLE_DEVICES'] = parser.gpu

net = FireResidualDetection(input_size=(480, 640) if isinstance(ds_train, CaltechPedestrianh5Py) or isinstance(ds_train, CaltechPedestrian) else (None, None))
net.solver.max_iter = 600000
net.solver.final_lr = 0.0001
net.solver.test_interval = 50000
net.solver.update_rule = 'momentum'
net.solver.base_lr = 0.01 if net.solver.update_rule == 'momentum' else 0.001
net.solver.gamma = get_gamma_exp_policy(net.solver.final_lr, net.solver.max_iter, net.solver.base_lr)
net.build_all()
net.data_train = ds_train
net.data_test = ds_val

neg2pos = parser.neg2pos
net.name = 'FireResidual_{}_{}_n2p{}'.format(parser.dataset, net.solver.update_rule, neg2pos)

net.train(os.path.join(parser.logdir, net.name + '/'),
          restore_from=parser.restore_from,
          **{'cur_iter': parser.current_iter, 'phase': 0, 'neg2pos': neg2pos})
