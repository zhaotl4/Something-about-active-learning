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
from network.metrics.metrics import *
import os
import argparse

s = 480. / 1024
input_size = (int(2048 * s), int(1024. * s))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="Dataset used for active learning. Default='caltech'", default='caltech', choices=['caltech', 'bdd', 'cityperson'])
parser.add_argument('--uncertainty_method', help="The method used for computing the uncertainty. Default: mutual_info", default="mutual_info", choices=['mutual_info', 'entropy', 'mcdropout'])
parser.add_argument('--gpu', help="The index of GPU for running the code. Default: '0'", default='0', type=str)
parser = parser.parse_args()

if parser.dataset == 'caltech':
    # ds_val = CaltechPedestrian(['set06', 'set07', 'set08', 'set09', 'set10'], step=10, skip_empty_frames=True, acceptable_labels=['person', 'people'])
    ds_val = CaltechPedestrianh5Py('val.h5py')
elif parser.dataset == 'bdd':
    ds_val = BDD100KPedestrianImages('val', skip_empty_frames=True)
elif parser.dataset == 'cityperson':
    ds_val = CityPerson('val')
    ds_val.resize_shape = input_size
else:
    raise ValueError('Invalid dataset')

ds_val.mean_image = 127.
ds_val.scale = 1 / 128.
ds_val.shuffle = False
ds_val.augment_data = False
ds_val.randomly_expland_bbs = False
ds_val.resize_along_x_prob = False

os.environ['CUDA_VISIBLE_DEVICES'] = parser.gpu

if isinstance(ds_val, CaltechPedestrianh5Py) or isinstance(ds_val, CaltechPedestrian):
    input_size = (480, 640)
elif isinstance(ds_val, BDD100KPedestrianImages):
    input_size = (None, None)
elif isinstance(ds_val, CityPerson):
    input_size = (int(1024. * s), int(2048 * s))

net = FireResidualDetection(input_size=input_size, uncertainty_method=parser.uncertainty_method)
net.build_network()


def get_ckpt_name(folder, fname):
    """
    Change the path in this function to the directory where checkpoints are stored.
    """
    return '/home/{}/Desktop/{}/{}'.format(os.environ['USER'], folder, fname)


# ###########################################################################################
# Fill the following list using your desired checkpoints. You may use get_ckpt_name function
# for this purpose.
# ###########################################################################################
ckpt_list = [
   # get_ckpt_name('FireResidual_caltech_momentum_n2p15', '-600000')
]
os.environ['CUDA_VISIBLE_DEVICES'] = parser.gpu


for ckpt in ckpt_list:
    list_th = []
    list_pre = []
    list_rec = []
    list_miss = []
    list_fppi = []

    print '*' * 40
    print ckpt
    print '*' * 40
    fname_results = ckpt + '_res'
    fname_fppi_miss = ckpt + '_fppi_miss.png'
    print 'Evaluating with {}'.format(ckpt)
    print 'Results will be saved to ', fname_results
    np.set_printoptions(precision=2)
    for i in range(0, 101, 10):
        print 'Evaluation using threshold {}'.format(i / 100.)
        if i == 0:
            i = 1
        if i == 100:
            i = 99
        cm, list_unc = net.evaluate(dataset=ds_val, ckpt_file=ckpt, detection_threshold=i / 100., apply_flipping=False, save_video=False, save_images=False)
        print 'Validation error on {}'.format(ds_val.__class__.__name__)
        miss, fppi, pre, rec = compute_miss_rate_fppi(cm,  'average')
        print '-- Global: precision={}, recall={}, fppi={}, miss rate={}'.format(pre, rec, fppi, miss)

        list_th.append(i / 100.)
        list_pre.append(pre)
        list_rec.append(rec)
        list_fppi.append(fppi)
        list_miss.append(miss)

    list_rec = np.asarray(list_rec)
    list_pre = np.asarray(list_pre)
    list_fppi = np.asarray(list_fppi)
    list_miss = np.asarray(list_miss)

    np.savez(fname_results, list_rec=list_rec, list_pre=list_pre, list_fppi=list_fppi, list_miss=list_miss)
