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

from image_datasets.datasources import CaltechPedestrian, DS_ROOT_DIR
c = CaltechPedestrian(['set01', 'set02', 'set03', 'set04', 'set05'], step=1, skip_empty_frames=True, acceptable_labels=['person', 'people'])
c.write_to_h5(DS_ROOT_DIR + '/Caltech_Pedestrian/train.h5py')

c = CaltechPedestrian(['set06', 'set07', 'set08', 'set09', 'set10'], step=10, skip_empty_frames=True, acceptable_labels=['person', 'people'])
c.write_to_h5(DS_ROOT_DIR + '/Caltech_Pedestrian/val.h5py')

c = CaltechPedestrian(['set01', 'set02', 'set03', 'set04', 'set05'], step=2, skip_empty_frames=False, acceptable_labels=['person', 'people'])
c.write_to_h5(DS_ROOT_DIR + '/Caltech_Pedestrian/al.h5py')



