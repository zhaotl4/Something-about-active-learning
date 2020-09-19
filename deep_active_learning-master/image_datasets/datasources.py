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

from base import DetectionSource
import cv2
import os
from base import EncodedImageString, H5PyRecord
import h5py
import numpy as np

DS_ROOT_DIR = '/home/{}/Desktop/'.format(os.environ['USER'])
if DS_ROOT_DIR[-1] != '/':
    DS_ROOT_DIR += '/'


class CityPerson(DetectionSource):
    """"
    The annotations are downloadable from:
    https://bitbucket.org/shanshanzhang/citypersons/src/f44d4e585d51d0c3fd7992c8fb913515b26d4b5a/annotations/?at=default
    """
    def __init__(self, dataset_type='train'):
        assert dataset_type in ['train', 'val']
        super(CityPerson, self).__init__()
        self.name = 'CityPerson'
        import scipy.io as sp
        bbs = sp.loadmat(os.path.join(DS_ROOT_DIR, 'CityScape/anno_{}.mat'.format(dataset_type)))
        tbl = bbs['anno_{}_aligned'.format(dataset_type)]
        data_table = []
        for i in xrange(tbl.shape[1]):
            annot = tbl[0, i][0, 0]
            folder_name = annot[0][0]
            if folder_name == 'tuebingen':
                folder_name = 'tubingen'
            file_name = annot[1][0]
            bb_info = annot[2]
            list_bbs = []
            for ii in xrange(bb_info.shape[0]):
                bb = bb_info[ii]
                # Row format: [class_label, x1,y1,w,h, instance_id, x1_vis, y1_vis, w_vis, h_vis]
                cls = bb[0]

                if cls in [1, 2]:
                    x1, y1, w, h = bb[1:5]
                # elif cls in [3, 4]:
                #     x1, y1, w, h = bb[6:]
                else:
                    continue
                x2 = x1 + w
                y2 = y1 + h

                s = 480/1024.
                if w * s > 24 or h * s > 45:
                    bb = [x1, y1, x2, y2]
                    list_bbs.append(bb)

            if len(list_bbs) > 0:
                data_table.append([os.path.join(DS_ROOT_DIR, 'CityScape/leftImg8bit/{}/{}/{}'.format(dataset_type, folder_name, file_name)), list_bbs])

        self.data_table = data_table
        print '{}\n\tThere are {} instances of pedestrian in the {} labeled frames'.format(self.name, sum([len(r[1]) for r in data_table]), len(data_table))


class CaltechPedestrian(DetectionSource):
    def __init__(self, set_name='set01', step=1, skip_empty_frames=False, acceptable_labels=('person',)):
        super(CaltechPedestrian, self).__init__()
        if isinstance(set_name, basestring):
            set_name = [set_name]
        self.root_folder = os.path.join(DS_ROOT_DIR, 'Caltech_Pedestrian/data')
        self.img_width = 640
        self.img_height = 480
        self.original_shape = (640, 480)
        self.name = 'CaltechPedestrian'
        data_table = []
        dic = {}
        for sname in set_name:
            seq_files = os.listdir(os.path.join(self.root_folder, sname))
            for f in seq_files:
                ims = self.open_seq_file(os.path.join(self.root_folder, sname, f))
                bbs, lbls = self.load_annotation(sname, f)

                for im, bb in zip(ims, bbs):
                    list_bbs = []
                    if len(bb) > 0:
                        objects = bb[0]
                        for id, pos, posv, occl in zip(objects['id'], objects['pos'], objects['posv'], objects['occl']):
                            id = int(id[0] - 1)
                            lbl = str(lbls[id][0])
                            if lbl not in dic:
                                dic[lbl] = 1

                            if lbl not in acceptable_labels:
                                continue
                            pos_ped = np.ceil(posv[0]).astype(np.int32) if occl else np.ceil(pos[0]).astype(np.int32)
                            x1, y1, w, h = pos_ped
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = x1 + w
                            y2 = y1 + h
                            x2 = min(self.img_width, x2)
                            y2 = min(self.img_height, y2)

                            occl_ratio = posv[0][-2:].prod()/(pos[0][-2:].prod()+1e-6)
                            occl_ratio = 1 if occl_ratio == 0 else occl_ratio

                            w_2_h = float(w)/(h+1e-5)

                            if h >= 40 and (0.2 < w_2_h < 0.6) and occl_ratio > 0.65:
                                bb = [x1, y1, x2, y2]
                                list_bbs.append(bb)
                    if not skip_empty_frames:
                        data_table.append([im, list_bbs])
                    else:
                        if len(list_bbs) > 0:
                            data_table.append([im, list_bbs])

        print dic
        data_table = data_table[::step]
        self.data_table = data_table
        print '{}\n\tThere are {} instances of pedestrian in the {} labeled frames'.format(self.name, sum([len(r[1]) for r in data_table]), len(data_table))

    def open_seq_file(self, filename):
        """
        The main credit of this function goes to https://github.com/hamidb/pedestrian_detection.git.
        I borrowed it from this link.
        """
        import struct
        # read .seq file, and save the images into the savepath
        list_images = []
        with open(filename, 'rb') as f:
            # get rid of seq header containing some info about the seq
            header = str(f.read(548))
            self.img_width = struct.unpack('@i', f.read(4))[0]
            self.img_height = struct.unpack('@i', f.read(4))[0]
            # get rid of the rest
            header = str(f.read(468))
            string = str(f.read())
            # each image's header
            img_header = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"
            # split .seq file into segment with the image header
            strlist = string.split(img_header)
        count = 0
        for i in xrange(0, len(strlist)):
            img = strlist[i]
            # ignore the header
            if count > 0:
                # add image header to img data
                img = os.path.join(img_header[0:9], img)
                image = EncodedImageString(img) #cv2.imdecode(np.frombuffer(img, np.uint8), 1)
                list_images.append(image)
            count += 1

        return list_images

    def load_annotation(self, setname, seqname):
        """
        The main credit of this function goes to https://github.com/hamidb/pedestrian_detection.git.
        I borrowed it from this link.
        """
        from scipy.io import loadmat

        filepath = self.root_folder + '/annotations/' + setname + '/' + seqname.split('.')[0] + '.vbb'
        print(filepath)
        if not os.path.exists(filepath):
            print('Warning: annotation file for %s/%s is missing' % (setname, seqname))
            return [], [], -1
        vbbfile = loadmat(filepath)
        vbbdata = vbbfile['A'][0][0]
        objList = vbbdata[1][0]
        objLbl = vbbdata[4][0]
        return objList, objLbl

    def write_to_h5(self, save_to):
        f = h5py.File(save_to, 'w')
        # ds_images = f.create_dataset('images', (len(self.data_table), 480, 640, 3), dtype='f')
        ds_images = f.create_dataset('images', (len(self.data_table), ), dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        ds_bbs = f.create_dataset('bbs', (len(self.data_table), ), dt)

        for i in xrange(len(self.data_table)):
            row = self.data_table[i]
            im = np.frombuffer(row[0].image_string, np.uint8)
            bbs = row[1]
            bbs = np.asarray(bbs)
            bbs_flat = np.reshape(bbs, [-1])
            # bbs_rec = np.reshape(bbs_flat, [-1, 4])
            # ds_images[i, ...] = im[...]
            ds_images[i] = im
            ds_bbs[i] = bbs_flat
        f.close()


class CaltechPedestrianh5Py(DetectionSource):
    def __init__(self, h5py_name, step=1):
        super(CaltechPedestrianh5Py, self).__init__()
        self.root_folder = os.path.join(DS_ROOT_DIR, 'Caltech_Pedestrian/')
        self.img_width = 640
        self.img_height = 480
        self.original_shape = (640, 480)
        self.name = 'CaltechPedestrian'
        data_table = []
        f = h5py.File(os.path.join(self.root_folder, h5py_name), 'r')
        ds_bbs = f['bbs']
        for i in xrange(len(ds_bbs)):
            data_table.append([H5PyRecord(f, i), np.reshape(ds_bbs[i], [-1, 4]).tolist()])
        data_table = data_table[::step]
        self.data_table = data_table
        print '{}\n\tThere are {} instances of pedestrian in the {} labeled frames'.format(self.name, sum([len(r[1]) for r in data_table]), len(data_table))


class BDD100KPedestrianImages(DetectionSource):
    def __init__(self, source_type='train', step=1, skip_empty_frames=False):
        super(BDD100KPedestrianImages, self).__init__()
        assert source_type in ['train', 'val']

        self.name = 'BDD100KPedestrian'
        self.root_folder = os.path.join(DS_ROOT_DIR, 'BDD100K/bdd100k/videos/100k/{}'.format(source_type))
        self.img_width = 1280
        self.img_height = 720
        self.original_shape = (1280, 720)
        data_table = []
        import json as js
        root = os.path.join(DS_ROOT_DIR, 'BDD100K/bdd100k')
        with open('{}/labels/bdd100k_labels_images_{}.json'.format(root, source_type), 'r') as f:
            js_obj = js.load(f)
        for i in range(len(js_obj)):
            rec = js_obj[i]
            fname = os.path.splitext(rec['name'])[0]
            list_bbs = []
            for lbl in rec['labels']:
                if lbl['category'] != 'person':
                    continue
                bb = lbl['box2d']
                x1, y1, x2, y2 = int(bb['x1']), int(bb['y1']), int(bb['x2']), int(bb['y2'])
                w = x2 - x1
                h = y2 - y1
                w_2_h = float(w) / (h + 1e-5)

                if h >= 40 and (0.2 < w_2_h < 0.65):
                    list_bbs.append([x1, y1, x2, y2])

            im_path = os.path.join(root, 'images/100k/{}/{}.jpg'.format(source_type, fname))

            # if len(list_bbs) > 0:
            #     im = cv2.imread(im_path)
            #     im_draw = im.astype('uint8').copy()
            #     for b in list_bbs:
            #         cv2.rectangle(im_draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
            #     cv2.imshow('r', im_draw)
            #     cv2.waitKey(0)
            #
            # # cv2.imshow('src', im)
            # # cv2.waitKey(0)
            if not skip_empty_frames:
                data_table.append([im_path, list_bbs])
            else:
                if len(list_bbs) > 0:
                    data_table.append([im_path, list_bbs])

        data_table = data_table[::step]
        self.data_table = data_table
        print '{}\n\tThere are {} instances of pedestrian in the {} labeled frames'.format(self.name, sum([len(r[1]) for r in data_table]), len(data_table))
        print '{} out of {} frames contain pedestrian'.format(sum([len(r[1]) > 0 for r in data_table]), len(data_table))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # c = BDD100KPedestrianImages(source_type='train', skip_empty_frames=True)
    # c.crop_shape = (640, 480)
    c = CaltechPedestrianh5Py('/home/hamed/Desktop/Caltech_Pedestrian/train.h5py')
    c.augment_data = True
    for ii in range(0, 200):
        if ii % 1 == 0:
            print ii
        for i in range(1000):
            c.cur_ind = ii
            im_list, im_bb_score, im_bb_regr, list_path, list_bbs = c.read_batch(1)
            # continue

            # plt.figure(1, figsize=(18, 10))
            #
            # plt.gcf().add_subplot(231)
            im_draw = im_list[0].astype('uint8')
            for bb in list_bbs[0]:
                im_draw = cv2.rectangle(im_draw, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 255), 1)
            # plt.imshow(im_list[0].astype('uint8'))
            # plt.gcf().add_subplot(232)
            # plt.imshow(im_bb_score[0][..., 0], cmap='gray')
            # plt.gcf().add_subplot(233)
            # plt.imshow(im_bb_score[0][..., 1], cmap='gray')
            # plt.gcf().add_subplot(234)
            # plt.imshow(im_bb_score[0][..., 2], cmap='gray')
            # plt.gcf().add_subplot(235)
            # plt.imshow(im_bb_score[0][..., 3], cmap='gray')
            # plt.gcf().add_subplot(236)
            # plt.imshow(im_bb_score[0][..., 4], cmap='gray')
            plt.figure(2, figsize=(16, 9))
            plt.clf()
            plt.imshow(im_draw)
            plt.show()

    # c = CaltechPedestrianh5Py('/home/hamed/Desktop/Caltech_Pedestrian/val.h5py')
    # # c = CaltechPedestrian(['set01', 'set02', 'set03', 'set04', 'set05'], step=2, acceptable_labels=['person', 'people'])
    print len(c)
    for i in xrange(2000):
        im = c.read_at(i)[0]
        print c.data_table[i][0].vid, c.data_table[i][0].rotation
        bbs = c.data_table[i][1]
        im_draw = im.astype('uint8').copy()
        for b in bbs:
            cv2.rectangle(im_draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.imshow('r', im_draw)
        cv2.waitKey(0)

    # c = CaltechPedestrian(['set06', 'set07', 'set08', 'set09', 'set10'], step=10, skip_empty_frames=True, acceptable_labels=['person', 'people'])
    # c = CaltechPedestrian(['set01', 'set02', 'set03', 'set04', 'set05'], step=2, acceptable_labels=['person', 'people'])
    c = CaltechPedestrian(['set01', 'set02', 'set03', 'set04', 'set05'], step=1, skip_empty_frames=True, acceptable_labels=['person', 'people'])  # CityPerson('train') #
    c.write_to_h5('/home/hamed/Desktop/Caltech_Pedestrian/train.h5py')
    exit()

    # c = CaltechPedestrian(['set01', 'set02', 'set03', 'set04', 'set05'], step=1, skip_empty_frames=True, acceptable_labels=['person', 'people'])
    c = CaltechPedestrian(['set06', 'set07', 'set08', 'set09', 'set10'], step=10, skip_empty_frames=True,
                          acceptable_labels=['person', 'people'])

    # c = CaltechRoadsidePedestrian()
    exit()
