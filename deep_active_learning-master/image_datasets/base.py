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

import csv
import cv2
import random
import numpy as np
import augment_image.augment_data as aug
import augment_image.image_degrade as deg
from os.path import join
from collections import OrderedDict
from random import sample, uniform


class EncodedImageString:
    def __init__(self, img_str=''):
        self._img_str = img_str

    @property
    def image_string(self):
        return self._img_str

    @image_string.setter
    def image_string(self, value):
        self._img_str = value


class H5PyRecord:
    def __init__(self, f, ind):
        self._f = f
        self._ind = ind
        self.hash = str(hash(str(bytes(self._f['images'][self._ind].data))))

    def read(self):
        ds_images = self._f['images']
        im = cv2.imdecode(ds_images[self._ind], 1)
        return im


class DataSource(object):
    """
    Base class for reading data from disk and augmenting it on-the-fly
    """
    def __init__(self):
        self.cur_ind = 0
        self.data_table = None
        self.num_of_records = 0
        self.num_of_classes = 0
        self.resize_shape = None
        self.crop_shape = None
        self.is_color = True
        self.scale = 1
        self.mean_image = None
        self.shuffle = True
        self.aug_dic = None
        self.augment_data = False
        self.class_frequency = [1] * self.num_of_classes
        self.name = self.__class__.__name__
        self.root_folder = None
        self.on_end_of_dataset = None

    def __delitem__(self, key):
        if isinstance(key, int):
            del self.data_table[key]
            self.num_of_records = len(self.data_table)

    def __len__(self):
        return len(self.data_table)

    def __setattr__(self, key, value):
        if key == 'data_table':
            self.__dict__[key] = value
            self.num_of_records = len(value) if value is not None else 0
        else:
            self.__dict__[key] = value

    def clone(self):
        raise NotImplementedError()

    def write_to_h5(self, save_to):
        raise NotImplementedError()

    def read_imagesources(self, imagelist_file, delimiter=' '):
        raise NotImplementedError()

    def _read_image(self, ind):
        raise NotImplementedError()

    def seek_ind(self, x, type='absolute'):
        if type == 'absolute':
            self.cur_ind = x
        elif type == 'relative':
            self.cur_ind += x
        return self

    def shuffle_data(self):
        random.shuffle(self.data_table)

    def read_next(self):
        self.cur_ind += 1
        if self.cur_ind >= self.num_of_records:
            self.cur_ind = 0
            if callable(self.on_end_of_dataset):
                self.on_end_of_dataset(self)

            if self.shuffle is True:
                # print 'Shuffling data {}'.format(self.name)
                self.shuffle_data()

        return self.read_at(self.cur_ind)

    def read_previous(self):
        self.cur_ind -= 1
        if self.cur_ind < 0:
            self.cur_ind = self.num_of_records

        return self.read_at(self.cur_ind)

    def read_at(self, ind):
        if ind >= self.num_of_records:
            ind = 0
        elif ind < 0:
            ind = self.num_of_records
        self.cur_ind = ind

        return self._read_image(ind)

    def read_batch(self, n, format='NHWC'):
        raise NotImplementedError()


class DetectionSource(DataSource):
    """
    Inherited from DataSource, this class reads images from disk, H5 or seq files along with their bounding boxes
    and generate regression map as well as labeling map.
    """
    def __init__(self):
        super(DetectionSource, self).__init__()
        self.original_shape = None
        self.aug_dic = [#(aug.blur_median, {'ks': 3}),
                        # (aug.blur_median, {'ks': 5}),
                        # (aug.blur_median, {'ks': 7}),
                        (aug.sharpen, {}),
                        (aug.smooth_gaussian, {'ks': 3}),
                        # (aug.smooth_gaussian, {'ks': 5}),
                        # (aug.smooth_gaussian, {'ks': 7}),
                        (aug.motion_blur, {'theta': np.random.RandomState(), 'ks': 5}),
                        # (aug.motion_blur, {'theta': np.random.RandomState(), 'ks': 7}),
                        (deg.dropout, {'prob': 0.07}),
                        (deg.gaussian_noise_shared, {}),
                        # (deg.hurl, {'prob': 0.05}),
                        (aug.hsv, {'scale': np.random.uniform(0.7, 1.05), 'channel': 2}),
                        (aug.hsv, {'scale': np.random.uniform(0.9, 1.1), 'channel': 1}),
                        (deg.pick, {'prob': 0.1}),
                        (aug.flip, {}),
                        (aug.flip, {}),
                        (aug.flip, {}),
                        (aug.flip, {}),
                        ]
        self.resize_along_x_prob = 0.0
        self.generate_fixed_random_crops = False

        self.default_bbs = [[31, 55], [47, 79], [79, 143], [127, 223], [159, 271]]

    def __setattr__(self, key, value):
        if key == 'data_table':
            self.__dict__[key] = value
            self.num_of_records = len(value) if value is not None else 0
        else:
            self.__dict__[key] = value

    @staticmethod
    def compute_iou(bb1, bb2):
        """
        computes IoU of two bounding boxes
        :param bb1: [x1, y1, x2, y2]
        :param bb2: [x1. y1. x2. y2]
        :return:
        """

        x1 = max([bb1[0], bb2[0]])
        y1 = max([bb1[1], bb2[1]])
        x2 = min([bb1[2], bb2[2]])
        y2 = min([bb1[3], bb2[3]])
        if x1 > x2 or y1 > y2:
            intersection = 0
        else:
            intersection = float((x2 - x1) * (y2 - y1))

        iou = intersection / (
                (bb1[2] - bb1[0]) * (bb1[3] - bb1[1]) + (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]) - intersection)
        iou = 0 if iou < 0 else iou

        assert 0 <= iou <= 1
        return iou

    def clone(self):
        ds = DetectionSource()
        ds.cur_ind = self.cur_ind + 0
        ds.data_table = self.data_table + []
        ds.num_of_records = self.num_of_records + 0
        ds.num_of_classes = self.num_of_classes + 0
        ds.resize_shape = (self.resize_shape[0] + 0, self.resize_shape[1] + 0) if self.resize_shape is not None else None
        ds.crop_shape = (self.crop_shape[0] + 0, self.crop_shape[1] + 0) if self.crop_shape is not None else None
        ds.is_color = True if self.is_color is True else False
        ds.scale = self.scale + 0
        ds.mean_image = self.mean_image
        ds.shuffle = True if self.shuffle is True else False
        ds.augment_data = True if self.augment_data is True else False
        ds.class_frequency = self.class_frequency + []
        ds.root_folder = self.root_folder + ''
        ds.original_shape = (
        self.original_shape[0] + 0, self.original_shape[1] + 0) if self.original_shape is not None else None
        ds.aug_dic = self.aug_dic + []
        ds.resize_along_x_prob = self.resize_along_x_prob + 0
        return ds

    def read_imagesources(self, imagelist_file, delimiter=' '):
        with open(imagelist_file, 'r') as file_id:
            data_reader = csv.reader(file_id, delimiter=delimiter)
            data_reader = [[r[0], int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])] for r in data_reader]
        del file_id
        self.num_of_records = len(data_reader)

        dic_bbs = OrderedDict()
        for r in data_reader:
            if r[0] in dic_bbs:
                dic_bbs[r[0]].append(r[1:])
            else:
                dic_bbs[r[0]] = []
                dic_bbs[r[0]].append(r[1:])

        self.data_table = dic_bbs.items()

    def get_output_shape(self):
        if self.crop_shape is not None:
            return self.crop_shape
        if self.resize_shape is not None:
            return self.resize_shape
        return self.original_shape

    def _read_image(self, ind):
        row = self.data_table[ind]
        gt_bbs = []
        for bb in row[1]:
            gt_bbs.append([b + 0 for b in bb])

        if isinstance(row[0], basestring):
            if self.root_folder is not None:
                row = [join(self.root_folder, row[0]), [b + [] for b in gt_bbs]]
            im = cv2.imread(row[0])
            img_path = row[0]
        elif isinstance(row[0], np.ndarray):
            im = row[0].copy()
            img_path = '{}.png'.format(self.cur_ind)
        elif isinstance(row[0], EncodedImageString):
            im = cv2.imdecode(np.frombuffer(row[0].image_string, np.uint8), 1)
            img_path = hash(row[0].image_string)
        elif isinstance(row[0], H5PyRecord):
            im = row[0].read()
            img_path = row[0].hash
        else:
            raise ValueError('Invalid data')

        if im is None:
            print row
            raise Exception('Image is None')

        if self.augment_data is True and random.uniform(0, 1) < 0.6:
            aug_func = random.sample(self.aug_dic, 1)[0]
            im = aug_func[0](im, **aug_func[1])
            if aug_func[0] is aug.flip:
                for bb in gt_bbs:
                    temp = bb[2]
                    bb[2] = im.shape[1] - bb[0]
                    bb[0] = im.shape[1] - temp

        fx = 1
        fy = 1
        if self.resize_shape is not None:
            fx = float(self.resize_shape[0]) / im.shape[1]
            fy = float(self.resize_shape[1]) / im.shape[0]
            im = cv2.resize(im, self.resize_shape)

        if uniform(0, 1) > 1 - self.resize_along_x_prob:
            fx = 0.7
            fy = 1
            im = cv2.resize(im, None, fx=fx, fy=fy)

        for bb in gt_bbs:
            bb[0] = int(bb[0] * fx)
            bb[2] = int(bb[2] * fx)
            bb[1] = int(bb[1] * fy)
            bb[3] = int(bb[3] * fy)

        im = im.astype('float32')
        if self.mean_image is not None:
            im = im - self.mean_image

        im = im * self.scale
        w = float(im.shape[1])
        h = float(im.shape[0])

        if self.crop_shape is not None:
            if self.crop_shape[0] > w or self.crop_shape[1] > h:
                raise Exception('Crop size is larger than the image!')

            if len(gt_bbs) < 1:
                rand = np.random.RandomState()
                if self.generate_fixed_random_crops:
                    rand.seed(np.sum(np.power(im[0, 0, :], 2)).astype('int32'))
                x1 = rand.randint(0, im.shape[1] - self.crop_shape[0])
                y1 = rand.randint(0, im.shape[0] - self.crop_shape[1])
                y2 = y1 + self.crop_shape[1]
                x2 = x1 + self.crop_shape[0]
                im = im[y1:y2, x1:x2, :]
            else:
                rand = np.random.RandomState()
                if self.generate_fixed_random_crops:
                    rand.seed(np.sum(np.power(im[0, 0, :], 2)).astype('int32'))
                bb = sample(gt_bbs, 1)[0]
                x1 = rand.randint(0, bb[0] + 1)
                y1 = rand.randint(0, bb[1] + 1)
                if x1 + self.crop_shape[0] >= w:
                    dx = x1 + self.crop_shape[0] - w
                else:
                    dx = 0

                if y1 + self.crop_shape[1] >= h:
                    dy = y1 + self.crop_shape[1] - h
                else:
                    dy = 0

                if x1 - dx < 0 or y1 - dy < 0:
                    print x1, y1, dx, dy
                x1 -= dx
                y1 -= dy
                y2 = y1 + self.crop_shape[1]
                x2 = x1 + self.crop_shape[0]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                im = im[y1:y2, x1:x2, :]

                w = float(im.shape[1])
                h = float(im.shape[0])

                i = 0
                while i < len(gt_bbs):
                    bb = gt_bbs[i]
                    bb[0] -= x1
                    bb[1] -= y1
                    bb[2] -= x1
                    bb[3] -= y1
                    i += 1

                i = 0
                list_bbs = []
                while i < len(gt_bbs):
                    bb = gt_bbs[i]
                    w = bb[2] - bb[0]
                    h = bb[3] - bb[1]
                    w_2_h = float(w) / (h + 1e-5)

                    if bb[0] >= 0 and bb[1] >= 0 and h >= 40 and (0.2 < w_2_h < 0.65):
                        list_bbs.append(bb)
                    i += 1
                gt_bbs = list_bbs

        # #############################################################
        # GENERATING DETECTION NETWORK MULTI SCALE GT
        # #############################################################
        def assign_label(bb_class_score, d, class_ind):
            if isinstance(d, list):
                sz_h = 5 + int(np.sqrt(d[1] * 2))
                sz_w = 5 + int(np.sqrt(d[0] * 2))
            else:
                sz_h = 5 + int(np.sqrt(d * 2))
                sz_w = sz_h

            bb_class_score[max(0, center_h - sz_h):min(im.shape[0] - 1, center_h + sz_h + 1),
            max(0, center_w - sz_w):min(im.shape[1] - 1, center_w + sz_w + 1),
            :] = 0

            if isinstance(d, list):
                sz_h = 5 + int(np.sqrt(d[1] * 0.2))
                sz_w = 5 + int(np.sqrt(d[0] * 0.2))
            else:
                sz_h = 5 + int(np.sqrt(d * 0.2))
                sz_w = sz_h

            bb_class_score[max(0, center_h - sz_h):min(im.shape[0] - 1, center_h + sz_h + 1),
            max(0, center_w - sz_w):min(im.shape[1] - 1, center_w + sz_w + 1),
            class_ind] = 1

            bb_class_score[max(0, center_h - sz_h):min(im.shape[0] - 1, center_h + sz_h + 1),
            max(0, center_w - sz_w):min(im.shape[1] - 1, center_w + sz_w + 1),
            -1] = 0

        bb_class_score = np.zeros((im.shape[0], im.shape[1], len(self.default_bbs) + 1), dtype='float32')
        bb_regression = np.zeros((im.shape[0], im.shape[1], 4), dtype='float32')
        bb_class_score[..., -1] = 1

        dic_items = self.default_bbs

        for bb in gt_bbs:
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            max_iou = -1
            max_class_ind = None
            max_d = 0
            max_matched_bb = None
            # for d, class_ind in dic_items:
            for class_ind, d in enumerate(dic_items):
                if isinstance(d, list):
                    d_bb_w, d_bb_h = d[0] / 2, d[1] / 2
                else:
                    d_bb_w, d_bb_h = d / 2, d / 2
                d_bb = [bb[0] + bb_w / 2 - d_bb_w,
                        bb[1] + bb_h / 2 - d_bb_h,
                        bb[0] + bb_w / 2 + d_bb_w,
                        bb[1] + bb_h / 2 + d_bb_h]
                iou = self.compute_iou(bb, d_bb)
                if iou > max_iou:
                    max_class_ind = class_ind
                    max_iou = iou
                    max_d = d
                    max_matched_bb = bb

                if iou > 0.7:
                    center_h = bb[1] + bb_h / 2
                    center_w = bb[0] + bb_w / 2
                    # print d
                    assign_label(bb_class_score, d, class_ind)

            if max_class_ind is None:
                print 'Max ind is None:'
                print 'Defaults bbs:'
                print dic_items
                print 'GT bbs:'
                print gt_bbs

            center_h = bb[1] + bb_h / 2
            center_w = bb[0] + bb_w / 2
            # print max_d
            assign_label(bb_class_score, max_d, max_class_ind)

        return im, bb_class_score, bb_regression, img_path, gt_bbs + []

    def read_batch(self, n, format='NHWC'):
        n_samples_to_read = len(n) if isinstance(n, list) else n
        w, h = self.crop_shape if self.crop_shape is not None else (
            self.original_shape if self.resize_shape is None else self.resize_shape)
        if format == 'NCHW':
            im_list = np.zeros([n_samples_to_read,
                                3 if self.is_color else 1,
                                h,
                                w],
                               dtype='float32')
        elif format == 'NHWC':
            im_list = np.zeros([n_samples_to_read,
                                h,
                                w,
                                3 if self.is_color else 1],
                               dtype='float32')
        else:
            raise Exception('Invalid format')

        im_bb_score = np.zeros([n_samples_to_read,
                                h,
                                w,
                                len(self.default_bbs) + 1],
                               dtype='float32')
        im_bb_regr = np.zeros([n_samples_to_read,
                               h,
                               w,
                               4],
                              dtype='float32')

        list_path = [None] * n_samples_to_read
        list_bbs = [None] * n_samples_to_read
        if isinstance(n, list):
            for i in xrange(n_samples_to_read):
                im, score, regress, path, bbs = self.read_at(n[i])
                if format == 'NCHW':
                    im = np.transpose(im, (2, 0, 1))
                im_list[i, ...] = im[...]
                im_bb_score[i, ...] = score[...]
                im_bb_regr[i, ...] = regress[...]
                list_path[i] = path
                list_bbs[i] = bbs
        else:
            for i in xrange(n_samples_to_read):
                im, score, regress, path, bbs = self.read_next()
                if format == 'NCHW':
                    im = np.transpose(im, (2, 0, 1))
                im_list[i, ...] = im[...]
                im_bb_score[i, ...] = score[...]
                im_bb_regr[i, ...] = regress[...]
                list_path[i] = path
                list_bbs[i] = bbs

        return im_list, im_bb_score, im_bb_regr, list_path, list_bbs

