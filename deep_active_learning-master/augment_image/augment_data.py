"""
This is cloned and modified from my old repository in GitHub.

This code is implemented as a part of the following paper and it is only meant to reproduce the results of the paper:
    "Active Learning for Deep Detection Neural Networks,
    "Hamed H. Aghdam, Abel Gonzalez-Garcia, Joost van de Weijer, Antonio M. Lopez", ICCV 2019
_____________________________________________________

Developer/Maintainer:  Hamed H. Aghdam
Year:                  2018-2019
License:               BSD
_____________________________________________________

"""

import cv2
import numpy as np


def smooth_gaussian(im, ks):
    """
    To Smooth the image using a Gaussian kernel. The variance of Gaussian filter is computed based on the kernel size.
    :param im: Image to be smoothed
    :param ks: (tuple, list or int) Size of Guassian filter
    :return:
    """

    if isinstance(ks, int) or isinstance(ks, float):
        ks = (ks, ks)

    sigma_x = (ks[1] // 2.) / 3.
    sigma_y = (ks[0] // 2.) / 3.
    return cv2.GaussianBlur(im, ksize=ks, sigmaX=sigma_x, sigmaY=sigma_y)


def motion_blur(im, theta, ks):
    """
    Simulates motion blur effect on the image.
    :param im: Input image.
    :param theta: (float) Direction of blur in Degrees
    :param ks: Size of filter.
    :return: Image after applying motion blur effect.
    """

    if ks < 3:
        return im

    if isinstance(theta, np.random.RandomState):
        theta = theta.randint(0, 180)
    theta = theta * np.pi / 180.

    # Creating a filter where all elements except elementing lying of line with oreinetaion theta are zero
    kernel = np.zeros((ks, ks), dtype='float32')
    half_len = ks // 2
    x = np.linspace(-half_len, half_len, 2*half_len+1,dtype='int32')
    y = -np.round(np.sin(theta)*x/(np.cos(theta)+1e-4)).astype('int32')
    ind = np.where(np.abs(y) <= half_len)
    # print x, y,theta
    x += half_len
    y += half_len
    kernel[y[ind], x[ind]] = 1.0

    y = np.linspace(-half_len, half_len, 2 * half_len + 1, dtype='int32')
    x = -np.round(np.cos(theta) * y/ (np.sin(theta)+1e-4)).astype('int32')
    ind = np.where(np.abs(x) <= half_len)
    # print x, y, theta
    x += half_len
    y += half_len
    kernel[y[ind], x[ind]] = 1.0

    #Normalizing filter
    kernel = np.divide(kernel, kernel.sum())
    im_res = cv2.filter2D(im, cv2.CV_8UC3, kernel)
    # np.set_printoptions(2,linewidth=120)
    # print kernel
    # import matplotlib.pyplot as plt
    # plt.clf()
    # plt.imshow(kernel)
    # plt.show()

    return im_res


def blur_median(im, ks):
    """
    Smoothing the image using median filtering.
    :param im: Input image
    :param ks: (int) Window size
    :return: Smoothed image.
    """
    return cv2.medianBlur(im, ks)


def sharpen(im, ks=(3, 3), alpha=1):
    """
    Sharpens the input image.
    :param im: Input image.
    :param ks: (tuple or list) Kernel size
    :param alpha: Strength of fine image. [Default=1]
    :return: Sharpenned image.
    """
    sigma_x = (ks[1] // 2.) / 3.
    sigma_y = (ks[0] // 2.) / 3.
    im_res = im.astype('float32') * 0.0039215
    im_coarse = cv2.GaussianBlur(im_res, ks, sigmaX=sigma_x, sigmaY=sigma_y)
    im_fine = im_res - im_coarse
    im_res += alpha * im_fine
    return np.clip(im_res * 255, 0, 255).astype('uint8')


def crop(im, shape, rand=np.random.RandomState()):
    """
    Randomly crops the image.
    :param im: Input image
    :param shape: (list, tuple) shape of cropped image.
    :param rand: An instance of numpy.random.RandomState objects
    :return: Randomlly cropped image.
    """

    if len(shape) == 2:
        if 0 < shape[0] <= 1 and 0 < shape[1] <= 1:
            dx = rand.randint(0, max(1, int((1-shape[0])*im.shape[1])))
            dy = rand.randint(0, max(1, int((1-shape[1])*im.shape[0])))
            im_res = im[dy:int(im.shape[0] * shape[0]) + dy, dx:int(im.shape[1] * shape[1]) + dx, :].copy()
        else:
            dx = rand.randint(0, im.shape[1] - shape[0])
            dy = rand.randint(0, im.shape[0] - shape[1])
            im_res = im[dy:shape[0] + dy, dx:shape[1] + dx, :].copy()
    elif len(shape) == 4:
        if 0 < shape[2] <= 1 and 0 < shape[3] <= 1:
            dx = int(shape[0] * im.shape[1])
            dy = int(shape[1] * im.shape[0])
            im_res = im[dy:int(im.shape[0] * shape[3]), dx:int(im.shape[1] * shape[2]), :].copy()
        else:
            dx = shape[0]
            dy = shape[1]
            im_res = im[dy:shape[3], dx:shape[2], :].copy()
    else:
        raise ValueError()
    return im_res


def hsv(im, scale, p=1, channel=2):
    """
    Scales hue, saturation or value of image (res = (scale*HSV[:,:,channel])^p).
    :param im: Input image.
    :param scale: Scale
    :param p: power
    :param channel: H=0, S=1, V=2
    :return:
    """
    if channel != 1 and channel != 2:
        raise Exception('componenet can be only 1 or 2')

    im_res = im.astype('float32')/255.
    im_res = cv2.cvtColor(im_res, cv2.COLOR_BGR2HSV)
    im_res[:, :, channel] = np.power(im_res[:, :, channel] * scale, p)
    im_res[:, :, channel] = np.clip(im_res[:, :, channel], 0, 1)
    im_res = (cv2.cvtColor(im_res, cv2.COLOR_HSV2BGR)*255).astype('uint8')
    return im_res


def resize(im, scale_x, scale_y, interpolation=cv2.INTER_NEAREST, keep_original_size=False):
    im_res = cv2.resize(im, None, fx=scale_x, fy=scale_y, interpolation=interpolation)
    if keep_original_size:
        im_zero = im * 0
        im_zero[:im_res.shape[0], :im_res.shape[1], ...] = im_res[...]
        im_res = im_zero

    return im_res


def flip(im):
    return im[:, -1::-1, :].copy()

