"""
This is cloned and modified from my old repository in GitHub
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


def _check_im(im):
    """
    It checks the image and it raise an exception if it is not a 3-channel image and it is not of type uint8.
    :param im: image written by cv2.imread function.
    :return: None
    """
    assert im.dtype == 'uint8' and im.ndim == 3, 'Image must be a 3 channel uint array.'
    # if im.dtype != 'uint8' or im.ndim != 3:
    #     raise Exception('Image must be a 3 channel uint array.')


def _check_rand(rand):
    assert (rand is None) or (isinstance(rand, np.random.RandomState)), 'rand has to be an instance of np.random.RandomState'
    # if (rand is not None) and (not isinstance(rand, np.random.RandomState)):
    #     raise Exception('rand has to be an instance of np.random.RandomState')


def _check_prop(prop):
    assert 0 <= prop <= 1, 'prob has to be in interval [0,1]'
    # if not(0 < prop < 1):
    #     raise  Exception('sparse_prob has to be in interval [0,1]')


def sparsify(mat, sparse_prob=0.2, rand_object=None):
    """
    Given a tensor, it randomly zeros some of the elements.
    :param mat: The matrix to be sparsified.
    :param sparse_prob: Probability of zeroing an element in the tensor.
    :param rand_object: (optional) An instance of np.random.RandomState
    :return: returns the sparsified tensor.
    """

    _check_prop(sparse_prob)
    _check_rand(rand_object)

    if rand_object is None:
        mask = np.random.binomial(1, 1-sparse_prob, mat.shape)
    else:
        mask = rand_object.binomial(1, 1 - sparse_prob, mat.shape)
    return np.multiply(mat, mask)


def gaussian_noise_shared(im, mu=0, sigma=1, sparse_prob=0.0, rand_object=None):
    """
    Add a Gaussian noise to the input image. The same noise pattern is genrated for each channel.
    :param im: Image that must be degraded.
    :param mu: mean of Gaussian distribution.
    :param sigma: Variance of Gaussian distrobution.
    :param sparse_prob: Probability of adding noise to an element in the input image.
    :param rand_object:
    :return:(optional) An instance of np.random.RandomState
    """
    _check_im(im)
    _check_prop(sparse_prob)
    _check_rand(rand_object)
    if rand_object is None:
        noise_mask = np.random.normal(mu, sigma, (im.shape[0], im.shape[1], 1))
    else:
        noise_mask = rand_object.normal(mu, sigma, (im.shape[0], im.shape[1], 1))
    noise_mask = np.dstack((noise_mask, noise_mask, noise_mask))

    if sparse_prob is not None:
        noise_mask = sparsify(noise_mask, sparse_prob)
    return cv2.add(im, noise_mask, dtype=cv2.CV_8UC3)


def pick(im, prob, win_size=(3, 3), rand_object=None):
    """
    It degrades the image by randomly replcaging the current pixel with a pixel in its neighborhood.
    More information can be found at:
    [https://docs.gimp.org/en/plug-in-randomize-pick.html]
    :param im: image that must be degraded.
    :param prob: probabiityt of degrading a pixel
    :param win_size: neighbourhood size.
    :param rand_object: (optional) An instance of numpy.random.RandomState object.
    :return: degraded image.
    """
    _check_prop(prob)
    if isinstance(win_size, tuple) is False or len(win_size) < 2:
        raise Exception('win_size must be a *tuple* containing only 2 elemens.')

    if rand_object is None:
        mask_X = np.random.randint(0, win_size[1]-1, (im.shape[0], im.shape[1]))-win_size[1]//2
        mask_Y = np.random.randint(0, win_size[0]-1, (im.shape[0], im.shape[1]))-win_size[0]//2
        mask = np.random.binomial(1, prob, im.shape[:2])
    else:
        mask_X = rand_object.randint(0, win_size[1] - 1, (im.shape[0], im.shape[1])) - win_size[1] // 2
        mask_Y = rand_object.randint(0, win_size[0] - 1, (im.shape[0], im.shape[1])) - win_size[0] // 2
        mask = rand_object.binomial(1, prob, im.shape[:2])

    mask_X[mask == 0] = 0
    mask_Y[mask == 0] = 0
    Y, X = np.meshgrid(np.arange(0, im.shape[0]), np.arange(0, im.shape[1]), indexing='ij')

    X_pick = X+mask_X
    Y_pick = Y+mask_Y

    Y_pick[Y_pick < 0] = 0
    Y_pick[Y_pick >= im.shape[0]] = im.shape[0]-1

    X_pick[X_pick < 0] = 0
    X_pick[X_pick >= im.shape[1]] = im.shape[1]-1

    ind_pick = np.add(np.multiply(Y_pick, im.shape[1]), X_pick).astype('int32')

    im_c0 = im[:, :, 0]
    im_c1 = im[:, :, 1]
    im_c2 = im[:, :, 2]
    im_c0 = np.reshape(im_c0.flatten()[ind_pick], im_c0.shape)
    im_c1 = np.reshape(im_c1.flatten()[ind_pick], im_c1.shape)
    im_c2 = np.reshape(im_c2.flatten()[ind_pick], im_c2.shape)

    im_noise = im * 0
    im_noise[:, :, 0] = im_c0
    im_noise[:, :, 1] = im_c1
    im_noise[:, :, 2] = im_c2

    return im_noise


def dropout(im, prob=0.2, rand_object=None):
    """
    Degrade the image by randomly zeroing pixels.
    :param im: Image that must be degraded.
    :param prob: Probability of zeroing a pixel.
    :param rand_object: (optional) An instance of numpy.random.RandomState object.
    :return: Degraded image.
    """
    _check_prop(prob)
    _check_rand(rand_object)
    if rand_object is None:
        mask = np.random.binomial(1, 1 - prob, (im.shape[0], im.shape[1]))
    else:
        mask = rand_object.binomial(1, 1 - prob, (im.shape[0], im.shape[1]))

    mask = np.dstack((mask,mask,mask))
    return np.multiply(im.astype('float32'), mask).astype('uint8')


if __name__ == '__main__':
    im = cv2.imread('./examples/nature.jpg')
    im = cv2.resize(im, dsize=None, fx= 0.4, fy= 0.4)
    # im_noise = gaussian_noise(im,mu=0, sigma=30, sparse_prob=0.1)
    # im_noise = uniform_noise(im, d_min=-60, d_max=60, sparse_prob=0.1)
    im_noise = gaussian_noise_shared(im, mu=0, sigma=30, sparse_prob=0.1)
    # im_noise = uniform_noise_shared(im, d_min=-60, d_max=60, sparse_prob=0.1)
    # im_noise = pick(im, 0.9, (5,5))
    # im_noise = dropout(im, prob=0.3)
    # im_noise = hurl(im, prob=0.2)
    # im_noise = hue(im, prob=0.1, d_min=-20,d_max=20)
    # im_noise = hsv_uniform(im, prob=0.1, d_min=-50,d_max=50,h_scale=0,s_scale=0,v_scale=1)
    # im_noise = hsv_gaussian(im, prob=0.1, mue=-30, sigma=20, h_scale=0, s_scale=0, v_scale=1)

    cv2.namedWindow('degraded', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('degraded', im_noise)
    ch = cv2.waitKey()
    if ch == 115:
        print 'Saving on disk!'
        cv2.imwrite('degraded_hsv_gauss_value.png', im_noise)
    cv2.destroyAllWindows()


