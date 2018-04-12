import cv2
import numpy as np
from ..retinaunet.lib.pre_processing import my_PreProc

from ..segmentation.mask_retina import create_mask


def unet_preproc(img):

    width, height, channels = img.shape
    imgs_arr = np.empty((1, width, height, channels))
    imgs_arr[0] = img

    imgs_arr = np.transpose(imgs_arr, (0, 3, 1, 2))
    pre_proc = np.stack([my_PreProc(imgs_arr)[0, 0, :, :]] * 3, axis=-1)

    pre_proc = ((pre_proc / pre_proc.max()) * 255).astype(np.uint8)
    return pre_proc


# https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/
# kaggleDiabeticRetinopathy/preprocessImages.py
def kaggle_BG(img, scale=300):

    # Create a mask from which the approximate retinal center can be calculated
    guide_mask = create_mask(img)
    retina_center = tuple((np.mean(np.argwhere(guide_mask), axis=0)).astype(np.uint16)[::-1])

    # Generate a circle of the approximate size, centered based on the guide mask
    cf = 0.95
    circular_mask = np.zeros(img.shape)
    cv2.circle(circular_mask, retina_center, int(scale * cf), (1, 1, 1), -1, 8, 0)

    # Compute weight sum of image, blurred image and mask it
    w_sum = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), scale / 30), -4, 128) * circular_mask + 128 * (1 - circular_mask)
    return w_sum.astype(np.uint8)


# https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/
# kaggleDiabeticRetinopathy/preprocessImages.py
def scale_radius(img, scale):
    x = img[img.shape[0] / 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def normalize_channels(img):
    for colorband in range(img.shape[2]):
        img[:, :, colorband] = image_histogram_equalization(img[:, :, colorband])
    return img

# http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
def image_histogram_equalization(image, number_bins=256):

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def binary_morph(img, thresh=50, min_size=None, mask_only=True):

    if min_size is None:  # default to 10% of largest image dimension
        min_size = float(max(img.shape)) * .1

    if len(img.shape) == 3:  # flatten if RGB image
        img = np.mean(img, 2).astype(np.uint8)

    # Apply binary threshold and erode
    ret, thresh_im = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    # Connected component labelling
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_im)
    mask = np.zeros_like(labels)

    # Loop through areas in order of size
    areas = [s[4] for s in stats]
    sorted_idx = np.argsort(areas)

    for lidx, cc in zip(sorted_idx, [areas[s] for s in sorted_idx][:-1]):

        if cc > min_size:
            mask[labels == lidx] = 1

    if mask_only:
        return mask * 255
    return np.dstack([img * mask] * 3).astype(np.uint8)
