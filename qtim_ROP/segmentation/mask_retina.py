from skimage import measure
from skimage.color import rgb2gray
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, selem
from skimage.measure import regionprops
import cv2

from PIL import Image
import numpy as np
from os.path import split, join, splitext
from ..utils.common import find_images


def create_mask(im_arr, erode=0):

    if im_arr.shape[2] == 3:
        im_arr = rgb2gray(im_arr)

    thresh = 0.05
    inv_bin = np.invert(im_arr > thresh)
    all_labels = measure.label(inv_bin)

    # Select largest object and invert
    seg_arr = all_labels == 0

    if erode > 0:
        strel = selem.disk(erode, dtype=np.bool)
        seg_arr = binary_erosion(seg_arr, selem=strel)
    elif erode < 0:
        strel = selem.disk(abs(erode), dtype=np.bool)
        seg_arr = binary_dilation(seg_arr, selem=strel)

    return seg_arr.astype(np.bool)


def circular_mask(img, cf=.98):

    # Generate approximate mask and estimate radius and retina center
    guide_mask = create_mask(img)

    # Approximate the radius
    mask_idx = np.asarray(np.argwhere(guide_mask))
    min_x, max_x = np.min(mask_idx[:, 1]), np.max(mask_idx[:, 1])

    # If either min/max horizontal extents are 0, we need to adjust our retina center
    if min_x == 0:  # image is shifted to the left
        x_shift = -int((img.shape[1] - max_x))
    elif max_x == 0:
        x_shift = int(min_x)
    else:
        x_shift = 0

    radius = (max_x - min_x + x_shift) / 2.
    retina_center = np.round(regionprops(guide_mask.astype(np.uint8))[0].centroid).astype(np.uint16)[::-1]
    retina_center[0] += x_shift

    # Generate a circle of the approximate size, centered based on the guide mask
    c_mask = np.zeros(img.shape)
    cv2.circle(c_mask, tuple(retina_center), int(radius * cf), (1, 1, 1), -1, 8, 0)

    return c_mask


def apply_mask(im, mask):

    im[np.invert(mask.astype(np.bool))] = 0
    return im[:,:,0]


if __name__ == "__main__":

    import sys
    out_dir = sys.argv[2]

    for im_path in find_images(sys.argv[1]):

        im = np.asarray(Image.open(im_path))
        mask = circular_mask(im).astype(np.uint8) * 255

        _, file_name = split(im_path)
        name, ext = splitext(file_name)
        Image.fromarray(mask).save(join(out_dir, name + '_mask.gif'))
