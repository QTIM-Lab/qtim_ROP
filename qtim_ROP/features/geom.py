import numpy as np
from scipy.spatial import KDTree
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.feature import canny
from PIL import Image
from ..utils.image import *
import cv2
from addict import Dict


def vec_length(vector):
    return np.linalg.norm(vector)


def unit_vector(vector):
    return vector / vec_length(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def pairwise_angles(lines):

    # line_starts = np.asarray([list(x[0]) for x in lines])
    line_ends = np.asarray([list(x[1]) for x in lines])
    tree = KDTree(line_ends)

    angles = []
    for i, line_A in enumerate(lines):

        # Find nearest line
        dists, j = tree.query(line_A[0])

        if i == j or j > len(lines):
            continue

        line_B = lines[j]

        # Calculate angle between vectors
        a0, a1 = np.asarray(line_A)
        b0, b1 = np.asarray(line_B)
        angles.append(angle_between(a1 - a0, b1 - b0))

    return angles


def line_lengths(lines):

    lengths = [np.linalg.norm([x1 - x0, y1 - y0]) for (x0, y0), (x1, y1) in lines]
    return lengths


def vessel_thickness(edt, skel):

    diameters = []

    for x, y in zip(*np.where(skel == 1)):
        diameters.append(edt[x, y])

    return diameters


def detect_optic_disk(image_rgb, disk_center, out_name):

    scale = 100
    w_sum = cv2.addWeighted(image_rgb, 4, cv2.GaussianBlur(image_rgb, (0, 0), scale / 30), -4, 128)#  * circular_mask + 128 * (1 - circular_mask)

    # Image.fromarray(np.mean(w_sum, axis=2).astype(np.uint8)).save(out_name)

    edges = canny(np.mean(w_sum, axis=2).astype(np.uint8), sigma=1.,
              low_threshold=50.)#, high_threshold=100.)
    result = hough_ellipse(edges, threshold=10, min_size=45, max_size=55)
    result.sort(order='accumulator')

    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)

    Image.fromarray(image_rgb).save(out_name)


def mask_od_vessels(skel, od_center):

    # Create optic disk mask
    od_mask = np.zeros_like(skel, dtype=np.uint8)
    cv2.circle(od_mask, od_center, 30, (1, 1, 1), -1)
    od_mask_inv = np.invert(od_mask) / 255.

    skel = skel.astype(np.float)
    masked_skel = skel * od_mask_inv

    return masked_skel.astype(np.uint8)


# def line_diameters(edt, lines):
#
#     diameters = []
#
#     for line in lines:
#
#         p0, p1 = [np.asarray(pt) for pt in line]
#         vec = p1 - p0  # vector between segment end points
#         vec_len = np.linalg.norm(vec)
#
#         pts_along_line = np.uint(np.asarray([p0 + (i * vec) for i in np.arange(0., 1., 1. / vec_len)]))
#
#         for pt in pts_along_line:
#
#             try:
#                 diameters.append(edt[pt[0], pt[1]])
#             except IndexError:
#                 pass
#
#     return diameters