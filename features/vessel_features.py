from PIL import Image
import numpy as np
from glob import glob
from os.path import join, basename
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.color import rgb2gray


def vessel_features(orig_dir, seg_dir, out_dir):

    prob = .5
    orig_images = [np.asarray(Image.open(x)) for x in sorted(glob(join(orig_dir, '*.*')))]
    seg_images = [np.asarray(Image.open(x)) for x in sorted(glob(join(seg_dir, '*.png')))]

    for i, (orig, seg) in enumerate(zip(orig_images, seg_images)):

        mask = (seg > (255 * prob)).astype(np.uint8)
        overlay_mask(orig, mask, join(out_dir, '{}.png'.format(i)))


def overlay_mask(img, mask, out):

    img_gray = sitk.GetImageFromArray(np.mean(img, axis=2).astype(np.uint8))
    overlay = sitk.LabelOverlay(img_gray, sitk.GetImageFromArray(mask))
    sitk.WriteImage(overlay, out)

if __name__ == '__main__':

    root_dir = '/home/james/QTIM/data/vessel_features/'
    orig_dir = join(root_dir, 'data/images')
    seg_dir = join(root_dir, 'data/vessels')
    out_dir = join(root_dir, 'output')

    vessel_features(orig_dir, seg_dir, out_dir)