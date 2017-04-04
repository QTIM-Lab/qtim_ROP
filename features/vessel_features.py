from PIL import Image
import numpy as np
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def vessel_features(orig_dir, seg_dir):

    orig_images = [np.asarray(Image.open(x)) for x in sorted(glob(join(orig_dir, '*.*')))]
    seg_images = [np.asarray(Image.open(x)) for x in sorted(glob(join(seg_dir, '*.gif')))]

    index = 0
    probability = .5
    test_orig = orig_images[index]
    test_seg = (seg_images[index] > 255 * probability).astype(np.uint8)

    overlay_mask(test_orig, test_seg)


def overlay_mask(img, mask):

    seg_masked = np.ma.masked_where(mask == 0, mask)

    plt.figure()
    plt.imshow(img)
    plt.imshow(seg_masked, interpolation='nearest', alpha=0.7)
    plt.show()

if __name__ == '__main__':

    root_dir = '/home/james/QTIM/data/vessel_features/'
    orig_dir = join(root_dir, 'data/images')
    seg_dir = join(root_dir, 'data/1st_manual')

    vessel_features(orig_dir, seg_dir)