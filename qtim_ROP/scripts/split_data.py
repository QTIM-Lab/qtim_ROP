from os.path import basename, join
import numpy as np
from shutil import copy
from ..utils.common import get_subdirs, find_images, make_sub_dir


def split_data(in_dir, out_dir, n=5):

    # Count images in subdirs and verify equal amounts
    sub_dirs = get_subdirs(in_dir)

    image_lists = {basename(sub_dir): find_images(sub_dir, extensions=['*.gif']) for sub_dir in sub_dirs}
    no_imgs = np.asarray([len(x) for x in list(image_lists.values())])

    try:
        assert(np.array_equal(no_imgs, no_imgs))
    except AssertionError:
        print("Number of images in directories '{}' must be equal".format(list(image_lists.keys())))

    split_size = int(round(no_imgs[0] / float(n)))

    for i in range(0, n):

        # Create split directory
        split_dir = make_sub_dir(out_dir, 'split_{}'.format(i))

        for dir_name, image_list in list(image_lists.items()):

            sub_list = image_list[i * split_size:(i * split_size) + split_size]
            img_dir = make_sub_dir(split_dir, dir_name)

            for img in sub_list:
                copy(img, img_dir)


if __name__ == '__main__':

    import sys
    split_data(sys.argv[1], sys.argv[2])