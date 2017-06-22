from ..utils.common import get_subdirs, make_sub_dir, find_images
from shutil import copy, copytree
from os.path import join


def leave_split_out(in_dir, out_dir):

    # Get list of all splits
    splits_list = sorted(get_subdirs(in_dir))

    # For each split, create a dataset that excludes it
    for i, split_i in enumerate(splits_list):

        split_out = make_sub_dir(out_dir, 'leave_{}'.format(i), tree=split_i)

        for j, split_j in enumerate(splits_list):

            if i == j:
                continue

            for img_type in ['images', '1st_manual', 'mask']:

                copy_images(join(split_j, 'training'), join(split_out, 'training'), img_type)
                copy_images(join(split_j, 'test'), join(split_out, 'test'), img_type)


def copy_images(in_dir, out_dir, name):

    for src_img in find_images(join(in_dir, name), extensions=['*.gif']):
        copy(src_img, join(out_dir, name))


if __name__ == '__main__':

    import sys
    leave_split_out(sys.argv[1], sys.argv[2])
