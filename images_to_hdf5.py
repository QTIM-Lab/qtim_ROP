from utils.common import get_subdirs, find_images_by_class, dict_reverse
import numpy as np
from PIL import Image
from os.path import basename, join
import h5py
from scipy.misc import imresize


def images_to_hdf5(image_dirs, out_path, shape, class_names=None):
    """
    Take one or more directories of images (split by class) and return a HDF5 file with fields 'data' and 'labels'
    :param image_dir: director(y/ies) containing subdirectories of images, split by class
    :param out_path: full path of the HD5 file to save
    :param class_names: the set of classes (uses all subdirectories in image_dir, if None)
    """
    X, y, filenames = [], [], []

    for image_dir in image_dirs:

        if class_names is None:
            class_names = [basename(x) for x in get_subdirs(image_dir)]

        imgs_by_class = find_images_by_class(image_dir, classes=class_names)

        for class_name, imgs in imgs_by_class.items():

            for src_img in imgs:

                img_arr = np.asarray(Image.open(src_img))

                if img_arr.shape[-1] > 3:  # rgba...
                    img_arr = img_arr[:, :, :3]

                if img_arr.shape != shape:
                    img_arr = imresize(img_arr, tuple(shape), interp='bicubic')

                X.append(img_arr)
                y.append(class_name)
                filenames.append(basename(src_img))

    X = np.transpose(np.asarray(X), (0, 3, 2, 1))
    y = np.asarray(y)
    filenames = np.asarray(filenames)

    with h5py.File(out_path, "w") as f:
        f.create_dataset('data', data=X, dtype=X.dtype)
        f.create_dataset('labels', data=y, dtype=y.dtype)
        f.create_dataset('filenames', data=filenames, dtype=filenames.dtype)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--image-dirs', nargs='*', required=True, dest='image_dirs', help="Image directories")
    parser.add_argument('-o', '--out-path', required=True, dest='h5_file', help="Output file (.h5)")
    parser.add_argument('-s', '--shape', default=(640, 480), dest='shape', help="Output file (.h5)")

    args = parser.parse_args()

    images_to_hdf5(args.image_dirs, args.h5_file, args.shape)
