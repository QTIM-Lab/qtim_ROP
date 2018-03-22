from qtim_ROP.utils.common import get_subdirs, find_images_by_class, dict_reverse
import numpy as np
from PIL import Image
from os.path import basename, splitext
import h5py


def images_to_hdf5(input_images, original_images, out_path, class_names=None):
    """
    Take one or more directories of images (split by class) and return a HDF5 file with fields 'data' and 'labels'
    :param input_images: director(y/ies) containing subdirectories of images, split by class
    :param out_path: full path of the HD5 file to save
    :param class_names: the set of classes (uses all subdirectories in image_dir, if None)
    """
    X, y, filenames, original_files = [], [], [], []

    if class_names is None:
        class_names = [basename(x) for x in get_subdirs(input_images)]

    imgs_by_class = find_images_by_class(input_images, classes=class_names)
    orig_by_class = find_images_by_class(original_images, classes=class_names)

    for class_name, imgs in list(imgs_by_class.items()):

        for i, src_img in enumerate(imgs):

            img_arr = np.asarray(Image.open(src_img))
            X.append(img_arr)
            y.append(class_name)

            orig_img = orig_by_class[class_name][i]
            assert(splitext(basename(src_img))[0] == splitext(basename(orig_img))[0])

            filenames.append(src_img)
            original_files.append(basename(orig_img))

    X = np.transpose(np.asarray(X), (0, 3, 2, 1))
    y = np.asarray(y)
    filenames = np.asarray(filenames)
    original_files = np.asarray(original_files)

    with h5py.File(out_path, "w") as f:
        f.create_dataset('data', data=X, dtype=X.dtype)
        f.create_dataset('labels', data=y, dtype=y.dtype)
        f.create_dataset('filenames', data=filenames, dtype=filenames.dtype)
        f.create_dataset('original_files', data=original_files, dtype=original_files.dtype)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--preprocessed', required=True, dest='preprocessed', help="Pre-processed images")
    parser.add_argument('-r', '--raw', required=True, dest='raw_images', help='Original (raw) images')
    parser.add_argument('-o', '--out-path', required=True, dest='h5_file', help="Output file (.h5)")

    args = parser.parse_args()

    images_to_hdf5(args.preprocessed, args.raw_images, args.h5_file)
