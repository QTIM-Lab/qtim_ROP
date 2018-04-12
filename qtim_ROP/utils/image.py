from os.path import isdir, basename
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import cv2
import h5py
from scipy.misc import imresize
from ..utils.common import find_images, find_images_by_class

DEFAULT_CLASSES = {'No': 0, 'Pre-Plus': 1, 'Plus': 2}

def overlay_mask(img, mask, out):

    # img_gray = sitk.GetImageFromArray(np.mean(img, axis=2).astype(np.uint8))
    # overlay = sitk.LabelOverlay(img_gray, sitk.GetImageFromArray(mask))
    # sitk.WriteImage(overlay, out)
    # return sitk.GetArrayFromImage(overlay)
    return NotImplementedError()


def imgs_to_th_array(img_dir, resize=(480, 640)):

    img_names, img_arr = [], []

    for img_path in find_images(img_dir):

        # Load and prepare image
        img = cv2.imread(img_path)
        if resize and img.shape != resize:
            try:
                img = imresize(img, resize, interp='bicubic')
            except ValueError as e:
                print('{} skipped due to error'.format(img_path))
                print(e)
                continue

        img_names.append(basename(img_path))
        img = img.transpose((2, 0, 1))  # channels first
        img_arr.append(img)

    # Create single array of inputs
    img_arr = np.stack(img_arr, axis=0)  # samples, channels, rows, cols
    return img_arr, img_names


def imgs_by_class_to_th_array(img_dir, class_labels):

    img_arr, img_names, y_true = [], [], []
    imgs_by_class = find_images_by_class(img_dir)

    for class_, img_list in list(imgs_by_class.items()):

        for img_path in img_list:

            # Load and prepare image
            img = cv2.imread(img_path)
            img = img.transpose((2, 0, 1))  # channels first
            img_arr.append(img)

            img_names.append(basename(img_path))
            y_true.append(class_labels[class_])

    # Create single array of inputs
    img_arr = np.stack(img_arr, axis=0)  # samples, channels, rows, cols
    return img_names, img_arr, y_true


def create_generator(data_path, input_shape, batch_size=32, training=True, regression=False, no_samples=None, **kwargs):

    datagen = ImageDataGenerator(**kwargs)  # augmentation options, can come from a dictionary

    if isdir(data_path):  # if we have directories of images split by class

        dg = datagen.flow_from_directory(
            data_path,
            target_size=input_shape[1:],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=training)
        return dg, dg.classes, dg.class_indices

    else:  # otherwise, assume HDF5 file

        print("Loading data from HDF5 file '{}'".format(data_path))
        f = h5py.File(data_path, 'r')
        data = f['data']

        # Check if in TF ordering
        if data.shape[3] != 3:
            print("Data was transposed to have channels last")
            data = np.transpose(data, (0, 2, 3, 1))

        # Convert string labels to numeric if necessary
        labels = list(f['labels'])

        try:
            labels = [DEFAULT_CLASSES[l.decode("utf-8")] for l in labels]
            print("Converted string labels to to integers")
            print(DEFAULT_CLASSES)
        except AttributeError:
            print("Numeric labels in range {} to {}".format(min(labels), max(labels)))

        if not regression:
            labels = to_categorical(labels)  # categorical (one-hot encoded)

        if no_samples is None:
            no_samples = data.shape[0]

        print("{} on {} samples".format('Training' if training else 'Testing', no_samples))
        cw = None  # cw = class_weight.compute_class_weight('balanced', np.unique(tuple(labels)), tuple(labels))
        return datagen.flow(data, y=labels, batch_size=batch_size, shuffle=training), f['data'].shape[0], labels, cw


def hdf5_images_and_labels(data_path):

    f = h5py.File(data_path, 'r')
    class_indices = {k: v for v, k in enumerate(np.unique(f['labels']))}
    classes = [class_indices[k] for k in f['labels']]
    labels = to_categorical(classes)
    return f['data'], labels, class_indices


def normalize(img):

    img_min, img_max = np.min(img), np.max(img)
    img_norm = (img - img_min) / (img_max - img_min)
    return img_norm.astype(np.uint8) * 255
