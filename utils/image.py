from os.path import isdir
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import SimpleITK as sitk
import cv2
import h5py
from utils.common import find_images


def overlay_mask(img, mask, out):

    img_gray = sitk.GetImageFromArray(np.mean(img, axis=2).astype(np.uint8))
    overlay = sitk.LabelOverlay(img_gray, sitk.GetImageFromArray(mask))
    sitk.WriteImage(overlay, out)
    return sitk.GetArrayFromImage(overlay)


def imgs_to_th_array(img_dir):

    img_arr = []

    for img_path in find_images(img_dir):

        # Load and prepare image
        img = cv2.imread(img_path)
        img = img.transpose((2, 0, 1))  # channels first
        img_arr.append(img)

    # Create single array of inputs
    img_arr = np.stack(img_arr, axis=0)  # samples, channels, rows, cols
    return img_arr


def create_generator(data_path, input_shape, batch_size=32, training=True):

    datagen = ImageDataGenerator()

    if isdir(data_path):  # if we have directories of images split by class

        dg = datagen.flow_from_directory(
            data_path,
            target_size=input_shape[1:],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=training)
        return dg, dg.classes, dg.class_indices

    else:  # otherwise, assume HDF5 file

        f = h5py.File(data_path, 'r')
        class_indices = {k: v for v, k in enumerate(np.unique(f['labels']))}
        classes = [class_indices[k] for k in f['labels']]
        labels = to_categorical(classes)

        return datagen.flow(f['data'], y=labels, batch_size=batch_size, shuffle=training), classes, class_indices

def normalize(img):

    img_min, img_max = np.min(img), np.max(img)
    img_norm = (img - img_min) / (img_max - img_min)
    return img_norm.astype(np.uint8) * 255