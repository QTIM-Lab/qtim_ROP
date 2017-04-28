from os.path import isdir
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import SimpleITK as sitk
import h5py


def overlay_mask(img, mask, out):

    img_gray = sitk.GetImageFromArray(np.mean(img, axis=2).astype(np.uint8))
    overlay = sitk.LabelOverlay(img_gray, sitk.GetImageFromArray(mask))
    sitk.WriteImage(overlay, out)
    return sitk.GetArrayFromImage(overlay)


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