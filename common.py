from glob import glob
from os.path import join
import h5py
from PIL import Image
import numpy as np


def find_images(im_path):

    files = []
    for ext in ['*.bmp', '*.BMP', '*.png', '*.jpg', '*.tif']:
        files.extend(glob(join(im_path, ext)))

    return sorted(files)





def imgs_to_unet_array(img_list):

    n_imgs = len(img_list)
    test_im = np.asarray(Image.open(img_list[0]))

    width, height, channels = test_im.shape

    imgs_arr = np.empty((n_imgs, width, height, channels))

    for i, im_path in enumerate(img_list):

        img = Image.open(im_path)
        imgs_arr[i] = np.asarray(img)

    imgs_arr = np.transpose(imgs_arr, (0, 3, 1, 2))
    print imgs_arr.shape

    return imgs_arr

def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


if __name__ == '__main__':

    im_list = find_images('/home/james/QTIM/DRIVE/training/images')
    res = imgs_to_unet_array(im_list)