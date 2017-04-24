import cv2
from scipy.misc import imresize
import yaml
from utils.common import find_images
from os.path import join, basename
from PIL import Image


def preprocess(src_img, dst_img, params):

    print "Preprocessing {}".format(src_img)
    resize, crop = params['resize'], params['crop']
    crop_width = (resize['width'] - crop['width']) / 2
    crop_height = (resize['height'] - crop['height']) / 2

    # Resize, preprocess and augment
    try:
        im_arr = cv2.imread(src_img)[:, :, ::-1]
    except TypeError:
        print "Error loading '{}'".format(src_img)
        return False

    im_arr = im_arr[:,:,:3]

    # Resize and preprocess
    interp = resize.get('interp', 'bilinear')
    resized_im = imresize(im_arr, (resize['height'], resize['width']), interp=interp)
    cropped_im = resized_im[crop_height:-crop_height, crop_width:-crop_width]

    Image.fromarray(cropped_im).save(dst_img)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', help="Images to preprocess", dest='images', required=True)
    parser.add_argument('-o', '--out-dir', help="Output directory", dest='out_dir', required=True)
    parser.add_argument('-c', '--config', help="Configuration file", dest='conf', required=True)
    args = parser.parse_args()

    with open(args.conf, 'rb') as yam:
        params = yaml.load(yam)

    for src_img in find_images(join(args.images)):

        dst_img = join(args.out_dir, basename(src_img))
        preprocess(src_img, dst_img, params['pipeline'])