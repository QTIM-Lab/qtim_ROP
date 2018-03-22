from os.path import basename, splitext, join
from ..utils.common import find_images, make_sub_dir
from PIL import Image
import numpy as np
from shutil import copy
from scipy.misc import imresize


def prepare_GAN(img_dir, mask_dir, out_dir, thresh=0.4):

    imgs = {splitext(basename(img))[0]: img for img in find_images(img_dir)}
    seg_imgs = {splitext(basename(seg_img))[0]: seg_img for seg_img in find_images(mask_dir)}

    train, val, test = int(len(seg_imgs) * .7), int(len(seg_imgs) * .2), int(len(seg_imgs) * .1)

    train_dir =  make_GAN_dirs(out_dir, 'train')
    val_dir = make_GAN_dirs(out_dir, 'val')
    test_dir = make_GAN_dirs(out_dir, 'test')

    i = 0
    for key, seg_img in list(seg_imgs.items()):

        if key in list(imgs.keys()):

            img_arr = np.asarray(Image.open(imgs[key]))
            img_arr = imresize(img_arr, (512, 512), interp='bicubic')

            seg_arr = np.asarray(Image.open(seg_img))
            seg_arr = imresize(seg_arr, (512, 512), interp='bicubic')
            seg_arr = (seg_arr > (255 * thresh)).astype(np.uint8) * 255

            if i < train:
                sub_dir = train_dir
            elif i < train + val:
                sub_dir = val_dir
            else:
                sub_dir = test_dir

            img_out = join(sub_dir, 'A', '{}.png'.format(i))
            seg_out = join(sub_dir, 'B', '{}.png'.format(i))

            Image.fromarray(img_arr).save(img_out)
            Image.fromarray(seg_arr).save(seg_out)

            i += 1


def make_GAN_dirs(out_dir, split):

    split_dir = make_sub_dir(out_dir, split)
    make_sub_dir(split_dir, 'A')
    make_sub_dir(split_dir, 'B')

    return split_dir

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', dest='images', required=True)
    parser.add_argument('-m', '--masks', dest='masks', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()

    prepare_GAN(args.images, args.masks, args.out_dir)