from os.path import join, dirname, basename
import cv2
from methods import binary_morph
from common import find_images


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-dir', dest='in_dir', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    parser.add_argument('-t', '--thresh', dest='thresh', help="0.0 < thresh < 1.0", type=float, default=.8)
    parser.add_argument('-s', '--smallest', dest='smallest', help="Smallest object size allowed", default=50)

    args = parser.parse_args()

    for im_path in find_images(args.in_dir):

        im_name = basename(im_path)
        img = cv2.imread(im_path)

        pruned = binary_morph(img, thresh=args.thresh, min_size=args.smallest)
        cv2.imwrite(join(args.out_dir, im_name), pruned)
