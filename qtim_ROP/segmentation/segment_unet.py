#!/usr/bin/env python

from os import makedirs
from os.path import isdir, isfile, basename

from .prepare_unet_data import imgs_to_unet_array
from .mask_retina import *
from ..utils.common import find_images
from ..utils.models import load_model
import h5py
import csv

try:
    from ..retinaunet.lib.help_functions import *
    from ..retinaunet.lib.extract_patches import *
    from ..retinaunet.lib.pre_processing import my_PreProc
except ImportError:
    print("Unable to import retinaunet - git clone https://github.com/QTIM-Lab/retinaunet.git")
    raise


class SegmentUnet(object):

    def __init__(self, unet_dir, out_dir=None, stride=(8, 8), erode=10, ext='.png'):

        self.model = load_model(unet_dir)
        self.out_dir = out_dir
        if out_dir and not isdir(out_dir):
            makedirs(out_dir)
        self.stride_x, self.stride_y = stride[0], stride[1]
        self.erode = erode
        self.patch_x, self.patch_y = 48, 48
        self.ext = ext

    def segment_batch(self, img_data, batch_size=100):

        # Lists to keep track of images already segmented, to segment, and failed
        newly_segmented, already_segmented, failures = [], [], []

        if self.out_dir is None:
            to_segment = [im for im in img_data]
        else:
            to_segment = [im for im in img_data if not isfile(join(self.out_dir, splitext(basename(im))[0] + self.ext))]
            already_segmented = [join(self.out_dir, splitext(basename(im))[0] + self.ext)
                                 for im in img_data if im not in to_segment]
            print("{} image(s) already segmented".format(len(already_segmented)))

        # Split into chunks of size batch_size
        chunks = [to_segment[x:x + batch_size] for x in range(0, len(to_segment), batch_size)]

        for chunk_no, img_list in enumerate(chunks):

            print("Segmenting batch {} of {} ".format(chunk_no + 1, len(chunks)))

            # Load images and create masks
            imgs_original, masks, skipped = imgs_to_unet_array(img_list, erode=self.erode)
            failures.extend(skipped)

            # Pre-process the images, and return as patches (TODO: get patch size from the model)
            img_patches, new_height, new_width, padded_masks = self.pre_process(imgs_original, masks)

            # Get predictions
            print("Running predictions...")
            predictions = self.model.predict(img_patches, batch_size=32, verbose=2)

            pred_imgs = pred_to_imgs(predictions, self.patch_x, self.patch_y)

            # Reconstruct images
            img_segs = recompone_overlap(pred_imgs, new_height, new_width, self.stride_x, self.stride_y)

            for im_name, seg, mask in zip(img_list, img_segs, padded_masks):

                # Mask the segmentation and transpose
                seg_masked = apply_mask(seg, mask)

                print(seg_masked.shape)

                # Save masked segmentation
                if self.out_dir is not None:
                    name, ext = splitext(basename(im_name))
                    filename = join(self.out_dir, name) + self.ext
                    newly_segmented.append(filename)
                    # print "Writing {}".format(filename)
                    # Save masked segmentation
                    vessel_img = (seg_masked * 255).astype(np.uint8).reshape((seg_masked.shape[0], seg_masked.shape[1]))
                    Image.fromarray(vessel_img).save(filename)

        if len(failures) > 0 and self.out_dir is not None:
            with open(join(self.out_dir, 'invalid.csv')) as f_inv:
                writer = csv.writer(f_inv)
                for img in failures:
                    writer.writerow([img])

        return newly_segmented, already_segmented, failures

    def pre_process(self, imgs_original, masks):

        test_imgs = my_PreProc(imgs_original)

        # Pad images so they can be divided exactly by the patches dimensions
        test_imgs = paint_border_overlap(test_imgs, self.patch_x, self.patch_y, self.stride_x, self.stride_y)
        test_masks = paint_border_overlap(masks, self.patch_x, self.patch_y, self.stride_x, self.stride_y)

        # Extract patches from the full images
        patches_imgs_test = extract_ordered_overlap(test_imgs, self.patch_x, self.patch_y, self.stride_x, self.stride_y)

        return patches_imgs_test, test_imgs.shape[1], test_imgs.shape[2], test_masks


def segment(im_arr, unet):  # static method

    assert(len(im_arr.shape) == 3)
    mask = circular_mask(im_arr)[:,:,0].astype(np.uint8) * 255

    im_arr = np.expand_dims(im_arr, 0).transpose((0, 3, 1, 2))
    im_mask = np.zeros((1, 1, mask.shape[0],  mask.shape[1]))
    im_mask[0, :, :, :] = np.expand_dims(mask, 0)

    img_patches, h, w, padded_mask = unet.pre_process(im_arr, im_mask)
    predictions = unet.model.predict(img_patches, batch_size=32, verbose=2)
    pred_imgs = pred_to_imgs(predictions, unet.patch_x, unet.patch_y)
    seg = recompone_overlap(pred_imgs, h, w, unet.stride_x, unet.stride_y)

    # Remove singleton dimensions and apply mask
    return apply_mask(seg[0], im_mask[0])

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', help="Image or folder of images", dest='images', required=True)
    parser.add_argument('-o', '--out-dir', help="Output directory", dest="out_dir", default=None)
    parser.add_argument('-u', '--unet', help='retina-unet dir', dest='model', required=True)
    parser.add_argument('-e', '--erode', help='Size of structuring element for mask erosion', dest='erode', type=int, default=10)
    parser.add_argument('-s', '--stride', help="Stride dimensions (width, height)", type=int, default=8)
    args = parser.parse_args()

    unet = SegmentUnet(args.model, out_dir=args.out_dir, stride=(args.stride, args.stride), erode=args.erode)

    # Get list of images to segment
    data = []
    if isdir(args.images):

        results = unet.segment_batch(find_images(args.images))

        if results:
            results = np.asarray(results).transpose((1, 2, 0))

    elif isfile(args.images):
        seg_result = segment(np.asarray(Image.open(args.images)), unet)

        if args.out_dir:
            visualize(seg_result, join(args.out_dir, basename(args.images)))
    else:
        raise IOError("Please specify a valid image path or folder of images")
