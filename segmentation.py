#!/usr/bin/env python

from os import makedirs
from os.path import isdir, isfile, basename, splitext
from common import find_images, imgs_to_unet_array
from models import load_model
from mask_retina import *

try:
    from retinaunet.lib.help_functions import *
    from retinaunet.lib.extract_patches import *
    from retinaunet.lib.pre_processing import my_PreProc
except ImportError:
    print "Unable to import retinaunet - is it on your path?"
    exit()


class SegmentUnet(object):

    def __init__(self, out_dir, unet_dir, stride=(8, 8), erode=10):

        self.model = load_model(unet_dir)
        self.out_dir = out_dir
        if not isdir(out_dir):
            makedirs(out_dir)
        self.stride_x, self.stride_y = stride[0], stride[1]
        self.erode = erode
        self.patch_x, self.patch_y = 48, 48

    def segment_batch(self, img_data):

        # Loop through chunks of the data, as there may be thousands of images to segment
        data = [im for im in img_data if not isfile(join(self.out_dir, splitext(basename(im))[0] + '.png'))]

        print "{} image(s) already segmented - these will be skipped".format(len(img_data) - len(data))
        chunks = [data[x:x + 100] for x in xrange(0, len(data), 100)]

        for chunk_no, img_list in enumerate(chunks):

            print "Segmenting batch {} of {} ".format(chunk_no + 1, len(chunks))

            # Load images and create masks
            imgs_original, masks = imgs_to_unet_array(img_list, erode=self.erode)

            # Pre-process the images, and return as patches (TODO: get patch size from the model)
            img_patches, new_height, new_width, padded_masks = self.pre_process(imgs_original, masks)

            # Get predictions
            print "Running predictions..."
            predictions = self.model.predict(img_patches, batch_size=32, verbose=2)
            pred_imgs = pred_to_imgs(predictions)

            # Reconstruct images
            img_segs = recompone_overlap(pred_imgs, new_height, new_width, self.stride_x, self.stride_y)

            for im_name, seg, mask in zip(img_list, img_segs, padded_masks):

                # Mask the segmentation and transpose
                seg_masked = apply_mask(seg, mask)

                # Save masked segmentation
                name, ext = splitext(basename(im_name))
                filename = join(self.out_dir, name)
                print "Writing {}".format(filename)
                visualize(seg_masked, filename)

    def pre_process(self, imgs_original, masks):

        test_imgs = my_PreProc(imgs_original)

        # Pad images so they can be divided exactly by the patches dimensions
        test_imgs = paint_border_overlap(test_imgs, self.patch_x, self.patch_y, self.stride_x, self.stride_y)
        test_masks = paint_border_overlap(masks, self.patch_x, self.patch_y, self.stride_x, self.stride_y)

        # Extract patches from the full images
        patches_imgs_test = extract_ordered_overlap(test_imgs, self.patch_x, self.patch_y, self.stride_x, self.stride_y)

        return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks


def segment(im_arr, unet):  # static method

    assert(len(im_arr.shape) == 3)
    mask = create_mask(im_arr)

    im_arr = np.expand_dims(im_arr, 0).transpose((0, 3, 1, 2))
    im_mask = np.zeros((1, 1, mask.shape[0],  mask.shape[1]))
    im_mask[0, :, :, :] = np.expand_dims(mask, 0)

    img_patches, h, w, padded_mask = unet.pre_process(im_arr, im_mask)
    predictions = unet.model.predict(img_patches, batch_size=32, verbose=2)
    pred_imgs = pred_to_imgs(predictions)
    seg = recompone_overlap(pred_imgs, h, w, unet.stride_x, unet.stride_y)

    # Remove singleton dimensions and apply mask
    return apply_mask(seg[0], im_mask[0])

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', help="Image or folder of images", dest='images', required=True)
    parser.add_argument('-o', '--out-dir', help="Output directory", dest="out_dir", required=True)
    parser.add_argument('-u', '--unet', help='retina-unet dir', dest='model', required=True)
    parser.add_argument('-e', '--erode', help='Size of structuring element for mask erosion', dest='erode', type=int, default=10)
    parser.add_argument('-s', '--stride', help="Stride dimensions (width, height)", type=int, default=8)
    args = parser.parse_args()

    # Get list of images to segment
    data = []
    if isdir(args.images):
        data.extend(find_images(join(args.images)))
    elif isfile(args.images):
        data.append(args.images)
    else:
        raise IOError("Please specify a valid image path or folder of images")

    s = SegmentUnet(args.out_dir, args.model, stride=(args.stride, args.stride), erode=args.erode)
    s.segment_batch(data)
