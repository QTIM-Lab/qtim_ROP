#!/usr/bin/env python

from os.path import isdir, isfile, join, basename, splitext, split
from common import find_images, imgs_to_unet_array
from models import load_model

try:
    from retinaunet.lib.help_functions import *
    from retinaunet.lib.extract_patches import *
    from retinaunet.lib.pre_processing import my_PreProc
except ImportError:
    print "Unable to import retinaunet - is it on your path?"
    exit()


def segment_unet(input_path, out_dir, unet_dir):

    # Load model
    model = load_model(unet_dir)

    # Get list of images to segment
    data = []
    if isdir(input_path):
        data.extend(find_images(input_path))
    elif isfile(input_path):
        data.append(input_path)
    else:
        raise IOError("Please specify a valid image path or folder of images")

    # Loop through chunks of the data, as there may be thousands of images to segment
    chunks = [data[x:x + 100] for x in xrange(0, len(data), 100)]

    for chunk_no, img_list in enumerate(chunks):

        print "Segmenting batch {} of {} ".format(chunk_no + 1, len(chunks))

        # Load images and create masks
        imgs_original, masks = imgs_to_unet_array(img_list)

        # Pre-process the images, and return as patches
        stride_x, stride_y = 6, 6
        img_patches, new_height, new_width, padded_masks = preprocess_images(imgs_original, masks, 48, 48, stride_x, stride_y)

        # Get predictions
        print "Running predictions..."
        predictions = model.predict(img_patches, batch_size=32, verbose=2)
        pred_imgs = pred_to_imgs(predictions)

        # Reconstruct images
        img_segs = recompone_overlap(pred_imgs, new_height, new_width, stride_x, stride_y)  # not sure about the stride widths

        for im_name, seg, mask in zip(img_list, img_segs, padded_masks):

            # Mask the segmentation and transpose
            seg[np.invert(mask.astype(np.bool))] = 0
            seg_T = np.transpose(seg, (1, 2, 0))

            # Save masked segmentation
            name, ext = splitext(basename(im_name))
            filename = join(out_dir, name + '_seg')
            print "Writing {}".format(filename)
            visualize(seg_T, filename)


def preprocess_images(imgs_original, masks, patch_height, patch_width, stride_height, stride_width):

    test_imgs = my_PreProc(imgs_original)

    # Pad images so they can be divided exactly by the patches dimensions
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    test_masks = paint_border_overlap(masks, patch_height, patch_width, stride_height, stride_width)

    # Extract patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', help="Image or folder of images", dest='images', required=True)
    parser.add_argument('-o', '--out-dir', help="Output directory", dest="out_dir", required=True)
    parser.add_argument('-u', '--unet', help='retina-unet dir', dest='model', required=True)
    args = parser.parse_args()

    segment_unet(args.images, args.out_dir, args.model)
