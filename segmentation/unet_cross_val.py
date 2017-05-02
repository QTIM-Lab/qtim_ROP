import pandas as pd
from os.path import join, isfile, splitext, basename
from utils.common import find_images

def unet_cross_val(img_dir, mapping, splits):

    # Get images

    # Load spreadsheet and verify all images are present
    with pd.ExcelFile(mapping) as xls:
        df = pd.read_excel(xls, 'Sheet1').set_index('index')
        df['class'] = df['class'].map({'preplus': 'pre-plus', 'normal': 'normal', 'plus': 'plus'})

    for _, row in df.iterrows():

        img_path = join(img_dir, row['class'], row['image'])

        if not isfile(img_path):
            img_base, img_ext = splitext(row['image'])
            img_path = join(img_dir, row['class'], img_base + img_ext.upper())

        try:
            assert(isfile(img_path))
        except AssertionError:
            print '{} does not exist'.format(img_path)

def images_to_df(input_dir):

    # Get paths to all images
    im_files = find_images(join(input_dir, '*'))
    assert (len(im_files) > 0)

    # Split images into metadata
    imgs_split = [splitext(basename(x))[0].split('_') + [x] for x in im_files]
    imgs_split.sort(key=lambda x: x[1])  # sort by ID

    # Create DataFrame, indexed by ID
    return pd.DataFrame(imgs_split, columns=['patient_id', 'id', 'session', 'view', 'eye', 'class', 'full_path']).set_index('id')

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', dest='images', required=True, help="Folder of images split into No/Pre-Plus/Plus")
    parser.add_argument('-m', '--mapping', dest='mapping', required=True, help="Excel file defining the order of the images")
    parser.add_argument('-s', '--splits', dest='splits', required=True, help=".mat file containing the splits to be generated")

    args = parser.parse_args()
    unet_cross_val(args.images, args.mapping, args.splits)