import numpy as np
from PIL import Image
import pandas as pd
import yaml
from os.path import join, dirname, abspath
from utils.common import get_subdirs, find_images, imgs_and_labels_to_hdf5
from metadata import image_to_metadata


def get_unique(rater_dir, image_dir, csv_file, out_dir):

    conf_name = abspath(join(dirname(dirname(__file__)), 'config/conf.yaml'))
    with open(conf_name) as y:
        conf = yaml.load(y)
    gold = conf['golden_reader']

    csv_data = pd.DataFrame.from_csv(csv_file)
    img_list = []

    for reader_dir in get_subdirs(rater_dir):

        val_dir = join(reader_dir, 'validation')
        val_imgs = find_images(join(val_dir, '*'))

        for img in val_imgs:
            meta_dict = image_to_metadata(img)
            meta_dict.update({'path': img})
            img_list.append(meta_dict)

    df = pd.DataFrame(img_list).sort_values(by=['imID'])
    df = df.drop_duplicates(subset=['imID'])[['imID', 'path', 'class']]
    print df

    by_class = {class_: len(group) for class_, group in df.groupby('class')}
    print by_class

    all_imgs = []
    classes = []

    for i, unique_image in df.iterrows():

        # Get golden reader class
        csv_row = csv_data.iloc[i]
        golden_reader_class = csv_row[gold]
        classes.append(golden_reader_class)

        # Load image
        print unique_image['path']
        im_arr = np.asarray(Image.open(unique_image['path']))
        all_imgs.append(im_arr)

    data = np.transpose(np.asarray(all_imgs), (0, 3, 2, 1))
    imgs_and_labels_to_hdf5(data, classes, join(out_dir, 'test.h5'))

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-r', '--rater-dir', dest='rater_dir', required=True)
    parser.add_argument('-i', '--image-dir', dest='image_dir', required=True)
    parser.add_argument('-c', '--csv-file', dest='csv_file', required=True)
    parser.add_argument('-o', '--output-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    get_unique(args.rater_dir, args.image_dir, args.csv_file, args.out_dir)