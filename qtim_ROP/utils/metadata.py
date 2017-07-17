#!/usr/bin/env python

from os.path import basename, dirname, splitext, join
import pandas as pd
import yaml
from qtim_ROP.utils.common import find_images, make_sub_dir

QUERY = 'subjectID == "{subjectID}" and eye == "{eye}" and reader == "{gold}" and Session == "{session}"'
VIEWS = ['posterior', 'nasal', 'temporal']

try:
    dir_name = dirname(__file__)
    conf_name = join(dir_name, 'config', 'conf.yaml')
    with open(conf_name) as y:
        conf = yaml.load(y)
    GOLD = conf['golden_reader']
except IOError:
    GOLD = None


def image_to_metadata(im_path):

    im_name = basename(im_path)
    im_str = splitext(im_name)[0]

    try:
        subject_id, im_id, session, view, eye, class_ = im_str.split('_')[:6]
    except ValueError:
        subject_id, _, im_id, session, view, eye, class_ = im_str.split('_')[:7]

    return {'imID': im_id, 'subjectID': subject_id, 'session': session, 'eye': eye, 'class': class_,
            'image': im_path, 'prefix': im_str, 'view': view}


def image_csv_data(im_path, csv_df):

    meta_dict = image_to_metadata(im_path)
    meta_dict.update({'gold': GOLD})
    image_view = meta_dict['view']

    row = csv_df.query(QUERY.format(**meta_dict)).iloc[[0]]

    for view in VIEWS:
        if view != image_view:
            del row[view]

    row.rename(columns={image_view: 'image'}, inplace=True)

    row['ID'] = meta_dict['imID']
    row['downloadFile'] = basename(im_path)
    return row, meta_dict


def unique_images(in_dir):

    unique = []
    im_df = pd.DataFrame(data=[image_to_metadata(im) for im in find_images(join(in_dir, '*'))])

    for imID, group in im_df.groupby('imID'):

        unique.append((group['class'].iloc[0], group['image'].iloc[0]))

    return unique


if __name__ == '__main__':

    import sys
    from shutil import copy

    for class_name, u_im in unique_images(sys.argv[1]):

        class_dir = make_sub_dir(sys.argv[2], class_name)
        copy(u_im, join(class_dir, basename(u_im)))
