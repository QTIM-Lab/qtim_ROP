import pandas as pd
import numpy as np
from shutil import copy
from ..utils.common import make_sub_dir
from os.path import join, isfile


def filter_data(src_dir, dst_dir, mapping):

    df = pd.DataFrame.from_csv(mapping)

    # First, create a "view" column
    df['view'] = df['imageName'].str.split('_').str[3]

    # Next, eliminate grades 4 - 5, low quality and non-Posterior views
    df = df.loc[df['view'].isin(['Posterior']) & df['Golden Reading Stage'].isin([0, 1, 2, 3]) & df['quality']]

    # This is the filtered data we wish to use
    filtered = df[['posterior', 'Golden Reading Plus', 'imageName']]
    print({class_: group.shape[0] for class_, group in filtered.groupby('Golden Reading Plus')})

    for class_, group in filtered.groupby('Golden Reading Plus'):

        make_sub_dir(dst_dir, class_)

        for i, img_row in group.iterrows():

            src_img = join(src_dir, class_, img_row['imageName'])
            dst_img = join(dst_dir, class_, img_row['imageName'])
            if not isfile(src_img):
                print("'{}' is missing".format(src_img))
                exit()
            print("Copying '{}' to '{}'".format(src_img, dst_img))
            copy(src_img, dst_img)

if __name__ == '__main__':

    import sys
    filter_data(*sys.argv[1:])
