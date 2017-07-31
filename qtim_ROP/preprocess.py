

# Possible things we want to do with a set of images

# 1. Create a single dataset for training or testing
# 2. Create an arbitrary split of the dataset into training or testing
# 3. Generate multiple splits for cross-validation
#
# Options:
# * Which pre-processing pipeline to use (YAML file)
# * How much to split training/testing
# * Re-use old splits (CSV file) rather than generating random splits
# * What are the labels?

import sys
from os import listdir
from os.path import join, basename, splitext
import pandas as pd
from qtim_ROP.utils.common import find_images


def preprocess(data_csv, img_source, labels='plus', exclude_lq=True, exclude_ls=True):

    # Load the data into a DataFrame
    df = pd.DataFrame.from_csv(data_csv)

    # Perform exclusions
    if exclude_lq:  # exclude low quality
        df = df[df['quality'] == True]

    if exclude_ls:  # exclude late stage
        df = df[df['ROP_stage'] <= 3]

    # Get list of images and their names
    print img_source
    img_list = find_images(join(img_source, '*'))
    img_names = [splitext(basename(x))[0] for x in img_list]
    print img_names[0]

    # Check that all images in spreadsheet are present
    for x in df['imageName'].values:
        print splitext(x)[0]
        assert splitext(x)[0] in img_names


preprocess(sys.argv[1], sys.argv[2], exclude_ls=False)
