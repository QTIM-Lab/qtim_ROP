#!/usr/bin/env python

from os.path import join
from os import listdir

COMBINED_CLASSES = {'Plus': '1', 'Pre-Plus': '1', 'No': '0'}


def two_class(data_path):

    txt_out = join(data_path, 'two_class.txt')

    with open(txt_out, 'w') as txt_file:

        for class_name, class_id in COMBINED_CLASSES.items():

            class_dir = join(data_path, class_name)
            class_num = COMBINED_CLASSES[class_name]

            print(class_dir)

            for im_path in listdir(class_dir):

                full_path = join(class_dir, im_path)
                # full_path = full_path.replace(...)
                txt_file.write('{} {}\n'.format(full_path, class_num))


if __name__ == '__main__':

    import sys
    data_path = sys.argv[1]
    two_class(data_path)
