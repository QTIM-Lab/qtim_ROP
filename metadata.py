from os.path import basename, splitext, join, dirname
import pandas as pd
import yaml
from common import find_images

QUERY = 'subjectID == "{subjectID}" and eye == "{eye}" and reader == "{gold}" and Session == "{session}"'
VIEWS = ['posterior', 'nasal', 'temporal']


with open('config/conf.yaml') as y:
    conf = yaml.load(y)
GOLD = conf['golden_reader']


def image_to_metadata(im_path):

    im_name = basename(im_path)
    im_str = splitext(im_name)[0]
    subject_id, _, im_id, session, view, eye, class_ = im_str.split('_')[:7]
    return {'imID': im_id, 'subjectID': subject_id, 'session': session, 'eye': eye, 'class': class_,
            'image': im_path, 'prefix': im_str, 'view': view.lower()}


def image_csv_data(im_path, csv_df):

    meta_dict = image_to_metadata(im_path)
    meta_dict.update({'gold': GOLD})
    image_view = meta_dict['view']

    rows = csv_df.query(QUERY.format(**meta_dict))

    for view in VIEWS:
        if view != image_view:
            del rows[view]

    rows.rename(columns={image_view: 'image'}, inplace=True)

    rows['ID'] = meta_dict['imID']
    rows['downloadFile'] = basename(im_path)
    return rows

if __name__ == '__main__':

    import sys
    csv_data = pd.DataFrame.from_csv(sys.argv[2], index_col=None)
    csv_data['Session'] = csv_data['Session'].apply(lambda x: str(x).split()[-1])

    rows = []
    for im_path in find_images(join(sys.argv[1], '*')):

        row = image_csv_data(im_path, csv_data)
        rows.append(row)

    df = pd.DataFrame(data=pd.concat(rows))
    df = df.set_index('ID').sort_index()
    df.to_csv(join(sys.argv[1], 'downloaded.csv'))
