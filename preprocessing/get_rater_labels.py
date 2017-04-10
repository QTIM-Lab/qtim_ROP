import pandas as pd
from collections import defaultdict


CLASS_LABELS = {'No': 0, 'Plus': 1, 'Pre-Plus': 2, 'unknown': None}


def get_rater_labels(orig_csv, curated_csv):

    orig_df = pd.DataFrame.from_csv(orig_csv, index_col=None)
    curated_df = pd.DataFrame.from_csv(curated_csv)

    readers = []

    for id_, img in curated_df.iterrows():

        orig_rows = orig_df[orig_df['posterior'] == img['posterior']]

        labels = pd.Series({row['reader']: CLASS_LABELS[row['Golden Reading Plus']] for _, row in orig_rows.iterrows()},
                           name=img['posterior'])
        readers.append(labels)

    reader_df = pd.DataFrame(readers)

    # Let's look at a subset
    three_readers = reader_df[READERS].dropna()
    print three_readers.shape[0]

if __name__ == '__main__':

    import sys

    get_rater_labels(sys.argv[1], sys.argv[2])