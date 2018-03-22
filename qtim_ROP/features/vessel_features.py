import pandas as pd
from glob import glob
from os.path import join, isfile

from skimage.morphology import remove_small_objects
from skimage.morphology import skeletonize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from ..evaluation.metrics import roc_auc
from ..features.tracing import VesselTree
from .geom import *
from ..utils.common import make_sub_dir, dict_reverse

CLASS_LABELS = {'normal': 0, 'pre-plus': 2, 'plus': 1}


def vessel_features(orig_dir, seg_dir, out_dir, csv_file):

    csv_data = pd.DataFrame.from_csv(csv_file)

    orig_images = [np.asarray(Image.open(x)) for x in sorted(glob(join(orig_dir, '*.*')))]
    seg_images = [np.asarray(Image.open(x)) for x in sorted(glob(join(seg_dir, '*.png')))]
    prob = .4

    renamed_dir = make_sub_dir(out_dir, 'renamed_images')
    vessel_dir = make_sub_dir(out_dir, 'mask')
    skel_dir = make_sub_dir(out_dir, 'skel')
    feature_dir = make_sub_dir(out_dir, 'features')

    all_features = []

    classifier_dir = make_sub_dir(out_dir, 'classification')
    feature_csv = join(classifier_dir, 'features.csv')
    ground_truth_csv = join(classifier_dir, 'ground_truth.csv')

    y = []

    if not isfile(feature_csv):

        for i, (orig, seg) in enumerate(zip(orig_images, seg_images)):

            # Extract row
            csv_row = csv_data.iloc[i]
            class_ = csv_row['class_name']
            od_center = csv_row['optic_disk_x'], csv_row['optic_disk_y']
            img_name = '{}_{}'.format(i, class_)
            y.append(CLASS_LABELS[class_])

            Image.fromarray(orig).save(join(renamed_dir, img_name + '.png'))

            print("'{}' belongs to class {}".format(img_name, CLASS_LABELS[class_]))

            # Binarize and overlay
            vessel_mask = (seg > (255 * prob)).astype(np.uint8)
            overlay_mask(orig, vessel_mask * 255, join(vessel_dir, img_name + '.png'))

            # Extract medial axis
            skel = skeletonize(vessel_mask)

            # Remove small isolated segments
            labelled = label(skel)
            cleaned_skel = remove_small_objects(labelled, min_size=100)
            cleaned_skel = cleaned_skel > 0
            overlay_mask(orig, cleaned_skel.astype(np.uint8) * 255, join(skel_dir, img_name + '.png'))

            # Create masked skeleton
            masked_skel = mask_od_vessels(cleaned_skel, od_center)

            # Compute vessel tree
            tree = VesselTree(orig[:, :, 1], masked_skel, od_center[::-1], feature_dir, img_name)
            branches, features = tree.run()

            # Get features of tree
            all_features.append(features)

        df_features = pd.DataFrame(all_features)
        df_features.to_csv(feature_csv)

        df_ground_truth = pd.DataFrame(data=to_categorical(y))
        df_ground_truth.to_csv(ground_truth_csv)

    else:
        df_features = pd.DataFrame.from_csv(feature_csv)
        df_ground_truth = pd.DataFrame.from_csv(ground_truth_csv)

    random_forest(df_features, df_ground_truth, classifier_dir)


def random_forest(df_features, df_ground_truth, out_dir, n_splits=5):

    X = df_features.as_matrix()
    y_true = np.argmax(df_ground_truth.as_matrix(), axis=1)

    print("~~ Class distribution ~~")
    for k, v in sorted(list(CLASS_LABELS.items()), key=lambda x: x[1]):
        print("{}: {:.2f}%".format(k.capitalize(), (len(y_true[y_true == v]) / float(len(y_true))) * 100))

    # Use stratified k-fold cross-validation
    skf = StratifiedShuffleSplit(n_splits=5, test_size=.2)

    auc_results = []
    for i, (train, test) in enumerate(skf.split(X, y_true)):

        X_train, y_train = X[train, :], y_true[train]
        rf = RandomForestClassifier(class_weight='balanced')
        rf.fit(X_train, y_train)

        X_test, y_test = X[test, :], y_true[test]
        y_pred_prob = rf.predict_proba(X_test)
        auc = roc_auc(y_pred_prob, to_categorical(y_test), dict_reverse(CLASS_LABELS),
                      join(out_dir, 'roc_auc_split_{}.svg'.format(i)))

        auc_results.append(auc['micro'])

    print("\n~~ Average AUC over {} splits ~~\n{}".format(n_splits, np.mean(auc_results)))

    #X_train, X_test, y_train, y_test = train_test_split(X, y_true, train_size=.7, test_size=.3, random_state=4)
    #
    # rf.fit(X_train, y_train)
    # # joblib.dump(rf, join(out_dir, 'classifier.pkl'))
    #
    # y_pred = rf.predict(X_test)
    #
    # print classification_report(y_true, y_pred)
    # print confusion_matrix(y_true, y_pred)
    # print accuracy_score(y_true, y_pred)
    #
    # return
    #
    # y_pred_prob = rf.predict_proba(X_test)
    # roc_auc(y_pred_prob, to_categorical(y_test), CLASS_LABELS, join(out_dir, 'roc_auc.svg'))


def normalize(img):

    img_min, img_max = np.min(img), np.max(img)
    img_norm = (img - img_min) / (img_max - img_min)
    return img_norm.astype(np.uint8) * 255


def plot_lines(img, lines, out, lw=2.0):

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    for line in lines:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), alpha=0.7, linewidth=lw)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()

if __name__ == '__main__':

    import sys

    root_dir = sys.argv[1]
    orig_dir = join(root_dir, 'images')
    seg_dir = join(root_dir, 'vessels')
    csv_file = join(root_dir, 'mapping.csv')
    out_dir = join(root_dir, 'analysis')

    vessel_features(orig_dir, seg_dir, out_dir, csv_file)