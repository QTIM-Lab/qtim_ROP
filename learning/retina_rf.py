import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from retina_net import RetiNet
from utils.image import imgs_to_th_array


class RetinaRF(object):

    def __init__(self, model_config, rf_pkl=None, feature_layer='flatten_3'):

        self.model_config = model_config
        self.cnn = RetiNet(model_config)
        self.feature_layer = feature_layer
        self.rf = None if rf_pkl is None else joblib.load(rf_pkl)

    def train(self, training_data, trees=100,rf_out=None):

        # Use CNN to extract features
        self.cnn.set_intermediate(self.feature_layer)
        features = self.extract_features(training_data)

        # Create random forest
        self.rf = RandomForestClassifier(n_estimators=trees, class_weight='balanced_subsample')
        X_train = features['y_pred']  # inputs to train the random forest
        y_train = np.asarray(features['y_true'])  # ground truth for random forest

        print "Training RF..."
        self.rf.fit(X_train, y_train)

        if rf_out:
            joblib.dump(self.rf, rf_out)

        return self.rf, X_train, y_train

    def predict(self, test_data):

        if self.rf is None:
            raise ValueError('RF not trained - call train first or pass RF pickle file in constructor')

        # Extract features using CNN
        img_arr = imgs_to_th_array(test_data)
        self.cnn.set_intermediate(self.feature_layer)
        features = self.cnn.model.predict_on_batch(img_arr)

        # Return raw predictions (probabilities)
        y_pred = self.rf.predict_proba(features)
        return y_pred

    def extract_features(self, img_data):

        features = self.cnn.predict(img_data)
        return features

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cnn', dest='cnn', help='CNN config file', required=True)
    parser.add_argument('-r', '--rf', dest='rf', help='RF pkl file', default=None)
    parser.add_argument('-i', '--imgs', dest='imgs', help='Images to predict', required=True)

    args = parser.parse_args()
    model = RetinaRF(args.cnn, rf_pkl=args.rf)
    model.predict(args.imgs)
