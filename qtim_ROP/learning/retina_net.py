#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from os import chdir, getcwd
from os.path import dirname, splitext, abspath
from itertools import cycle
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Input, Dropout
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.utils.np_utils import to_categorical
from googlenet_custom_layers import PoolHelper, LRN
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

from ..utils.common import *
from ..utils.image import create_generator
from ..utils.models import SGDLearningRateTracker
from ..utils.plotting import *
from ..visualisation.tsne import tsne
from ..evaluation.metrics import calculate_metrics


OPTIMIZERS = {'sgd': SGD, 'rmsprop': RMSprop, 'adadelta': Adadelta, 'adam': Adam}


class RetiNet(object):

    def __init__(self, conf_file):

        # Parse config
        self.conf_file = conf_file
        self.config = parse_yaml(conf_file)
        self.ext = self.config.get('ext', '.png')

        self.conf_dir = dirname(abspath(self.conf_file))
        self.experiment_name = splitext(basename(self.conf_file))[0]

        cwd = getcwd()
        chdir(self.conf_dir)
        self.train_data = abspath(self.config['training_data'])
        self.val_data = abspath(self.config['validation_data'])

        try:
            self.config['mode']
        except KeyError:
            print "Please specify a mode 'train' or 'evaluate' in the config file."
            exit()

        if self.config['mode'] == 'train':

            # Set up logging
            self.experiment_dir = make_sub_dir(self.conf_dir, self.experiment_name)
            self.eval_dir = make_sub_dir(self.experiment_dir, 'eval')

            setup_log(join(self.experiment_dir, 'training.log'), to_file=self.config.get('logging', False))
            logging.info("Experiment name: {}".format(self.experiment_name))
            self._configure_network(build=True)
            # plot(self.model, join(self.experiment_dir, 'final_model.png'))

        elif self.config['mode'] == 'evaluate':

            # Set up logging
            self.experiment_dir = self.conf_dir
            self.eval_dir = make_sub_dir(self.experiment_dir, 'eval')
            self._configure_network(build=False)

        chdir(cwd)  # revert to original working directory

    def _configure_network(self, build=True):

        network = self.config['network']
        type_, weights = network['type'].lower(), network.get('weights', None)
        fine_tuning = " with pre-trained weights '{}'".format(weights) if weights else " without pre-training"

        if 'vgg' in type_:

            from keras.applications.vgg16 import VGG16
            logging.info("Instantiating VGG model" + fine_tuning)
            self.model = VGG16(weights=weights, input_shape=(3, 227, 227), include_top=True)

        elif 'resnet' in type_:

            from keras.applications.resnet50 import ResNet50
            logging.info("Instantiating ResNet model" + fine_tuning)

            input_layer = Input(shape=(3, 224, 224))
            base_model = ResNet50(weights=weights, include_top=False, input_tensor=input_layer)

            x = base_model.output
            x = Flatten()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(3, activation='softmax')(x)

            self.model = Model(input=base_model.input, output=predictions)
            # for layer in base_model.layers:
            #     layer.trainable = fine_tuning

        else:

            if 'googlenet' in type_:
                custom_objects = {"PoolHelper": PoolHelper, "LRN": LRN}
                mod_str = 'GoogLeNet'
            else:
                custom_objects = {}
                mod_str = 'custom'

            logging.info("Instantiating {} model".format(mod_str) + fine_tuning)
            arch = network.get('arch', None)
            self.model = model_from_json(open(arch).read(), custom_objects=custom_objects)

            if weights:
                print "Loading weights '{}'".format(weights)
                self.model.load_weights(weights, by_name=True)

        # Configure optimizer
        if build:
            opt_options = self.config['optimizer']
            name, params = opt_options['type'], opt_options['params']
            optimizer = OPTIMIZERS[name](**params)
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):

        # Train
        epochs = self.config.get('epochs', 50)  # default to 50 if not specified
        input_shape = self.model.input_shape[1:]
        train_batch, val_batch = self.config.get('train_batch', 32), self.config.get('val_batch', 1)

        # Create generators
        train_gen, _, _ = create_generator(self.train_data, input_shape, training=True, batch_size=train_batch)
        val_gen, _, _ = create_generator(self.val_data, input_shape, training=False, batch_size=val_batch)

        # Check point callback saves weights on improvement
        weights_out = join(self.experiment_dir, 'best_weights.h5')
        checkpoint_tb = ModelCheckpoint(filepath=weights_out, verbose=1, save_best_only=True)
        lr_tb = SGDLearningRateTracker()

        logging.info("Training model for {} epochs".format(epochs))
        history = self.model.fit_generator(
            train_gen,
            samples_per_epoch=train_gen.x.shape[0],
            nb_epoch=epochs,
            validation_data=val_gen,
            nb_val_samples=val_gen.x.shape[0], callbacks=[checkpoint_tb, lr_tb])

        # Save model arch, weights and history
        dict_to_csv(history.history, join(self.experiment_dir, "history.csv"))
        np.save(join(self.experiment_dir, 'learning_rate.npy'), lr_tb.lr)
        self.model.save_weights(join(self.experiment_dir, 'final_weights.h5'))

        with open(join(self.experiment_dir, 'model_arch.json'), 'w') as arch:
            arch.writelines(self.model.to_json())

        # Create modified copy of config file
        conf_eval = self.update_config('final')
        with open(join(self.experiment_dir, self.experiment_name + '.yaml'), 'wb') as ce:
            yaml.dump(conf_eval, ce, default_flow_style=False)

        # Plot histories
        plot_accuracy(history.history, join(self.experiment_dir, 'accuracy' + self.ext))
        plot_loss(history.history, join(self.experiment_dir, 'loss' + self.ext))
        lr = np.load(join(self.experiment_dir, 'learning_rate.npy'))
        plot_LR(lr, join(self.experiment_dir, 'lr_plot' + self.ext))

    def update_config(self, weights='final'):

        conf_eval = dict(self.config)
        conf_eval['mode'] = 'evaluate'
        conf_eval['network']['arch'] = 'model_arch.json'
        conf_eval['network']['weights'] = '{}_weights.h5'.format(weights)

        conf_eval['training_data'] = abspath(self.config['training_data'])
        conf_eval['validation_data'] = abspath(self.config['validation_data'])
        return conf_eval

    def predict(self, img_arr):

        return self.model.predict_on_batch(img_arr)

    def evaluate(self, data_path, n_samples=None):

        logging.info("Evaluating model for on {}".format(data_path))
        datagen, y_true, class_indices = create_generator(data_path, self.model.input_shape[1:],
                                                          batch_size=1, training=False)
        if not n_samples:
            n_samples = datagen.x.shape[0]

        predictions = self.model.predict_generator(datagen, n_samples)
        data_dict = {'data': datagen, 'classes': class_indices, 'y_true': to_categorical(y_true[:n_samples]), 'y_pred': predictions}

        cols = np.asarray(sorted([[k, v] for k, v in class_indices.items()], key=lambda x: x[1]))
        # pred_df = pd.DataFrame(data=predictions, columns=cols[:, 0])
        # true_df = pd.DataFrame(data=to_categorical(y_true), columns=cols[:, 0])
        #
        # pred_df.to_csv(join(self.eval_dir, 'predictions.csv'))
        # true_df.to_csv(join(self.eval_dir, 'ground_truth.csv'))

        return data_dict

    def set_intermediate(self, layer_name):

        self.model = Model(input=self.model.input,
                           output=self.model.get_layer(name=layer_name).output)


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

    def predict(self, img_arr):

        if self.rf is None:
            raise ValueError('RF not trained - call train first or pass RF pickle file in constructor')

        # Extract features using CNN
        # img_arr = imgs_to_th_array(test_data)
        self.cnn.set_intermediate(self.feature_layer)
        features = self.cnn.model.predict_on_batch(img_arr)

        # Return raw predictions (probabilities)
        y_pred = self.rf.predict_proba(features)
        return y_pred

    def evaluate(self, data_path):

        # Get features
        self.cnn.set_intermediate(self.feature_layer)
        data_dict = self.extract_features(data_path)

        # Get RF predictions
        y_pred = self.rf.predict_proba(data_dict['y_pred'])
        data_dict['y_pred'] = y_pred
        return data_dict

    def extract_features(self, img_data):

        features = self.cnn.evaluate(img_data)
        return features


def locate_config(search_dir):

    try:
        config_file = glob(join(search_dir, '*.yaml'))[0]
        rf_pkl = glob(join(search_dir, '*.pkl'))[0]
    except IndexError:
        print "Missing CNN (.yaml) or RF (.pkl) file - unable to load"
        raise

    return config_file, rf_pkl

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-d', '--data', dest='data', default=None)

    args = parser.parse_args()

    # Instantiate model and train
    r = RetiNet(args.config)
    if args.data is None:
        r.train()
    else:
        # Evaluate on validation data and calculate metrics
        data_dict = r.evaluate(args.data)
        calculate_metrics(data_dict, out_dir=r.eval_dir)
