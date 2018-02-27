#!/usr/bin/env python

from os import chdir, getcwd
from os.path import dirname, splitext, abspath
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from keras.layers import Dense, Flatten, Input, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.utils.np_utils import to_categorical
from googlenet_custom_layers import PoolHelper, LRN
from custom_loss import r2_keras
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from ..utils.common import *
from ..utils.image import create_generator
# from ..utils.models import SGDLearningRateTracker
from ..utils.plotting import *
from ..visualisation.tsne import tsne
from ..evaluation.metrics import plot_ROC_by_class


class RetiNet(object):

    def __init__(self, conf_file):

        # Parse config
        self.conf_file = conf_file
        self.config = parse_yaml(conf_file)
        self.ext = self.config.get('ext', '.png')

        self.model = None
        self.conf_dir = dirname(abspath(self.conf_file))
        self.experiment_name = splitext(basename(self.conf_file))[0]
        self.regression = self.config['network']['regression']

        cwd = getcwd()
        chdir(self.conf_dir)

        if self.config['training_data'] is not None:
            self.train_data = abspath(self.config['training_data'])
        else:
            self.train_data = None

        if self.config['validation_data'] is not None:
            self.val_data = abspath(self.config['validation_data'])
        else:
            self.val_data = None

        if self.config['test_data'] is not None:
            self.test_data = abspath(self.config['test_data'])
        else:
            self.test_data = None

        try:
            self.config['mode']
        except KeyError:
            print "Please specify a mode 'train' or 'evaluate' in the config file."
            exit()

        if self.config['mode'] == 'train':

            # Set up logging
            if not self.train_data:
                print "No training data specified! Exiting..."
                exit()

            self.experiment_dir = make_sub_dir(self.conf_dir, self.experiment_name)
            self.eval_dir = make_sub_dir(self.experiment_dir, 'eval')

            print "Logging to '{}'".format(join(self.experiment_dir, 'training.log'))
            setup_log(join(self.experiment_dir, 'training.log'), to_file=self.config.get('logging', False))
            logging.info("Experiment name: {}".format(self.experiment_name))
            self._configure_network(build=True)
            # plot(self.model, join(self.experiment_dir, 'final_model.png'))

        elif self.config['mode'] == 'evaluate':

            # Set up logging
            self.experiment_dir = self.conf_dir
            self.eval_dir = make_sub_dir(self.experiment_dir, 'eval')
            self._configure_network(build=False)

        self.history_file = join(self.experiment_dir, "history.csv")
        self.lr_file = join(self.experiment_dir, 'learning_rate.npy')
        chdir(cwd)  # revert to original working directory

    def _configure_network(self, build=True):

        network = self.config['network']
        type_, weights = network['type'].lower(), network.get('weights', None)
        fine_tuning = " with pre-trained weights '{}'".format(weights) if weights else " without pre-training"

        if 'vgg' in type_:

            from keras.applications.vgg16 import VGG16
            logging.info("Instantiating VGG model" + fine_tuning)
            self.customize_keras_model(VGG16, weights, network)

        elif 'mobile' in type_:

            from keras.applications.mobilenet import MobileNet
            logging.info("Instantiating Mobilenet model" + fine_tuning)
            self.customize_keras_model(MobileNet, weights, network)

        elif 'inception':

            custom_objects = {"PoolHelper": PoolHelper, "LRN": LRN}
            mod_str = 'GoogLeNet'

            from .googlenet import create_googlenet
            logging.info("Instantiating {} model".format(mod_str) + fine_tuning)
            arch = network.get('arch', None)

            if arch is None:
                self.model = create_googlenet(network.get('no_classes', 3), network.get('no_features', 128),
                                              network.get('regression'), network.get('input_shape', (3, 224, 224)))
            else:
                try:
                    self.model = model_from_json(open(arch).read(), custom_objects=custom_objects)
                except ValueError:  # keras compatibility issue
                    self.model = create_googlenet(network.get('no_classes', 3), network.get('no_features', 128),
                                                  network.get('regression'), network.get('input_shape', (3, 224, 224)))

            if weights:
                print "Loading weights '{}'".format(weights)
                self.model.load_weights(weights, by_name=True)

        else:
            raise Exception('Invalid model type!')

        # Configure optimizer
        if build:
            opt_options = self.config['optimizer']
            optimizer, loss, params = opt_options['type'], opt_options['loss'], opt_options['params']
            metrics = ['accuracy']
            if self.regression:
                metrics.append(r2_keras)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def customize_keras_model(self, keras_model, weights, params):

        base_model = keras_model(input_shape=(224, 224, 3), weights=weights, include_top=False)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(params.get('no_features'), activation='relu')(x)
        x = Dropout(0.5)(x)
        act = 'linear' if params.get('regression') is True else 'softmax'
        predictions = Dense(params.get('no_classes'), activation=act)(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

    def train(self):

        final_result = join(self.experiment_dir, 'final_weights.h5')
        if isfile(final_result):
            print "Training already concluded"
            self.conclude_training()
            return

        # Train
        logging.info("Training started")
        epochs = self.config.get('epochs', 50)  # default to 50 if not specified
        input_shape = self.model.input_shape[1:]
        train_batch, val_batch = self.config.get('train_batch', 32), self.config.get('val_batch', 1)

        # Create generators
        train_gen, train_n, _ = create_generator(self.train_data, input_shape, training=True, batch_size=train_batch,
                                                 regression=self.regression, tf=False)

        if self.val_data is not None:
            val_gen, val_n, _ = create_generator(self.val_data, input_shape, training=False, batch_size=val_batch,
                                                 regression=self.regression, tf=False)
        else:
            print "No validation data provided."
            val_gen = None
            val_n = None

        # Check point callback saves weights on improvement
        weights_out = join(self.experiment_dir, 'best_weights.h5')
        checkpoint_tb = ModelCheckpoint(filepath=weights_out, verbose=1, save_best_only=True)
        csv_log = CSVLogger(join(self.experiment_dir, 'history.csv'), separator=',', append=False)

        logging.info("Training model for {} epochs".format(epochs))
        history = self.model.fit_generator(
            train_gen,
            steps_per_epoch=train_n / float(train_batch),
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_n / float(val_batch),
            callbacks=[checkpoint_tb, csv_log])

        # Save model arch, weights and history
        dict_to_csv(history.history, join(self.experiment_dir, "history.csv"))
        # np.save(join(self.experiment_dir, 'learning_rate.npy'), lr_tb.lr)
        self.model.save_weights(join(self.experiment_dir, 'final_weights.h5'))

        with open(join(self.experiment_dir, 'model_arch.json'), 'w') as arch:
            arch.writelines(self.model.to_json())

        # Write updated YAML file and plot history
        return self.conclude_training()

    def conclude_training(self, weights='final'):

        # Create modified copy of config file
        conf_eval = self.update_config('final')
        with open(join(self.experiment_dir, self.experiment_name + '.yaml'), 'wb') as ce:
            yaml.dump(conf_eval, ce, default_flow_style=False)

        self.plot_history()

        # Evaluate the model on the test/val data
        print "Loading best weights and running inference..."
        self.model.load_weights(join(self.experiment_dir, '{}_weights.h5'.format(weights))) # final or best
        return self.evaluate(self.test_data)

    def plot_history(self):

        history = csv_to_dict(self.history_file)

        # Plot histories
        plot_accuracy(history, join(self.experiment_dir, 'accuracy' + self.ext))
        plot_loss(history, join(self.experiment_dir, 'loss' + self.ext))

    def update_config(self, weights='final'):

        conf_eval = dict(self.config)
        conf_eval['mode'] = 'evaluate'
        conf_eval['network']['arch'] = 'model_arch.json'
        conf_eval['network']['weights'] = '{}_weights.h5'.format(weights)

        conf_eval['training_data'] = self.train_data
        conf_eval['validation_data'] = self.val_data
        return conf_eval

    def predict(self, img_arr):

        return self.model.predict(img_arr, batch_size=100)

    def evaluate(self, data_path, n_samples=None):

        logging.info("Evaluating model for on {}".format(data_path))
        datagen, no_samples, y_true = create_generator(data_path, self.model.input_shape[1:],
                                                       batch_size=1, training=False, tf=False)
        if not n_samples:
            n_samples = no_samples

        # Get predictions and ground truth
        y_pred = self.model.predict_generator(datagen, n_samples)
        y_true = to_categorical(y_true[:n_samples])

        print y_pred
        print y_true

        # Confusion matrix
        confusion = confusion_matrix(np.argmax(y_true, axis=1),
                                     np.argmax(y_pred, axis=1))

        return y_pred, confusion, f1_score(y_true, y_pred > 0.5)

        # plt.figure(3)
        # plot_confusion(confusion, labels, join(self.experiment_dir, 'confusion.png'))
        # plt.clf()
        #
        # # ROC/AUC
        # class_indices.pop(1)  # remove Pre-Plus
        # plt.figure(4)
        # plot_ROC_by_class(y_true, y_pred, class_indices, outfile=join(self.experiment_dir, 'roc_curve.png'))
        # plt.clf()

    def set_intermediate(self, layer_name):

        self.model = Model(input=self.model.input,
                           output=self.model.get_layer(name=layer_name).output)


class RetinaRF(object):

    def __init__(self, model_config, rf_pkl=None, feature_layer='flatten_3'):

        self.model_config = model_config
        self.cnn = RetiNet(model_config)
        self.feature_layer = feature_layer
        self.rf = None if rf_pkl is None else joblib.load(rf_pkl)

    def train(self, training_data, trees=100, rf_out=None):

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

# class ROCCallback(Callback):
# 
#     def __init__(self, training_data, validation_data):
#         super(Roc).__init__
#         self.x = training_data[0]
#         self.y = training_data[1]
#         self.x_val = validation_data[0]
#         self.y_val = validation_data[1]
# 
#     def on_train_begin(self, logs={}):
#         return
# 
#     def on_train_end(self, logs={}):
#         return
# 
#     def on_epoch_begin(self, epoch, logs={}):
#         return
# 
#     def on_epoch_end(self, epoch, logs={}):
#         y_pred = self.model.predict(self.x)
#         roc = roc_auc_score(self.y, y_pred)
# 
#         y_pred_val = self.model.predict(self.x_val)
#         roc_val = roc_auc_score(self.y_val, y_pred_val)
# 
#         print(
#         '\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
#         return
# 
#     def on_batch_begin(self, batch, logs={}):
#         return
# 
#     def on_batch_end(self, batch, logs={}):
#         return

def locate_config(search_dir, rf=False):

    config_file, rf_pkl = None, None

    try:
        config_file = glob(join(search_dir, '*.yaml'))[0]
        if rf:
            rf_pkl = glob(join(search_dir, '*.pkl'))[0]
    except IndexError:
        print "Missing CNN (.yaml) or RF (.pkl) file - unable to load"

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
