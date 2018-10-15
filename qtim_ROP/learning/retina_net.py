#!/usr/bin/env python

from os import chdir, getcwd
from os.path import dirname, splitext, abspath
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.optimizers import *
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# from ..deep_rop import generate_report
from ..utils.common import *
from ..utils.image import create_generator
from ..utils.plotting import *
from ..utils.keras_to_tensorflow import keras_to_tensorflow
from ..evaluation.metrics import plot_confusion, plot_ROC_curves, plot_PR_curves

# Set various TF training parameters
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))


class RetiNet(object):

    def __init__(self, conf_file, out_dir=None):

        # Parse config
        self.conf_file = conf_file
        self.config = parse_yaml(conf_file)
        self.ext = self.config.get('ext', '.png')

        self.model = None
        self.conf_dir = dirname(abspath(self.conf_file))
        self.experiment_name = splitext(basename(self.conf_file))[0]
        self.regression = self.config['network']['regression']
        self.training_samples = self.config.get('training_samples', None)

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
            logging.error("Please specify a mode 'train' or 'evaluate' in the config file.")
            exit()

        if self.config['mode'] == 'train':

            # Set up logging
            if not self.train_data:
                logging.warning("No training data specified! Exiting...")
                exit()

            if out_dir is None:
                self.experiment_dir = make_sub_dir(self.conf_dir, self.experiment_name)
            else:
                self.experiment_dir = out_dir

            self.eval_dir = make_sub_dir(self.experiment_dir, 'eval')

            if self.config.get('logging', False):
                print("Logging to '{}'".format(join(self.experiment_dir, 'training.log')))
            self.logger = setup_log(join(self.experiment_dir, 'training.log'), to_file=self.config.get('logging', False))
            logging.info("Experiment name: {}".format(self.experiment_name))
            self._configure_network(build=True)

        elif self.config['mode'] == 'evaluate':

            # Set up logging
            self.experiment_dir = self.conf_dir
            self.eval_dir = make_sub_dir(self.experiment_dir, 'eval')
            #self._configure_network(build=False)
            print("Loading best model")
            self.model = load_model(join(self.experiment_dir, 'best_model.h5'))

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

        elif 'inception' in type_:

            mod_str = 'Inception v1'
            from .inception_v1 import InceptionV1
            logging.info("Instantiating {} model".format(mod_str) + fine_tuning)
            self.customize_keras_model(InceptionV1, weights, network)

        elif 'resnet' in type_:

            from keras.applications.resnet50 import ResNet50
            logging.info("Instantiating ResNet model" + fine_tuning)
            self.customize_keras_model(ResNet50, weights, network)

        elif 'densenet' in type_:

            from keras.applications.densenet import DenseNet121
            logging.info("Instantiating DenseNet model" + fine_tuning)
            self.customize_keras_model(DenseNet121, weights, network)

        else:
            raise Exception('Invalid model type!')

        # Configure optimizer
        if build:
            opt_options = self.config['optimizer']
            opt_type, loss, params = opt_options['type'], opt_options['loss'], opt_options['params']
            metrics = ['accuracy']

            optimizers = {'sgd': SGD, 'rmsprop': RMSprop, 'adam': Adam, 'nadam': Nadam}

            try:
                OPT = optimizers[opt_type]
                optimizer = OPT(**params)
            except KeyError:
                print(f'Invalid optimizer type "{opt_type}". Defaulting to SGD with 1e-4 learning rate.')
                optimizer = SGD(lr=0.0001)
            except TypeError:
                print("One or more invalid optimizer parameters specified for '{opt_type}'. Using defaults")
                optimizer = optimizers[opt_type]()

            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def customize_keras_model(self, keras_model, weights, params):

        input_shape = params.get('input_shape', (224, 224, 3))
        dropout = params.get('dropout', 0.5)
        logging.info("Dropout rate = {}".format(dropout))

        loadable_weights = None
        if weights is not None and weights.endswith('.h5'):
            loadable_weights = weights
            weights = None

        print(loadable_weights, weights)
        base_model = keras_model(input_shape=input_shape, weights=weights, include_top=False)
        x = base_model.output
        x = Flatten()(x)

        if dropout:
            x = Dense(params.get('no_features'), activation='relu')(x)
            x = Dropout(dropout)(x)

        regression_mode = params.get('regression')
        if type(regression_mode) is str:
            act = 'sigmoid' if regression_mode == 'ordinal' else 'linear'
        else:
            act = 'linear' if regression_mode is True else 'softmax'
            
        # act = 'linear' if params.get('regression') is True else 'softmax'
        predictions = Dense(params.get('no_classes'), activation=act)(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

        if loadable_weights:
            self.model.load_weights(join(self.conf_dir, loadable_weights))

    def train(self):

        final_result = join(self.experiment_dir, 'final_weights.h5')
        if isfile(final_result):
            logging.warning("Training already concluded")
            self.conclude_training(weights='best' if self.val_data else 'final')
            return

        # Train
        logging.info("Training started")
        epochs = self.config.get('epochs', 50)  # default to 50 if not specified
        input_shape = self.model.input_shape[1:]
        train_batch, val_batch = self.config.get('train_batch', 32), self.config.get('val_batch', 1)

        # Create generators
        train_gen, train_n, _, cw = create_generator(self.train_data, input_shape, training=True, batch_size=train_batch,
                                                     regression=self.regression, subset=self.training_samples)

        if self.val_data is not None:
            val_gen, val_n, _, _ = create_generator(self.val_data, input_shape, training=False, batch_size=val_batch,
                                                    regression=self.regression)
        else:
            logging.info("No validation data provided.")
            val_gen = None
            val_n = 1

        # Check point callback saves weights on improvement
        best_weights = join(self.experiment_dir, 'best_weights.h5')
        tb_dir = make_sub_dir(self.experiment_dir, 'tensorboard')
        checkpoint_cb = ModelCheckpoint(filepath=best_weights, verbose=1, save_best_only=True, save_weights_only=True)
        stop_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=10)
        tensorboard_cb = TensorBoard(log_dir=tb_dir, histogram_freq=0, write_graph=True, write_images=True)
        csv_log_cb = CSVLogger(join(self.experiment_dir, 'history.csv'), separator=',', append=False)

        logging.info("Training model for {} epochs".format(epochs))
        history = self.model.fit_generator(
            train_gen,
            steps_per_epoch=train_n / float(train_batch),
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_n / float(val_batch),
            callbacks=[checkpoint_cb, csv_log_cb, stop_cb, tensorboard_cb], class_weight=cw)

        # Save model arch, weights and history
        dict_to_csv(history.history, join(self.experiment_dir, "history.csv"))
        # np.save(join(self.experiment_dir, 'learning_rate.npy'), lr_tb.lr)
        self.model.save_weights(join(self.experiment_dir, 'final_weights.h5'))
        self.model.save(join(self.experiment_dir, 'final_model.h5'))

        with open(join(self.experiment_dir, 'model_arch.json'), 'w') as arch:
            arch.writelines(self.model.to_json())

        # Write updated YAML file and plot history
        return self.conclude_training('best' if self.val_data else 'final')

    def conclude_training(self, weights='final'):

        # Create modified copy of config file
        conf_eval = self.update_config(weights=weights)
        with open(join(self.experiment_dir, self.experiment_name + '.yaml'), 'w') as ce:
            yaml.dump(conf_eval, ce, default_flow_style=False)
        self.plot_history()

        # Evaluate the model on the test/val data
        logging.info("Loading {} weights and saving model".format(weights))
        model_json = join(self.experiment_dir, 'model_arch.json')
        model_weights = join(self.experiment_dir, '{}_weights.h5'.format(weights))
        model_out = join(self.experiment_dir, '{}_model.h5'.format(weights))
        self.model.save(model_out)

        # Create TensorFlow graph
        logging.info("Generating pure TF model")
        keras_to_tensorflow(model_json, model_weights, self.experiment_dir)

        #if not isfile(join(self.experiment_dir, 'roc_curve.png')):
        logging.info("Final evaluation on test data")
        self.model.load_weights(model_weights)
        self.evaluate(self.val_data)

        self.logger.handlers = []

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

        if isfile(data_path) and splitext(data_path)[1] == '.h5':
            val_gen, val_n, y_true, _ = create_generator(self.val_data, self.model.input_shape[1:], training=False,
                                                         batch_size=1, regression=self.regression)
        else:
            raise IOError("Please specify either a HDF5 file or folder of images with which to evaluate.")

        # Get predictions and ground truth
        y_pred = np.squeeze(self.model.predict_generator(val_gen, steps=n_samples))
        y_true = np.asarray(y_true[:n_samples])

        pd.DataFrame(data=np.concatenate([y_true, y_pred], axis=1))\
            .to_csv(join(self.eval_dir, 'predictions.csv'))

        plt.figure(3)

        # Confusion matrix
        if self.regression:
            confusion = confusion_matrix(y_true, np.round(np.clip(y_pred, -0.5, 2.5)))
            plot_ROC_curves(y_true, y_pred, {0: 'Normal', 2: 'Plus'}, regression=True, outfile=join(self.experiment_dir, 'roc_curve.png'))
        else:
            confusion = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
            roc_aucs = plot_ROC_curves(y_true, y_pred, {0: 'Normal', 2: 'Plus'}, outfile=join(self.experiment_dir, 'roc_curve.png'))
            pr_aucs = plot_PR_curves(y_true, y_pred, {0: 'Normal', 2: 'Plus'}, outfile=join(self.experiment_dir, 'pr_curve.png'))

            df = pd.concat((pd.DataFrame(roc_aucs), pd.DataFrame(pr_aucs)))
            df.to_csv(join(self.experiment_dir, 'evaluation.csv'))

        plt.clf()
        plt.figure(4)
        plot_confusion(confusion, ['Normal', 'Pre-Plus', 'Plus'], join(self.experiment_dir, 'confusion.png'))
        plt.clf()

        return y_pred

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

        logging.info("Training RF...")
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


def locate_config(search_dir, rf=False):

    config_file, rf_pkl = None, None

    try:
        config_file = glob(join(search_dir, '*.yaml'))[0]
        if rf:
            rf_pkl = glob(join(search_dir, '*.pkl'))[0]
    except IndexError:
        print("Missing CNN (.yaml) or RF (.pkl) file - unable to load")

    return config_file, rf_pkl
