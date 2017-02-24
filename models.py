from os.path import split, join
from keras.models import model_from_json


def load_model(model):
    # Load model
    _, model_basename = split(model)
    model_arch = join(model, model_basename + '_architecture.json')
    model_weights = join(model, model_basename + '_best_weights.h5')

    model = model_from_json(open(model_arch).read())
    model.load_weights(model_weights)

