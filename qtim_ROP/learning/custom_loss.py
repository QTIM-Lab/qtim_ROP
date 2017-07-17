import theano
import theano.tensor as T
import numpy as np
import keras.backend as K


def quad_kappa_loss(y, t, y_pow=1, eps=1e-15):
    num_scored_items = y.shape[0]
    num_ratings = 3
    tmp = T.tile(T.arange(0, num_ratings).reshape((num_ratings, 1)),
                 reps=(1, num_ratings)).astype(theano.config.floatX)
    weights = (tmp - tmp.T) ** 2 / (num_ratings - 1) ** 2

    y_ = y ** y_pow
    y_norm = y_ / (eps + y_.sum(axis=1).reshape((num_scored_items, 1)))

    hist_rater_a = y_norm.sum(axis=0)
    hist_rater_b = t.sum(axis=0)

    conf_mat = T.dot(y_norm.T, t)

    nom = T.sum(weights * conf_mat)
    denom = T.sum(weights * T.dot(hist_rater_a.reshape((num_ratings, 1)),
                                  hist_rater_b.reshape((1, num_ratings))) /
                  num_scored_items.astype(theano.config.floatX))

    return - (1 - nom / denom)


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

if __name__ == '__main__':

    y_true = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.3, 0.4, 0.3]])
    quad_kappa_loss(y_true, y_pred)
