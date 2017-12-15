import keras.backend as K


def r2_keras(y_true, y_pred):

    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )
