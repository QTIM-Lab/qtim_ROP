from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Reshape, MaxoutDense
from keras.layers import Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1l2
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils.visualize_util import plot
from keras.models import Model


def unet():

    n_chan = 3
    drop_freq = .0
    reg_init1 = 0
    reg_init2 = 0

    input_img = Input((n_chan, 256, 256))

    A1 = Convolution2D(64, 3, 3, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(input_img)
    A2 = BatchNormalization(mode=0, axis=1)(A1)
    A3 = Dropout(drop_freq)(A2)
    A4 = Convolution2D(64, 3, 3, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(A3)
    A5 = BatchNormalization(mode=0, axis=1)(A4)
    A6 = Dropout(drop_freq)(A5)
    A7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(A6)

    B1 = Convolution2D(128, 3, 3, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(A7)
    B2 = BatchNormalization(mode=0, axis=1)(B1)
    B3 = Dropout(drop_freq)(B2)
    B4 = Convolution2D(128, 3, 3, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(B3)
    B5 = BatchNormalization(mode=0, axis=1)(B4)
    B6 = Dropout(drop_freq)(B5)
    B7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(B6)

    C1 = Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(B7)
    C2 = BatchNormalization(mode=0, axis=1)(C1)
    C3 = Dropout(drop_freq)(C2)
    C4 = Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(C3)
    C5 = BatchNormalization(mode=0, axis=1)(C4)
    C6 = Dropout(drop_freq)(C5)
    C7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(C6)

    D1 = Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(C7)
    D2 = BatchNormalization(mode=0, axis=1)(D1)
    D3 = Dropout(drop_freq)(D2)
    D4 = Convolution2D(512, 6, 6, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(D3)
    D5 = BatchNormalization(mode=0, axis=1)(D4)
    D6 = Dropout(drop_freq)(D5)

    E1 = merge([UpSampling2D(size=(2, 2))(D6), Cropping2D(cropping=((7, 8), (7, 8)))(C6)], mode='concat', concat_axis=1)
    E2 = Convolution2D(256, 6, 6, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(E1)
    E3 = BatchNormalization(mode=0, axis=1)(E2)
    E4 = Dropout(drop_freq)(E3)
    E5 = Convolution2D(256, 6, 6, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(E4)
    E6 = BatchNormalization(mode=0, axis=1)(E5)
    E7 = Dropout(drop_freq)(E6)

    F1 = merge([UpSampling2D(size=(3, 3))(E7), Cropping2D(cropping=((13, 13), (13, 13)))(B6)], mode='concat', concat_axis=1)
    F2 = Convolution2D(128, 6, 6, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(F1)
    F3 = BatchNormalization(mode=0, axis=1)(F2)
    F4 = Dropout(drop_freq)(F3)
    F5 = Convolution2D(128, 6, 6, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(F4)
    F6 = BatchNormalization(mode=0, axis=1)(F5)
    F7 = Dropout(drop_freq)(F6)

    G1 = merge([UpSampling2D(size=(3, 3))(F7), ZeroPadding2D(padding=(3, 3))(A6)], mode='concat', concat_axis=1)
    G2 = Convolution2D(64, 6, 6, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(G1)
    G3 = BatchNormalization(mode=0, axis=1)(G2)
    G4 = Dropout(drop_freq)(G3)
    G5 = Convolution2D(64, 6, 6, activation='relu', W_regularizer=l1l2(l1=reg_init1, l2=reg_init2),
                       W_constraint=maxnorm(2))(G4)
    G6 = BatchNormalization(mode=0, axis=1)(G5)
    G7 = Dropout(drop_freq)(G6)
    G8 = MaxPooling2D(pool_size=(113, 113), strides=(1, 1), border_mode='valid')(G7)
    lastflatten = Flatten()(G8)
    lastdense = Dense(1, activation='sigmoid')(lastflatten)

    model = Model(input=input_img, output=lastdense)

    lr = 1e-4
    model.compile(optimizer=SGD(lr=lr, decay=0.05, momentum=0.9), loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model


# # os.chdir('/mnt/eminas01/sharedfolder/LGG_DeepLearning/Compiled_TCGA')
# os.chdir('/root/sharedfolder/LGG_DeepLearning/Compiled_TCGA')
# slices = np.load('slices25D.npy')
# labels = np.load('labels25D.npy')
#
# batch_size = 16
#
# label_cat = labels
# idx = range(len(labels) / 3)
# labels_compressed = np.reshape(labels, (len(labels) / 3, 3))
# labels_compressed = labels_compressed[:, 0]
# idx_n = np.where(labels_compressed == 0)[0]
# idx_p = np.where(labels_compressed == 1)[0]
# np.random.seed(1)
# np.random.shuffle(idx_n)
# np.random.shuffle(idx_p)
# train_idx_n = idx_n[0:int(round(len(idx_n) * .8))]
# val_idx_n = idx_n[int(round(len(idx_n) * .8)):]
# train_idx_p = idx_p[0:int(round(len(idx_p) * .8))]
# val_idx_p = idx_p[int(round(len(idx_p) * .8)):]
# train_idx = np.hstack([train_idx_n * 3, train_idx_n * 3 + 1, train_idx_n * 3 + 2, train_idx_p * 3, train_idx_p * 3 + 1,
#                        train_idx_p * 3 + 2])
# val_idx = np.hstack(
#     [val_idx_n * 3, val_idx_n * 3 + 1, val_idx_n * 3 + 2, val_idx_p * 3, val_idx_p * 3 + 1, val_idx_p * 3 + 2])
# X_train = slices[train_idx]
# Y_train = label_cat[train_idx]
# X_val = slices[val_idx]
# Y_val = label_cat[val_idx]
#
# val_categorical_accuracy = np.empty
# loss = np.empty
# categorical_accuracy = np.empty
# val_loss = np.empty
#
# hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, validation_data=(X_val, Y_val))
#
# train_scores = model.evaluate(X_train, Y_train, batch_size=16)
# print(train_scores)
#
# loss = train_scores[0]
# categorical_accuracy = train_scores[1]
# val_loss = np.asarray(hist.history['val_loss'])
# val_categorical_accuracy = np.asarray(hist.history['val_binary_accuracy'])
# best_accuracy = val_categorical_accuracy[0]
# best_loss = val_loss[0]
#
# # Apply data augmentation
# rotation_range = 180
# height_shift_range = 0.05
# width_shift_range = 0.05
# channel_axis = 0
# fill_mode = 'wrap'
# cval = 0.
# folds = 199
#
#
# def transform_matrix_offset_center(matrix, x, y):
#     o_x = float(x) / 2 + 0.5
#     o_y = float(y) / 2 + 0.5
#     offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
#     reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
#     transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
#     return transform_matrix
#
#
# def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='wrap', cval=0.):
#     x = np.rollaxis(x, channel_axis, 0)
#     final_affine_matrix = transform_matrix[:2, :2]
#     final_offset = transform_matrix[:2, 2]
#     channel_images = [
#         ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=0, mode=fill_mode,
#                                            cval=cval) for x_channel in x]
#     x = np.stack(channel_images, axis=0)
#     x = np.rollaxis(x, 0, channel_axis + 1)
#     return x
#
#
# def flip_axis(x, axis):
#     x = np.asarray(x).swapaxes(axis, 0)
#     x = x[::-1, ...]
#     x = x.swapaxes(0, axis)
#     return x
#
#
# for iter in range(folds):
#
#     print(iter)
#     X_train_aug = deepcopy(X_train)
#
#     for p in range(len(X_train)):
#         for m in range(0, 3):
#             theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
#             rotation_matrix = np.array(
#                 [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
#             tx = np.random.uniform(-height_shift_range, height_shift_range) * X_train.shape[2]
#             ty = np.random.uniform(-width_shift_range, width_shift_range) * X_train.shape[3]
#             translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
#             transform_matrix = np.dot(rotation_matrix, translation_matrix)
#             h, w = X_train.shape[2], X_train.shape[3]
#             transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
#             X_train_aug[p][range(m * n_modal, m * n_modal + n_modal)] = apply_transform(
#                 X_train_aug[p][range(m * n_modal, m * n_modal + n_modal)], transform_matrix, channel_axis, fill_mode,
#                 cval)
#             if np.random.random() < 0.5:
#                 X_train_aug[p][range(m * n_modal, m * n_modal + n_modal)] = flip_axis(
#                     X_train_aug[p][range(m * n_modal, m * n_modal + n_modal)], 1)
#             if np.random.random() < 0.5:
#                 X_train_aug[p][range(m * n_modal, m * n_modal + n_modal)] = flip_axis(
#                     X_train_aug[p][range(m * n_modal, m * n_modal + n_modal)], 2)
#
#     # Equal Sampling
#     train_idx_n = np.where(Y_train == 0)[0]
#     train_idx_p = np.random.choice(np.where(Y_train == 1)[0], len(train_idx_n))
#     curr_idx = np.hstack([train_idx_n, train_idx_p])
#     random.shuffle(curr_idx)
#
#     hist = model.fit(X_train_aug[curr_idx], Y_train[curr_idx], batch_size=batch_size, nb_epoch=1,
#                      validation_data=(X_val, Y_val))
#
#     train_scores = model.evaluate(X_train, Y_train, batch_size=16)
#     print(train_scores)
#
#     loss = np.append(loss, train_scores[0])
#     categorical_accuracy = np.append(categorical_accuracy, train_scores[1])
#     val_loss = np.append(val_loss, np.asarray(hist.history['val_loss']))
#     val_categorical_accuracy = np.append(val_categorical_accuracy, np.asarray(hist.history['val_binary_accuracy']))
#     if val_categorical_accuracy[-1] > best_accuracy:
#         best_loss = val_loss[-1]
#         best_accuracy = val_categorical_accuracy[-1]
#         model.save(savename + '.h5')
#     elif val_loss[-1] < best_loss and val_categorical_accuracy[-1] == best_accuracy:
#         best_loss = val_loss[-1]
#         best_accuracy = val_categorical_accuracy[-1]
#         model.save(savename + '.h5')
#
# np.savez(savename, loss, categorical_accuracy, val_loss, val_categorical_accuracy)


