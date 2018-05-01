import argparse
import os
import sys


import numpy as np
import keras
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D,BatchNormalization)
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import videoto3d

import time
import utils


#python 3dcnn_bis.py  --mode=test --model=


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--data_dir', type=str, default="./data/")
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=32)
    parser.add_argument('--calculateEvaluationCCC', type=str, default="./data/calculateEvaluationCCC.py")
    parser.add_argument('--validationCSV', type=str, default="./data/omg_ValidationVideos.csv")
    parser.add_argument('--trainCSV', type=str, default="./data/omg_TrainVideos.csv")


    args = parser.parse_args()
    timestamp = str(int(time.time()))
    args.out_dir = os.path.abspath(os.path.join("runs", timestamp))


    all_x = []
    all_y = []
    nb_classes = 2

    for mode in ["Train","Validation"]:
        img_rows, img_cols, frames = 32, 32, args.depth
        channel = 3 if args.color else 1
        fname_npz = os.path.join(args.data_dir,'dataset_{}_{}_{}_{}.npz').format(
            mode, nb_classes, args.depth, args.skip)
        vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
        print(fname_npz)
        if os.path.exists(fname_npz):
            loadeddata = np.load(fname_npz)
            X, Y = loadeddata["X"], loadeddata["Y"]
            print("Dataset found already for mode ",mode)
        else:
            x, y = utils.loaddata("{}_videos".format(mode), vid3d, args, mode, args.color, args.skip)
            X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
            X = X.astype('float32')
            Y = np.array(y)
            np.savez(fname_npz, X=X, Y=Y)
            print('Saved dataset to dataset.npz.')
        all_x.append(X)
        all_y.append(Y)

    X_train, X_test = all_x
    Y_train, Y_test = all_y

    print('Train : X_shape:{}\nY_shape:{}'.format(X_train.shape, Y_train.shape))
    print('Validation : X_shape:{}\nY_shape:{}'.format(X_test.shape, Y_test.shape))



    # Define model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(5, 5, 5), input_shape=(
        X.shape[1:]), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(4, 4, 4), padding='same'))
    model.add(Dropout(0.2))
    #
    model.add(Conv3D(32, kernel_size=(5, 5, 5), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='sigmoid'))

    model.add(Dropout(0.2))

    model.add(Dense(2, activation='linear'))

    model.compile(loss='mse',
                  optimizer=Adam(lr=0.001), metrics=['mse'])
    model.summary()

    # for i, j in enumerate(model.layers):
    #     print(i,j)
    # sys.exit()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    model_json = model.to_json()
    with open(os.path.join(args.out_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)


    filepath = os.path.join(args.out_dir,"weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath)
    predictions = utils.prediction_history(X_test, model, args)
    callbacks_list = [checkpoint, predictions]



    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True, callbacks=callbacks_list)


if __name__ == '__main__':
    main()
