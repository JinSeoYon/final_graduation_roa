
import feat_extract
from feat_extract import *
import time
import argparse
import numpy as np
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import os.path as op
from sklearn.model_selection import train_test_split
import tensorflow as tf

def train(args):
    if not op.exists('roa_features.npy') or not op.exists('roa_label.npy'):
        if input('No feature/labels found. Run feat_extract.py first? (Y/n)').lower() in ['y', 'yes', '']:
            feat_extract.main()
            train(args)
    else:
        X = np.load('roa_features.npy')
        y = np.load('roa_label.npy').ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=233)

    # Count the number of sub-directories in 'data'
    class_count = len(next(os.walk('data/'))[1])+1

    print(class_count)
    # Build the Neural Network
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
    model.add(Conv1D(64, 3, activation='relu',kernel_regularizer='l2'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu',kernel_regularizer='l2'))
    model.add(MaxPooling1D(3))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax',kernel_regularizer='l2'))
    model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
    print(model.summary())
    
    y_train = keras.utils.to_categorical(y_train, num_classes=class_count)

    y_test = keras.utils.to_categorical(y_test, num_classes=class_count)

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    #checkpoint_path = "/Users/sheoyonjin/Desktop/ROA/crying-classification/check_point/cp-{epoch:04d}.ckpt"
    #checkpoint_dir=os.path.dirname(checkpoint_path)

    #cp_callback =tf.keras.callbacks.ModelCheckpoint(
    #    checkpoint_path,save_weights_only=True, verbose=1,period=100)
    start = time.time()
    
    history = model.fit(X_train, y_train, validation_split=0.25, epochs=6000, batch_size=32, verbose=0)


    score,acc,loss = model.evaluate(X_test, y_test, batch_size=32)
    
      # Save list of words.
    print(score,acc,loss)

    # print('Test accuracy:', acc2)
    # print('loss :', loss)
    print('Training took: %d seconds' % int(time.time() - start))

    # list all data in history
    print(history.history.keys())
    
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save(args.model)

def predict(args):
    if op.exists(args.model):
        model = keras.models.load_model(args.model)
        predict_feat_path = 'predict_feat.npy'
        predict_filenames = 'predict_filenames.npy'
        filenames = np.load(predict_filenames)
        X_predict = np.load(predict_feat_path)
        print(X_predict)
        X_predict = np.expand_dims(X_predict, axis=2)

        pred = model.predict_classes(X_predict)
        #pred = model.predict(X_predict)

        for pair in list(zip(filenames, pred)):

            print(pair)

    elif input('Model not found. Train network first? (Y/n)').lower() in ['y', 'yes', '']:
        train()
        predict(args)

def real_time_predict(args):
    import sounddevice as sd
    import soundfile as sf
    import queue
    import librosa
    import sys
    #name ={'1':'dog','2':'rain','3':'sea','4':'baby crying','5':'clock','6':'person sneeze','7':'helicopter','8':'chainsaw','9':'rooster','10':'fire crackling'}
    name = {1: 'belly_pain', 2: 'burping', 3: 'discomfort', 4: 'hungry', 5: 'tired'}

    if op.exists(args.model):
        model = keras.models.load_model(args.model)
        predict_feat_path = 'predict_feat.npy'
        predict_filenames = 'predict_filenames.npy'
        while True:
            try:
                features = np.empty((0, 193))
                start = time.time()
                mfccs, chroma, mel, contrast, tonnetz = extract_feature()
                end = time.time()
                print('-----------------------------------------------------',end-start)
                ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                features = np.expand_dims(features, axis=2)
                filenames = np.load(predict_filenames)
                pred = model.predict_classes(features)

                for p in pred:
                    print(name[p])

                    if args.verbose: print('Time elapsed in real time feature extraction: ', time.time() - start)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                parser.exit(0)
            except Exception as e:
                parser.exit(type(e).__name__ + ': ' + str(e))
    elif input('Model not found. Train network first? (y/N)') in ['y', 'yes']:
        train()
        real_time_predict(args)


def main(args):
    if args.train: train(args)
    elif args.predict: predict(args)
    elif args.real_time_predict: real_time_predict(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--train',             action='store_true',                           help='train neural network with extracted features')
    parser.add_argument('-m', '--model',             metavar='path',     default='FINAL_graduation_air_model.h5',help='use this model path on train and predict operations')
    parser.add_argument('-e', '--epochs',            metavar='N',        default=6000,              help='epochs to train', type=int)
    parser.add_argument('-p', '--predict',           action='store_true',                           help='predict files in ./predict folder')
    parser.add_argument('-P', '--real-time-predict', action='store_true',                           help='predict sound in real time')
    parser.add_argument('-v', '--verbose',           action='store_true',                           help='verbose print')
    parser.add_argument('-s', '--log-speed',         action='store_true',                           help='performance profiling')
    parser.add_argument('-b', '--batch-size',        metavar='size',     default=32,                help='batch size', type=int)
    args = parser.parse_args()
    main(args)
