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

from tensorflow.keras.callbacks import ModelCheckpoint
import os
import os.path as op
from sklearn.model_selection import train_test_split
import tensorflow as tf
import code
import glob

import librosa
import soundfile as sf
import sounddevice as sd
import queue

def extract_feature(file_name=None):
    if file_name: 
        print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')

    else:  
        device_info = sd.query_devices(None, 'input')
        device_info['default_samplerate']=16000
        device_info['max_input_channels']=1
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()

        def callback(i,f,t,s): q.put(i.copy())
        data = []
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True: 
                if len(data) < 100000: data.extend(q.get())
                else: break
        X = np.array(data)

    if X.ndim > 1: X = X[:,0]
    X = X.T

    # short term fourier transform
   
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum)
    
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
   
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz



def parse_predict_files(parent_dir,file_ext='*.wav'):
    features = np.empty((0,193))
    filenames = []
    for fn in glob.glob(os.path.join(parent_dir, file_ext)):
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        filenames.append(fn)
        print("extract %s features done" % fn)
    return np.array(features), np.array(filenames)

def predict(args):
    if op.exists(args.model):
        model = keras.models.load_model(args.model)
        predict_feat_path = 'predict_feat.npy'
        predict_filenames = 'predict_filenames.npy'
        filenames = np.load(predict_filenames)
        X_predict = np.load(predict_feat_path)
        
        X_predict = np.expand_dims(X_predict, axis=2)
        
        pred = model.predict_classes(X_predict)
        #pred = model.predict(X_predict)
        for pair in list(zip(filenames, pred)):
            
            print(pair)

    elif input('Model not found. Train network first? (Y/n)').lower() in ['y', 'yes', '']:
        train()
        predict(args)

def main(args):
    features, filenames = parse_predict_files('predict')
    np.save('predict_feat.npy', features)
    np.save('predict_filenames.npy', filenames)
    predict(args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--train',             action='store_true',                           help='train neural network with extracted features')
    parser.add_argument('-m', '--model',             metavar='path',     default='FINAL_air_model3.h5',help='use this model path on train and predict operations')
    #parser.add_argument('-e', '--epochs',            metavar='N',        default=600,              help='epochs to train', type=int)
    parser.add_argument('-p', '--predict',           action='store_true',                           help='predict files in ./predict folder')
    # parser.add_argument('-P', '--real-time-predict', action='store_true',                           help='predict sound in real time')
    #parser.add_argument('-v', '--verbose',           action='store_true',                           help='verbose print')
    #parser.add_argument('-s', '--log-speed',         action='store_true',                           help='performance profiling')
    #parser.add_argument('-b', '--batch-size',        metavar='size',     default=64,                help='batch size', type=int)
    args = parser.parse_args()
    main(args)
