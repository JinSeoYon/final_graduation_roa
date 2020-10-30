from __future__ import print_function, division, unicode_literals
import flask
import os
from flask import Flask, request, render_template, jsonify
from sklearn.externals import joblib
import numpy as np
from scipy import misc
from werkzeug.utils import secure_filename
from tensorflow import keras
from keras.utils import Sequence                                # 이 모듈이 없으면 사용자가 만든 generator에서 'shape'가 없다고 하는 에러가 발생할 수 있음
import matplotlib.pyplot as plt
import time
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
import wave
from tensorflow.keras.callbacks import ModelCheckpoint

import os.path as op
from sklearn.model_selection import train_test_split
import tensorflow as tf
import code
import glob

import librosa
import soundfile as sf
import sounddevice as sd
import queue

server='http://192.168.0.16:2222'
app = Flask(__name__)
#UPLOAD_FOLDER = '/uploads'
UPLOAD_FOLDER = '/Users/sheoyonjin/Local_doc/402/final_graduation/DeepLearning/ROA/ROA_server_flask/predict'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wav'])

label=""


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/upload',methods = ['GET','POST'])
def upload_file():
    if request.method =='POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            #main('-p')
            features, filenames = parse_predict_files('predict')
            np.save('predict_feat.npy', features)
            np.save('predict_filenames.npy', filenames)
            label=predict('-p')
            #result=label

    #if request.method=='GET':
        #return jsonify(hello=result) 

        return jsonify(result=label)
    return jsonify(result='error')

@app.route('/result',methods = ['GET'])    
def return_result():
    if request.method =='GET':
        #final=predict('-p')
        features, filenames = parse_predict_files('predict')
        np.save('predict_feat.npy', features)
        np.save('predict_filenames.npy', filenames)
        label=predict('-p')
        #return jsonify(result='final')
        return jsonify(result=label)
    return jsonify(result='error')

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

def noise_reduction(parent_dir,file_ext='*.wav'):
    for fn in glob.glob(os.path.join(parent_dir, file_ext)):
        
        wr = wave.open(fn, 'r')
        #plot_graph(wr,1,'input')	
        wr = wave.open(fn, 'r')
        par = list(wr.getparams())
        par[3] = 0
        ww = wave.open('predict/filtered.wav', 'w')
        ww.setparams(tuple(par)) # Use the same parameters as the input file.
        lowpass = 1500 # Remove lower frequencies.
        highpass = 4000 # Remove higher frequencies.
        sz = wr.getframerate() # Read and process 1 second at a time.
        c = int(wr.getnframes()/sz) # whole file
        for num in range(c):
            print('Processing {}/{} s'.format(num+1, c))
            da = np.fromstring(wr.readframes(sz), dtype=np.int16)
            left, right = da[0::2], da[1::2] # left and right channel
            lf, rf = np.fft.rfft(left),np.fft.rfft(right)
            lf[:lowpass], rf[:lowpass] = 0, 0 # low pass filter
            #lf[55:66], rf[55:66] = 0, 0 # line noise in sample from site
            lf[highpass:], rf[highpass:] = 0,0 # high pass filter
            nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
            ns = np.column_stack((nl,nr)).ravel().astype(np.int16)
            ww.writeframes(ns.tostring())
        # Close the files.
        wr.close()
        ww.close()
        
        spf = wave.open("predict/filtered.wav", 'r')
       

def parse_predict_files(parent_dir,file_ext='filtered.wav'):
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
    model = keras.models.load_model('FINAL_graduation_air_model2.h5')
    

    predict_feat_path = 'predict_feat.npy'
    predict_filenames = 'predict_filenames.npy'
    filenames = np.load(predict_filenames)
    X_predict = np.load(predict_feat_path)
    print(X_predict)
    X_predict = np.expand_dims(X_predict, axis=2)

    pred = model.predict_classes(X_predict)

    
    label_list=pred.tolist()
    label=label_list[0]
    
    

    return label



def main(args):
    noise_reduction('predict')
    features, filenames = parse_predict_files('predict')
    np.save('predict_feat.npy', features)
    np.save('predict_filenames.npy', filenames)
    predict(args)
    return render_template('index.html', label=label)

if __name__ == '__main__':
    model=keras.models.load_model('FINAL_graduation_air_model2.h5')
    
 
    app.run(host="192.168.0.16", debug=True)