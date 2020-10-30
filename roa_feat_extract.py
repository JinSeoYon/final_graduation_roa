#!/usr/bin/env python
# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import code
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
import sounddevice as sd
import queue
import wave
def plot_graph(file,figno,title):
	#Extract Raw Audio from Wav File
	signal = file.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	fs = file.getframerate()
	Time=np.linspace(0, len(signal)/fs, num=len(signal))
	plt.figure(figno)
	plt.title(title)
	plt.plot(Time,signal)
	plt.show()
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

#def parse_audio_files(parent_dir,file_ext='*.ogg'):
def parse_audio_files(parent_dir,file_ext='*.wav'):
    sub_dirs = os.listdir(parent_dir)
    
    
    sub_dirs.sort()
    
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try: mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                except Exception as e:
                    print("[Error] extract feature error in %s. %s" % (fn,e))
                    continue
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                # labels = np.append(labels, fn.split('/')[1])
                labels = np.append(labels, label)
            print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)
def noise_reduction(parent_dir,file_ext='*.wav'):
    for fn in glob.glob(os.path.join(parent_dir, file_ext)):
        
        wr = wave.open(fn, 'r')
        plot_graph(wr,1,'input')	
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
        plot_graph(spf,2,'output')

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

def main():
    # Get features and labels

    #features, labels = parse_audio_files('data')
    #np.save('roa_features.npy', features)
    #np.save('roa_label.npy', labels)

    # Predict new
    noise_reduction('predict')
    features, filenames = parse_predict_files('predict')
    np.save('predict_feat.npy', features)
    np.save('predict_filenames.npy', filenames)

if __name__ == '__main__':
    main()
