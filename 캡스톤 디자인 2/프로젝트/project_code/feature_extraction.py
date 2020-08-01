from project_code.utils import *

import os
import numpy as np
import scipy
import sklearn
import librosa
import librosa.display
from librosa import lpc
from scipy.fft import fft


c(x, fs):
    mfccs = librosa.feature.mfcc(x, sr=fs, n_mfcc=40)
    mfccs = np.resize(mfccs, (40, 150))
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

    return mfccs

def convert_stft(x, fs):
    D_octave = np.abs(librosa.core.stft(x))
    D_octave = np.resize(D_octave, (40, 150))
    return D_octave

def gcc_phat(sig1, sig2):
    pad1 = np.zeros(len(sig1))
    pad2 = np.zeros(len(sig2))
    sig1 = np.hstack([sig1, pad1])
    sig2 = np.hstack([sig2, pad2])
    f_sig1 = scipy.fftpack.fft(sig1, 6000)
    f_sig2 = scipy.fftpack.fft(sig2, 6000)
    f_sig2_c = np.conj(f_sig2)
    f_sig = f_sig1 * f_sig2_c
    denom = abs(f_sig)

    f_sig = f_sig / denom
    return np.abs(scipy.fftpack.ifft(f_sig, 6000))

def LPC(x, order):
    x_norm = x/(len(x)*len(x))
    filt = lpc(x_norm, order)
    return filt

def fft_cepstrum(x, N, order):
    yr = fft(x, n=order)
    yr = yr*(1/N)
    return yr

def match_label(x):
    if x == '01' or x == '06' or x == '09' or x == '23' or x == '24' or x == '27' or x == '30' or \
        x == '33' or x == '35' or x == '36' or x == '44' or x == '48':
        return 'F'
    elif x == '02' or x == '03' or x == '05' or x == '10' or x == '12' or x == '15' or x == '16' or \
        x == '18' or x == '21' or x == '22' or x == '28' or x == '28' or x == '31' or x == '34' or \
        x == '37' or x == '40' or x == '41' or x == '43' or x == '47':
        return 'K'
    else : return 'M'

def extract_direction_feature():
    PROJECT_DIR = get_upper_dir()
    DATA_DIR = get_data_dir(PROJECT_DIR)
    FEATURE_DIR = get_feature_dir(PROJECT_DIR)

    name_list = os.listdir(DATA_DIR)

    feature_list = []
    label_list = []

    for name in name_list:
        ch1_name = DATA_DIR + '/' + name + '_ch1.wav'
        ch2_name = DATA_DIR + '/' + name + '_ch2.wav'

        x1, fs1 = librosa.load(ch1_name)
        x2, fs2 = librosa.load(ch2_name)

        mfcc1 = convert_mfcc(x1, fs1)
        mfcc2 = convert_mfcc(x2, fs2)
        mfcc = mfcc2 - mfcc1

        stft1 = convert_stft(x1, fs1)
        stft2 = convert_stft(x2, fs2)
        stft = stft2 - stft1

        gcc = gcc_phat(x1, x2)
        gcc = gcc.reshape([-1, 150])

        feature = np.concatenate((mfcc, stft, gcc), axis=0)
        feature_list.append(feature)

        label = name[-3:]
        label_list.append(label)

        np.save(FEATURE_DIR + 'direction/' + 'feature2.npy', np.array(feature_list))
        np.save(FEATURE_DIR + 'direction/' + 'label2.npy', np.array(label_list))

def extract_gender_feature():
    PROJECT_DIR = get_upper_dir()
    DATA_DIR = get_data_dir(PROJECT_DIR)
    FEATURE_DIR = get_feature_dir(PROJECT_DIR)

    name_list = os.listdir(DATA_DIR)

    feature_list = []
    label_list = []
    num = 0
    for name in name_list:
        ch1_name = DATA_DIR + '/' + name + '_ch1.wav'
        ch2_name = DATA_DIR + '/' + name + '_ch2.wav'

        idx = name[:-10][-2:]
        label = match_label(idx)

        x1, fs1 = librosa.load(ch1_name)
        x2, fs2 = librosa.load(ch2_name)
        try:
            mfcc1 = convert_mfcc(x1, fs1)
            mfcc2 = convert_mfcc(x2, fs2)

            lpc_10_1 = LPC(x1, 10)
            lpc_10_2 = LPC(x2, 10)
            lpc_12_1 = LPC(x1, 12)
            lpc_12_2 = LPC(x2, 12)

            fft_8_1 = fft_cepstrum(x1, len(x1), 8)
            fft_8_2 = fft_cepstrum(x2, len(x2), 8)
            fft_12_1 = fft_cepstrum(x1, len(x1), 12)
            fft_12_2 = fft_cepstrum(x2, len(x2), 12)

            feature = np.concatenate((mfcc1, lpc_10_1, lpc_12_1, fft_8_1, fft_12_1), axis=0)
            feature_list.append(feature)
            label_list.append(label)

            feature = np.concatenate((mfcc2, lpc_10_2, lpc_12_2, fft_8_2, fft_12_2), axis=0)
            feature_list.append(feature)
            label_list.append(label)

        except:
            num += 1
            continue

        np.save(DATA_DIR + '/gender_' + name + '.npy', np.array(feature_list))
        np.save(FEATURE_DIR + 'gender_age/' + 'label1.npy', np.array(label_list))

extract_direction_feature()
extract_gender_feature()