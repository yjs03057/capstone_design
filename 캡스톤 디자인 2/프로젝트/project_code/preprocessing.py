from code.utils import *

from scipy import signal
import librosa
import math
import numpy as np
from pysndfx import AudioEffectsChain

PROJECT_DIR = get_upper_dir()
DATA_DIR = get_data_dir(PROJECT_DIR)
name_list = extract_file_name(DATA_DIR)

def band_butter_pass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def band_butter_filter(data, lowcut, highcut, fs):
    b, a = band_butter_pass(lowcut, highcut, fs)
    y = signal.lfilter(b, a, data)
    return y

def reduce_noise_centroid_s(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    threshold_h = np.max(cent)
    threshold_l = np.min(cent)
    less_noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5).limiter(gain=6.0)
    y_cleaned = less_noise(y)

    return y_cleaned

def reduce_noise_centroid_mb(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5).highshelf(gain=-30.0, frequency=threshold_h, slope=0.5).limiter(gain=10.0)
    y_cleaned = less_noise(y)


    cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows/3*2)
    boost_l = math.floor(rows/6)
    boost = math.floor(rows/3)

    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5)
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted

def reduce_noise_mfcc_up(y, sr):

    hop_length = 512

    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5)#.highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.5)#.limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

def reduce_noise_power(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.5
    threshold_l = round(np.median(cent))*0.1

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)
    y_clean = less_noise(y)

    return y_clean


def noise_reduction(x, sr, lowpass, highpass):
    y = band_butter_filter(x, lowpass, highpass, sr)
    y_mfcc_up = reduce_noise_mfcc_up(y, sr)
    y_centroid = reduce_noise_centroid_mb(y_mfcc_up, sr)
    y_cleaned = reduce_noise_power(y_centroid, sr)

    return y_cleaned

def save_wav(y, sr, filter_name, channel):
    normalized_y = y / np.abs(y).max()
    librosa.output.write_wav(PROJECT_DIR + '/data/test/' + filter_name + '_' + name_list[100] + channel\
                             , normalized_y.astype(np.float32), fs1)

lowpass = 1700
highpass = 2300

ch1_name = DATA_DIR + '/' + name_list[100] + '_ch1.wav'
ch2_name = DATA_DIR + '/' + name_list[100] + '_ch2.wav'
x1, fs1 = librosa.load(ch1_name)
x2, fs2 = librosa.load(ch2_name)

y1 = band_butter_filter(x1, lowpass, highpass, fs1)
y1_centroid = reduce_noise_centroid_s(y1, fs1)
y1_centroid_boosted = reduce_noise_centroid_mb(y1, fs1)
y1_centroid_combine = reduce_noise_centroid_mb(y1_centroid, fs1)
save_wav(x1, fs1, 'original', '_ch1.wav')
save_wav(y1, fs1, 'bㅇㅊㅊand', '_ch1.wav')
save_wav(y1_centroid, fs1, 'centroid', '_ch1.wav')
save_wav(y1_centroid_boosted, fs1, 'boosted', '_ch1.wav')
save_wav(y1_centroid_combine, fs1, 'combine', '_ch1.wav')


y2 = band_butter_filter(x1, lowpass, highpass, fs2)
y2_centroid = reduce_noise_centroid_s(y2, fs2)
y2_centroid_boosted = reduce_noise_centroid_mb(y2, fs2)
y2_centroid_combine = reduce_noise_centroid_mb(y2_centroid, fs2)
save_wav(x2, fs2, 'original', '_ch2.wav')
save_wav(y2, fs2, 'band', '_ch2.wav')
save_wav(y2_centroid, fs2, 'centroid', '_ch2.wav')
save_wav(y2_centroid_boosted, fs2, 'boosted', '_ch2.wav')
save_wav(y2_centroid_combine, fs2, 'combine', '_ch2.wav')
