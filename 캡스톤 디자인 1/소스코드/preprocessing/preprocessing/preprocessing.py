import wave
import numpy as np
import os
import scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, librosa.display

def save_wav_channel(wav, fn, channel):
    '''
    Take Wave_read object as an input and save one of its
    channels into a separate .wav file.
    '''
    # Read data
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.fromstring(sdata, dtype=typ)
    ch_data = data[channel::nch]

    # Save channel to a separate file
    outwav = wave.open(fn, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(ch_data.tostring())
    outwav.close()
    
def convert_mfcc(x, fs):
    #x, fs = librosa.load(file_name)
    #librosa.display.waveplot(x, sr=fs)

    mfccs = librosa.feature.mfcc(x, sr=fs, n_mfcc=20)
    #print (mfccs.shape)
    mfccs = np.resize(mfccs, (20,150))
    #print (mfccs2.shape)

    #librosa.display.specshow(mfccs, sr=fs, x_axis='time')

    #mfccs2 = mfccs2 - mfccs

    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    
    #print (mfccs2.mean(axis=1))
    #print (mfccs2.var(axis=1))
    #librosa.display.specshow(mfccs, sr=fs, x_axis='time')
    
    return mfccs

def convert_stft(x, fs):
    D_octave = np.abs(scipy.fftpack.fft(x, 3000))
    return D_octave

def gcc_phat(sig1, sig2):
    pad1 = np.zeros(len(sig1))
    pad2 = np.zeros(len(sig2))

    sig1 = np.hstack([sig1, pad1])
    sig2 = np.hstack([sig2, pad2])

    f_sig1 = scipy.fftpack.fft(sig1, 3000)
    f_sig2 = scipy.fftpack.fft(sig2, 3000)

    f_sig2_c = np.conj(f_sig2)
    f_sig = f_sig1 * f_sig2_c

    denom = abs(f_sig)
    #denom[denom<1e-6] = 1e-6
    f_sig = f_sig / denom

    return np.abs(scipy.fftpack.ifft(f_sig, 3000))

path_dir = r"C:\Users\USER\Downloads\output_noisy_file\180"
div_dir = r"C:\Users\USER\Documents\카카오톡 받은 파일\20\ch_division_20"
save_dir = r"C:\Users\USER\Downloads\output_noisy_file_eachAngle\180"
file_list = os.listdir(path_dir)
data_arr1 = []
data_arr2 = []
data_arr3 = []

for item in file_list:
    before = item[0:14]
    angle = item[14:17]
    channel = item[18:21]
    after = item[21:]
    file_name = path_dir+"\\"+item
    wav = wave.open(file_name)
    #if channel == 'ch1' and angle == '160':
    #    print(item)
    #    ch1_name = path_dir+'\\'+before+angle+'_'+channel+after
    #    ch2_name = path_dir+'\\'+before+angle+'_'+'ch2'+after

    ch1_name = path_dir + '\\' + item.replace('.wav', '') + '_ch1.wav'
    ch2_name = path_dir + '\\' + item.replace('.wav', '') + '_ch2.wav'

    save_wav_channel(wav, ch1_name, 0)
    save_wav_channel(wav, ch2_name, 1)

    x1, fs1 = librosa.load(ch1_name)
    x2, fs2 = librosa.load(ch2_name)

    mfcc1 = convert_mfcc(x1, fs1)
    mfcc2 = convert_mfcc(x2, fs2)
    
    mfcc = mfcc2-mfcc1
    #mfcc = sklearn.preprocessing.scale(mfcc, axis = 1)

    stft1 = convert_stft(x1, fs1)
    stft2 = convert_stft(x2, fs2)
    stft = stft2-stft1
    stft = np.reshape(stft, (20, 150))
    #stft = sklearn.preprocessing.scale(stft, axis = 1)
    
    gcc = gcc_phat(x1, x2)
    #print(gcc.shape)
    #print(gcc)
    gcc = np.reshape(gcc, (20, 150))

    result1 = np.hstack((mfcc, gcc))
    #print(result1.shape)
    
    result2 = np.hstack((mfcc, stft))

    result3 = np.hstack((result1, stft))

    result1 = sklearn.preprocessing.scale(result1, axis = 1)
    result2 = sklearn.preprocessing.scale(result2, axis = 1)
    result3 = sklearn.preprocessing.scale(result3, axis = 1)

    data_arr1.append(result1)
    data_arr2.append(result2)
    data_arr3.append(result3)

    #os.remove(ch1_name)
    #os.remove(ch2_name)

np.save(save_dir + '\\' + 'mfcc+gcc.npy', data_arr1)
np.save(save_dir +'\\' + 'mfcc+stft.npy', data_arr2)
np.save(save_dir +'\\' + 'mfcc+stft+gcc.npy', data_arr3)