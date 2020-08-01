import os
import numpy as np
import librosa

def get_upper_dir():
    os.chdir('..')
    return os.getcwd()

def get_data_noise_dir(project_path):
    return os.path.join(project_path, 'data/output_noisy_file')

def get_data_dir(project_path):
    return os.path.join(project_path, 'data/output_file')

def get_feature_dir(project_path):
    return os.path.join(project_path, 'data/feature/')

def rename_data_file(data_path):
    wave_file_list = os.listdir(data_path)
    name_list = []

    for item in wave_file_list:
        before = item[0:18]
        after = item[21:]
        name_list.append(before + '&' + after)

    name_list = list(set(name_list))

    for idx, name in enumerate(name_list):
        file_name = name.split('&')
        ch1_name = data_path + '/' + file_name[0] + 'ch1' + file_name[1]
        ch2_name = data_path + '/' + file_name[0] + 'ch2' + file_name[1]
        data_num = file_name[0][-7:-5]
        angle = file_name[0][-4:-1]
        new_ch1_name = data_path + '/' + 'data_' + str(idx) + '_' + str(data_num) \
                       + '_angle_' + angle + '_ch1' + '.wav'
        new_ch2_name = data_path + '/' + 'data_' + str(idx) + '_' + str(data_num) \
                       + '_angle_' + angle + '_ch2' + '.wav'

        os.rename(ch1_name, new_ch1_name)
        os.rename(ch2_name, new_ch2_name)

def extract_file_name(data_path):
    wave_file_list = os.listdir(data_path)
    name_list = []
    for item in wave_file_list:
        name_list.append(item[:-8])
    name_list = list(set(name_list))
    return name_list

def save_wav(y, fs, name, channel, PROJECT_DIR):
    normalized_y = y / np.abs(y).max()
    librosa.output.write_wav(PROJECT_DIR + '/data/output_file/' + name + channel\
                             , normalized_y.astype(np.float32), fs)
    
def label_one_hot_encoder(y):
    y_vec = np.zeros((len(y), 8), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, int(y[i])] = 1.0
    return y_vec