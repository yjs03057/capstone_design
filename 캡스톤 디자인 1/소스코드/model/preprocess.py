import codecs
import numpy as np

# def read_train_data ():
#     with np.load('mfcc_data.npy') as f:
#         return np.array([k.strip() for k in f])

def read_label_file():
    with codecs.open('label.txt', 'r', encoding='utf-8') as f:
        return np.array([k.strip() for k in f])
def read_test_label_file():
    with codecs.open('test_label.txt', 'r', encoding='utf-8') as f:
        return np.array([k.strip() for k in f])

def preprocessing(x_data):
    result = []
    for element in x_data:
        ele_list = element.split(',')
        ele_list = [float(i) for i in ele_list]
        result.append(ele_list)
    return result

def _one_hot_encoder(y):
    y_vec = np.zeros((len(y), 4), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, int(y[i])] = 1.0
    return y_vec
# data = np.load('mfcc_data.npy')
# print(data[0].shape)
#print(_one_hot_encoder(read_label_file()))

#
# f = open("label.txt", 'w')
# for i in range(1, 2401):
#     data = "%d\n" % 0
#     f.write(data)
# for i in range(1, 2601):
#     data = "%d\n" % 1
#     f.write(data)
# for i in range(1, 2601):
#     data = "%d\n" % 2
#     f.write(data)
# for i in range(1, 1349):
#     data = "%d\n" % 3
#     f.write(data)
# f.close()