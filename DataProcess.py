import glob
import os

import numpy as np
from tensorflow import keras


def readData(path, test_size_rate=.2,start=0, each_count=500):
    matchesFilePath = 'D:\\6.PyProject\\drawSomething\\比赛数据集'
    matchesName = glob.glob(os.path.join(matchesFilePath, '*.npz'))

    class_names = []
    x = np.empty([0,784])
    y = np.empty([0])

    for index, file in enumerate(matchesName):
        print(file + ' read success!')
        file = file.replace(matchesFilePath, path).replace('npz', 'npy')
        data = np.load(file)

        data = data[start:each_count, :]
        label = np.full(each_count - start, index)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, label)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    y = keras.utils.to_categorical(y, len(class_names))

    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    test_size = int(x.shape[0] / 100 * (test_size_rate * 100))
    return x[test_size:x.shape[0], :], y[test_size:y.shape[0]], x[0:test_size, :], y[0:test_size], class_names


#a,b,c,d,e = readData(path='D:\\迅雷下载\\npy', each_count=20)
#c=1