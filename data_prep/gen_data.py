import os
import argparse
import numpy as np
#import cPickle as cp
import pickle as cp
from io import BytesIO
from pandas import Series
from numpy import linalg as LA


def gen_rand_data(subject, target_path):
    """Function to read the raw data and process

    :param subject: string
    :param target_path: string
        Processed file
    """

    DIM_FEAT1 = 4096*10
    DIM_FEAT2 = 4096*10

    data_x1 = np.empty((0, DIM_FEAT1))
    data_x2 = np.empty((0, DIM_FEAT2))
    data_x = np.hstack((data_x1, data_x2))
    data_y = np.empty((0)).reshape(-1,1)

    try:
        for i in range(1000):
            x1 = np.random.normal(size=(1, DIM_FEAT1))
            x2 = np.random.normal(size=(1, DIM_FEAT2))
            x = np.hstack((x1, x2))
            y = 0
            # normdist = LA.norm(x1-x2)
            # print(normdist)
            # if normdist>0.5:
            #     y = np.array(0).reshape((1,-1))
            # else:
            #     y = np.array(1).reshape((1,-1))
            data_x = np.vstack((data_x, x))
            data_y = np.vstack([data_y, y])

        for i in range(1000):
            x1 = np.random.normal(size=(1, DIM_FEAT1))
            x2 = x1
            x = np.hstack((x1, x2))
            y = np.array(1)
            data_x = np.vstack((data_x, x))
            data_y = np.vstack([data_y, y])

    except KeyError:
        print('ERROR: Did not find {0} in zip file'.format(filename))

    # shuffle
    data = np.hstack((data_x, data_y))
    np.random.shuffle(data)
    data_x = data[:,:-1]
    data_y = data[:,-1]

    # Dataset is segmented into train and test
    nb_training_samples = int(len(data_y)*2/3)
    # The first 18 OPPORTUNITY data files define the traning dataset, comprising 557963 samples
    X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]

    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))

    obj = [(X_train, y_train), (X_test, y_test)]
    f = open(os.path.join(target_path), 'wb')
    cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    f.close()


def gen_sin_data(subject, target_path):
    """Function to read the raw data and process

    :param subject: string
    :param target_path: string
        Processed file
    """

    DIM_FEAT1 = 4096*10
    DIM_FEAT2 = 4096*10

    data_x1 = np.empty((0, DIM_FEAT1))
    data_x2 = np.empty((0, DIM_FEAT2))
    data_x = np.hstack((data_x1, data_x2))
    data_y = np.empty((0)).reshape(-1,1)

    x1 = gen_sin(DIM_FEAT1).reshape((1, DIM_FEAT1))
    x2 = gen_cos(DIM_FEAT2).reshape((1, DIM_FEAT2))
    x = np.hstack((x1, x2))
    data_x = np.repeat(x, 1000, axis=0)
    data_y = np.zeros((1000,1))

    x1 = gen_sin(DIM_FEAT1).reshape(1, DIM_FEAT1)
    x2 = x1
    x = np.hstack((x1, x2))
    x = np.repeat(x, 1000, axis=0)
    y = np.ones((1000,1))

    data_x = np.vstack((data_x, x))
    data_y = np.vstack([data_y, y])


    # shuffle
    data = np.hstack((data_x, data_y))
    np.random.shuffle(data)
    data_x = data[:,:-1]
    data_y = data[:,-1]

    # Dataset is segmented into train and test
    nb_training_samples = int(len(data_y)*2/3)
    # The first 18 OPPORTUNITY data files define the traning dataset, comprising 557963 samples
    X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]

    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))

    obj = [(X_train, y_train), (X_test, y_test)]
    f = open(os.path.join(target_path), 'wb')
    cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    f.close()


def gen_cos(DIM_FEAT):    
    # import matplotlib.pyplot as plt

    Fs = 20000
    f = 5
    sample = DIM_FEAT
    x = np.arange(sample)
    y = np.cos(2 * np.pi * f * x / Fs)

    # plt.plot(x, y)
    # plt.xlabel('sample(n)')
    # plt.ylabel('voltage(V)')
    # plt.show()

    return y


def gen_sin(DIM_FEAT):
    Fs = 20000
    f = 5
    sample = DIM_FEAT
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)
    return y


def main():
    target_path = '../sin_test.data'
    print('generating data..')
    gen_sin_data('202',target_path)


if __name__ == "__main__":
    main()
