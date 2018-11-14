import os
import argparse
import numpy as np
#import cPickle as cp
import pickle as cp
from io import BytesIO
from pandas import Series



def gen_data(subject, target_path):
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
            y = np.array(0).reshape((1,-1))
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


def main():
    
    target_path = '../dummy_test.data'
    print('generating data..')
    gen_data('202',target_path)


if __name__ == "__main__":
    main()
