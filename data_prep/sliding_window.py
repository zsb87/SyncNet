import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from numpy.lib.stride_tricks import as_strided as ast
from utils import unixtime_to_datetime, datetime_str_to_unixtime


def string_to_datetime(startDatetime, endDatetime):
    # startDatetime = '06-27-17_11'
    # endDatetime = '06-27-17_15'
    start = datetime.strptime(startDatetime, '%m-%d-%y_%H_%M_%S')
    end = datetime.strptime(endDatetime, '%m-%d-%y_%H_%M_%S')
    return start, end


def datetime_to_foldername(dt):
    return dt.strftime('%m-%d-%y')

def datetime_to_filename(dt):
    return dt.strftime('%m-%d-%y_%H.csv')


def list_date_folders_hour_files(interval):
    """
    param interval: python datetime format

    """
    start = interval[0]
    end = interval[1]

    # FFList means dateFolderHourFileList
    FFList = [[datetime_to_foldername(start),datetime_to_filename(start)]]
    curr = start + timedelta(hours = 1)

    while curr <= end:
        FFList.append([datetime_to_foldername(curr), datetime_to_filename(curr)])
        curr += timedelta(hours = 1)

    return FFList


def read_data(SUBJ, DEVICE, SENSOR, interval, reliThres):
    """
    param interval: python datetime format

    """

    # 1. read in all the data within the range
    dfConcat = []

    RESAMPLE_PATH = '/Volumes/Seagate/SHIBO/MD2K/RESAMPLE/wild'
    FFList = list_date_folders_hour_files(interval)

    dfConcat = [pd.read_csv(os.path.join(RESAMPLE_PATH, SUBJ, DEVICE, SENSOR, FFList[i][0], FFList[i][1]))\
                for i in range(len(FFList))]

    df = pd.concat(dfConcat)

    # 2. the starts and ends of continuous chunks in returned data


    return df


def read_reliability(SUBJ, DEVICE, SENSOR, interval, reliThres):
    """
    param interval: python datetime format

    """

    # 1. read in all the data within the range
    dfConcat = []

    RESAMPLE_PATH = '/Volumes/Seagate/SHIBO/MD2K/RESAMPLE/wild'
    FFList = list_date_folders_hour_files(interval)

    dfConcat = [pd.read_csv(os.path.join(RESAMPLE_PATH, SUBJ, DEVICE, SENSOR+'_reliability', FFList[i][0], FFList[i][1]))\
                for i in range(len(FFList))]

    df = pd.concat(dfConcat)
    df = df.set_index('Time')

    # 2. the starts and ends of continuous chunks in returned data


    return df


def get_sliding_segment_from_reliability(df, winSizeSec, strSizeSec, sampleCountsThres):
    start = df.index[0]
    end = df.index[-1]
    segmentStartList = []
    segmentEndList = []

    for segStart in range(start, end, strSizeSec):
        segDataDf = df[(df.index >= segStart) & (df.index < segStart + winSizeSec)]
        if segDataDf.SampleCounts.sum() >= sampleCountsThres:
            segmentStartList.append(segStart)
            segmentEndList.append(segStart + winSizeSec)

    segDf = pd.DataFrame({'start':segmentStartList,'end':segmentEndList},\
                            columns=['start','end'])
    return segDf



if __name__ == '__main__':

    winSizeSec = 5
    strSizeSec = 1
    freq = 10

    print(datetime_str_to_unixtime('2017-10-04 18:14:22-05:00'))
    exit()


    # start, end = string_to_datetime()

    # Remove dark and lying in bed video:
    # startEndPairs = [[1498573212000+588000, 1498573212000+820000],\
    #                 [1498578527000+960000, 1498580653000+290000],\
    #                 ]
    
    # All video included:
    startEndPairs = [[1498572149100, 1498582779000]] # 202
    # startEndPairs = [[1507155672920, 1507163113920+84000]] # 211
    segDfConcat = []

    for start, end in startEndPairs:

        start = unixtime_to_datetime(start)
        end = unixtime_to_datetime(end)

        print(start)
        print(end)
        # df = read_data('202', 'CHEST', 'ACCELEROMETER' ,[start, end], 0.8)
        df = read_reliability('202', 'CHEST', 'ACCELEROMETER' ,[start, end], 0.8)
        segDf = get_sliding_segment_from_reliability(df, winSizeSec = winSizeSec, strSizeSec = strSizeSec,\
                 sampleCountsThres = winSizeSec*freq)
        
        segDfConcat.append(segDf)

        # print(end-start)

    df = pd.concat(segDfConcat)
    # df = df.reset_index()
    print(df)














# def norm_shape(shape):
#     '''
#     Normalize numpy array shapes so they're always expressed as a tuple,
#     even for one-dimensional shapes.

#     Parameters
#         shape - an int, or a tuple of ints

#     Returns
#         a shape tuple
#     '''
#     try:
#         i = int(shape)
#         return (i,)
#     except TypeError:
#         # shape was not a number
#         pass

#     try:
#         t = tuple(shape)
#         return t
#     except TypeError:
#         # shape was not iterable
#         pass

#     raise TypeError('shape must be an int, or a tuple of ints')

# def sliding_window(a,ws,ss = None,flatten = True):
#     '''
#     Return a sliding window over a in any number of dimensions

#     Parameters:
#         a  - an n-dimensional numpy array
#         ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
#              of each dimension of the window
#         ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
#              amount to slide the window in each dimension. If not specified, it
#              defaults to ws.
#         flatten - if True, all slices are flattened, otherwise, there is an
#                   extra dimension for each dimension of the input.

#     Returns
#         an array containing each n-dimensional window from a
#     '''

#     if None is ss:
#         # ss was not provided. the windows will not overlap in any direction.
#         ss = ws
#     ws = norm_shape(ws)
#     ss = norm_shape(ss)

#     # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
#     # dimension at once.
#     ws = np.array(ws)
#     ss = np.array(ss)
#     shape = np.array(a.shape)


#     # ensure that ws, ss, and a.shape all have the same number of dimensions
#     ls = [len(shape),len(ws),len(ss)]
#     if 1 != len(set(ls)):
#         raise ValueError(\
#         'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

#     # ensure that ws is smaller than a in every dimension
#     if np.any(ws > shape):
#         raise ValueError(\
#         'ws cannot be larger than a in any dimension.\
#  a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

#     # how many slices will there be in each dimension?
#     newshape = norm_shape(((shape - ws) // ss) + 1)
#     # the shape of the strided array will be the number of slices in each dimension
#     # plus the shape of the window (tuple addition)
#     newshape += norm_shape(ws)
#     # the strides tuple will be the array's strides multiplied by step size, plus
#     # the array's strides (tuple addition)
#     newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
#     strided = ast(a,shape = newshape,strides = newstrides)
#     if not flatten:
#         return strided

#     # Collapse strided so that it has one more dimension than the window.  I.e.,
#     # the new array is a flat list of slices.
#     meat = len(ws) if ws.shape else 0
#     firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
#     dim = firstdim + (newshape[-meat:])

#     # remove any dimensions with size 1
#     #dim = filter(lambda i : i != 1,dim)
#     return strided.reshape(dim)
