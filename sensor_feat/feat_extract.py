import os
import sys
import numbers
import numpy as np
import pandas as pd
import pickle as cp
from datetime import datetime, timedelta
from utils import unixtime_to_datetime, datetime_str_to_unixtime,\
                string_to_datetime, datetime_to_foldername, datetime_to_filename,\
                lprint, create_folder

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim  
from torch.nn import functional as F
from torch.autograd import Variable

from resample import resample

sys.path.append("../DeepConvLstm_for_Actionrecog-master/")
import MyFeatureExtractor
from MyDataset import MyDataset 


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 100


def list_date_folders_hour_files(interval):
    """
    param interval: python datetime format or unixtimestamp (int)

    """
    start = interval[0]
    end = interval[1]

    if isinstance(start, numbers.Integral):
        start = unixtime_to_datetime(start)
    if isinstance(end, numbers.Integral):
        end = unixtime_to_datetime(end)

    # FFList means dateFolderHourFileList
    FFList = [[datetime_to_foldername(start), datetime_to_filename(start)]]
    curr = start + timedelta(hours = 1)

    while curr <= end:
        FFList.append([datetime_to_foldername(curr), datetime_to_filename(curr)])
        curr += timedelta(hours = 1)

    return FFList


def read_data(SUBJ, DEVICE, SENSOR, interval, reliThres):
    """
    param interval: python unixtimestamp

    """

    # 1. read in all the data within the range
    start = interval[0]
    end = interval[1]

    dfConcat = []

    RESAMPLE_PATH = '/Volumes/Seagate/SHIBO/MD2K/RESAMPLE/wild'
    FFList = list_date_folders_hour_files(interval)

    dfConcat = [pd.read_csv(os.path.join(RESAMPLE_PATH, SUBJ, DEVICE, SENSOR, FFList[i][0], FFList[i][1]))\
                for i in range(len(FFList))]

    df = pd.concat(dfConcat)

    # 2. the starts and ends of continuous chunks in returned data
    if len(str(abs(start))) == 10:
        start = start * 1000
    if len(str(abs(end))) == 10:
        end = end * 1000

    df = df[(df['time'] > start) & (df['time'] < end)]

    return df


def feat_extract(model, epoch, data_src, data_tar=None):
    featList = []

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_src):
            inputs, targets = data.data.view(-1, 1, 24,3).float(), target.type(torch.cuda.LongTensor)
            if inputs.shape[0] != 100:
                continue
            model.eval()
            feat = model(inputs)
            print(feat.shape)
            featList.append(feat)


# def test(model, epoch, data_src, data_tar=None):
#     total_loss_test = 0
#     criterion = nn.CrossEntropyLoss()
#     correct = 0
#     test_pred = np.empty((0))
#     test_true = np.empty((0))
#     list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar)) 
#     with torch.no_grad():
#         from sklearn.metrics import f1_score
#         for batch_id, (data, target) in enumerate(data_src):
#              #_, (x_tar, y_target) = list_tar[batch_j]
#             inputs, targets = data.data.view(-1, 1, 24,3).to(DEVICE), target.to(DEVICE).type(torch.cuda.LongTensor)
#             if inputs.shape[0] != 100:
#                 continue
#             model.eval()
#             y_src = model(inputs)
#             loss_c = criterion(y_src, targets)
#             pred = y_src.data.max(1)[1]
#             correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
#             total_loss_test += loss_c.data
#             test_pred = np.append(test_pred, pred, axis=0)
#             test_true = np.append(test_true, targets, axis=0)
#             res_i = 'Epoch: [{}/{}], Batchid: {}, correct:{} ,loss: {:.6f}'.format(epoch,
#                     EPOCH, batch_id, correct, loss_c.data)
#             tqdm.write(res_i)
#         acc = correct*100.0 / len(data_src.dataset)
#         total_loss_test /= (batch_id+1)
#         f1 = f1_score(test_true, test_pred, average='weighted')
#         res_e = 'Epoch: [{}/{}], test loss: {:.6f}, test accuracy: {:.4f}%, f1:{}'.format(
#             epoch, EPOCH, total_loss_test,  acc, f1)
#         tqdm.write(res_e)


def save_data(dataDir, targetFilename):
	assert os.path.isdir(dataDir) == 1
	SegDf = pd.read_csv('../../data/sens_data/segment/segments_2017-06-27 08_2017-06-27 08')
	print(SegDf)

	dataList = []
	for i in range(len(SegDf)):
	    start = SegDf['start'].iloc[i]
	    end = SegDf['end'].iloc[i]

	    dataDf = read_data('202', 'CHEST', 'ACCELEROMETER' ,[start, end], 0.8)
	    timeColHeader = 'time'
	    samplingRate = 10 / 50 * 24
	    newDataDf = resample(dataDf, timeColHeader, samplingRate, gapTolerance=np.inf, fixedTimeColumn=None)

	    if not len(newDataDf) == 24:
	        samplingRate = 10 / 50 * 25
	        newDataDf = resample(dataDf, timeColHeader, samplingRate, gapTolerance=np.inf, fixedTimeColumn=None)
	    
	    if not len(newDataDf) == 24:
	        continue

	    data = newDataDf[['accx', 'accy', 'accz']].values
	    dataList.append(data)

	data = np.stack(dataList)
	print(data.shape)
	f = open(os.path.join(dataDir, targetFilename), 'wb')
	cp.dump(data, f, protocol=cp.HIGHEST_PROTOCOL)
	f.close()



# =================================================================================
# 
#  1. save data
# 
# =================================================================================
# dataDir = '../../data/sen_feat/'
# targetFilename = '202_1.data'

# save_data(dataDir, targetFilename)



# =================================================================================
# 
#  2. extract features
# 
# =================================================================================

filename = '../../data/sen_data/202_1.data'
f = open(filename, 'rb')
data = cp.load(f)
y_test = np.zeros(data.shape[0])

    # data = Variable(torch.from_numpy(data))
    # data = data.view(64,1,-1,1)

model = torch.load('../DeepConvLstm_for_Actionrecog-master/model.pkl')# map_location='cpu'
sensMdl = MyFeatureExtractor.MyFeatureExtractor(model).float()

kwargs = {'num_workers': 1, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(MyDataset(data, y_test), 
                        batch_size=BATCH_SIZE, shuffle=False, **kwargs)

for e in tqdm(range(1, 1 + 1)):
    feat_extract(sensMdl, e, data_src=test_loader, data_tar=test_loader)






