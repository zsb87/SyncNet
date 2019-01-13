import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class MyFeatureExtractor(nn.Module):
    def __init__(self, model, transform_input=False):
        super(MyFeatureExtractor, self).__init__()
        #batchsize,1,24,113
        self.NUM_FILTERS = 128
        self.BATCH = 100
        self.conv1 = nn.Conv2d(1,64,(5,1),(1,1))
        self.conv2 = nn.Conv2d(64,128,(5,1),(1,1))
        self.conv3 = nn.Conv2d(128,256,(5,1),(1,1))
        self.conv4 = nn.Conv2d(256,512,(5,1),(1,1))
        self.dropout = nn.Dropout(0.5)
        #self.shuff = a.permute(,,,,)
        #inpdim, hiden units, lstm layers,
        #self.lstm1 = nn.LSTM(8, self.NUM_FILTERS, 2,batch_first=True)
        self.lstm1 = nn.LSTM(512, self.NUM_FILTERS, 1,batch_first=True)
        self.lstm2 = nn.LSTM(128,128,1,batch_first=True)
        # self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        print(x.shape)
        convs = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        #convs = convs.view(self.BATCH, 64*97, 8)
        convs = convs.view(self.BATCH, 8*3, 512)#8*97
        lstm1,_ = self.lstm1(self.dropout(convs))
        lstm2,_ = self.lstm2(self.dropout(lstm1))
        #lstm2 = lstm2[:,-1,:]
        lstm2 = lstm2.contiguous().view(-1,128)
        # print(lstm2.shape) # ([2400, 128])
        # fc = self.fc(lstm2)
        # print(fc.shape) # ([2400, 18])
        # fc2 = fc.view(self.BATCH, -1, 18)
        # print(fc2.shape) # ([100, 24, 18])
        # fc2 = fc2[:,-1,:]
        # return fc2
        return lstm2
