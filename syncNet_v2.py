import os
import sys
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt
import torch.optim as optim
from tensorboardX import SummaryWriter
writer = SummaryWriter(comment=str(random.randint(0,100)))


class RandomDatasetShift(Dataset):

    def __init__(self, nsample, size):
        length = 60*100000
        self.len = 3 * nsample
        self.data_v = torch.randn(nsample, size, length)
        self.data_s = torch.sin(self.data_v)
        self.v_neg_inc = np.random.randint(1, nsample, (nsample,))
        self.s_neg_inc = np.random.randint(1, nsample, (nsample,))

    def __getitem__(self, index):
        if index % 3 == 0:
            return torch.cat((self.data_v[index // 3], self.data_s[index // 3]), 0), 1
        elif index % 3 == 1:
            return torch.cat((self.data_v[index // 3], self.data_s[(index // 3 + self.v_neg_inc[index // 3]) % (self.len // 3)]), 0), 0
        else:
            return torch.cat((self.data_v[(index // 3 + self.s_neg_inc[index // 3]) % (self.len // 3)], self.data_s[index // 3]), 0), 0

    def __len__(self):
        return self.len


class RandomDataset(Dataset):

    def __init__(self, nsample, size):
        length = 60
        self.len = 3 * nsample
        self.data_v = torch.randn(nsample, size, length)
        self.data_s = torch.sin(self.data_v)
        self.v_neg_inc = np.random.randint(1, nsample, (nsample,))
        self.s_neg_inc = np.random.randint(1, nsample, (nsample,))

    def __getitem__(self, index):
        if index % 3 == 0:
            return torch.cat((self.data_v[index // 3], self.data_s[index // 3]), 0), 1
        elif index % 3 == 1:
            return torch.cat((self.data_v[index // 3], self.data_s[(index // 3 + self.v_neg_inc[index // 3]) % (self.len // 3)]), 0), 0
        else:
            return torch.cat((self.data_v[(index // 3 + self.s_neg_inc[index // 3]) % (self.len // 3)], self.data_s[index // 3]), 0), 0

    def __len__(self):
        return self.len


class RandomDataset2d(Dataset):

    def __init__(self, nsample, width, noise_ratio):
        height = 40
        self.len = 3 * nsample
        self.data_v = torch.randn(nsample, 1, height, width) # channel: 1
        self.data_s = torch.sin(self.data_v) + noise_ratio * torch.randn(nsample, 1, height, width) 
        self.v_neg_inc = np.random.randint(1, nsample, (nsample,))
        self.s_neg_inc = np.random.randint(1, nsample, (nsample,))

    def __getitem__(self, index):
        if index % 3 == 0:
            return torch.cat((self.data_v[index // 3], self.data_s[index // 3]), 0), 1
        elif index % 3 == 1:
            return torch.cat((self.data_v[index // 3], self.data_s[(index // 3 + self.v_neg_inc[index // 3]) % (self.len // 3)]), 0), 0
        else:
            return torch.cat((self.data_v[(index // 3 + self.s_neg_inc[index // 3]) % (self.len // 3)], self.data_s[index // 3]), 0), 0

    def __len__(self):
        return self.len


# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # visual branch
#         self.conv1a = nn.Conv1d(256, 64, 5)
#         self.conv2a = nn.Conv1d(64, 512, 5)
#         self.conv3a = nn.Conv1d(512, 1024, 5)
#         # sensor branch
#         self.conv1b = nn.Conv1d(256, 64, 5)
#         self.conv2b = nn.Conv1d(64, 512, 5)
#         self.conv3b = nn.Conv1d(512, 1024, 5)

#     def forward(self, input_data):
#         size = input_data.size()[1]
#         x_v = input_data[:, :size // 2]
#         x_s = input_data[:, size // 2:]
#         # visual branch
#         # Max pooling over a 4 window
#         x_v = F.max_pool1d(F.relu(self.conv1a(x_v)), 4, stride=2)
#         x_v = F.max_pool1d(F.relu(self.conv2a(x_v)), 4, stride=2)
#         x_v = F.relu(self.conv3a(x_v))
#         x_v = F.max_pool1d(x_v, x_v.size()[2])
#         # sensor branch
#         x_s = F.max_pool1d(F.relu(self.conv1b(x_s)), 4, stride=2)
#         x_s = F.max_pool1d(F.relu(self.conv2b(x_s)), 4, stride=2)
#         x_s = F.relu(self.conv3b(x_s))
#         x_s = F.max_pool1d(x_s, x_s.size()[2])
        
#         x_v = x_v.view(input_data.size()[0], 1, -1)
#         x_v_norm = x_v.norm(p=2, dim=2, keepdim=True)
#         x_v_normalized = x_v.div(x_v_norm.expand_as(x_v))
        
#         x_s = x_s.view(input_data.size()[0], -1, 1)
#         x_s_norm = x_s.norm(p=2, dim=1, keepdim=True)
#         x_s_normalized = x_s.div(x_s_norm.expand_as(x_s))
        
        
        
#         #x_out = torch.bmm(x_v_normalized, x_s_normalized).reshape((-1, 1))
        
#         x_out = torch.bmm(x_v, x_s).reshape((-1, 1))
#         return torch.cat((1 - x_out, x_out), 1)



class Net2d(nn.Module):

    def __init__(self):
        super(Net2d, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # visual branch
        self.conv1a = nn.Conv2d(1, 64, (40, 5), stride=1)
        self.conv2a = nn.Conv2d(64, 512, (1, 25), stride=1)
        self.conv3a = nn.Conv2d(512, 1024, (1, 25), stride=1)
        # sensor branch
        self.conv1b = nn.Conv2d(1, 64, (40, 5), stride=1)
        self.conv2b = nn.Conv2d(64, 512, (1, 25), stride=1)
        self.conv3b = nn.Conv2d(512, 1024, (1, 25), stride=1)

    def forward(self, input_data):
        size = input_data.size()[1]
        x_v = input_data[:, :size // 2]
        x_s = input_data[:, size // 2:]

        # visual branch
        # Max pooling over a 4 window

        x_v = F.max_pool2d(F.relu(self.conv1a(x_v)), (1, 4), stride=2) # relu+maxpool or maxpool+relu
        x_v = F.max_pool2d(F.relu(self.conv2a(x_v)), (1, 4), stride=2)
        x_v = F.relu(self.conv3a(x_v))
        # print(x_v.size())
        # print(x_v.size()[2])
        # x_v = F.max_pool2d(x_v, x_v.size()[2])
        # print(x_v.size())

        # sensor branch
        x_s = F.max_pool2d(F.relu(self.conv1b(x_s)), (1, 4), stride=2)
        x_s = F.max_pool2d(F.relu(self.conv2b(x_s)), (1, 4), stride=2)
        x_s = F.relu(self.conv3b(x_s))
        # x_s = F.max_pool2d(x_s, x_s.size()[2])
        
        x_v = x_v.view(input_data.size()[0], 1, -1)
        # print(x_v.size())

        # x_v_norm = x_v.norm(p=2, dim=2, keepdim=True)
        # x_v_normalized = x_v.div(x_v_norm.expand_as(x_v))
        
        x_s = x_s.view(input_data.size()[0], -1, 1)
        # x_s_norm = x_s.norm(p=2, dim=1, keepdim=True)
        # x_s_normalized = x_s.div(x_s_norm.expand_as(x_s))
        
        # x_out = torch.bmm(x_v_normalized, x_s_normalized).reshape((-1, 1))
        
        x_out = torch.bmm(x_v, x_s).reshape((-1, 1))
        return torch.cat((1 - x_out, x_out), 1)


net = Net2d()
print(net)


def train_net(net, trainloader, valloader=None, epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    running_total = running_correct = 0

    for epoch in range(epochs):  # loop over the dataset multiple times
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            writer.add_scalar('data/training_loss', loss, i+epoch*len(trainloader))
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            # print(predicted)
            # print statistics
            running_loss += loss.item()
            if (i + epoch * len(trainloader)) % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f, training accuracy: %d %%' %
                      (epoch + 1, i + 1, running_loss / 2000, 100 * running_correct / running_total))
                running_loss = 0.0
                running_total = running_correct = 0

        # save model parameters
        if epoch%5 == 4:
            torch.save(net.state_dict(), 'save_checkpoint_epoch'+str(epoch)+'_'+str(datetime.datetime.now())+'.pt')

        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                writer.add_scalar('data/val_loss', loss, epoch*len(trainloader))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the %d val images: %d %%' % (len(
            valloader), 100 * correct / total))
    print('Finished Training')


def test_net(net, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    batch_size = 1
    trainloader = DataLoader(dataset=RandomDataset(10000, 256),
                             batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=RandomDataset(2000, 256),
                           batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=RandomDataset(2000, 256), batch_size=batch_size,
                            shuffle=False, num_workers=2)
    net = Net()
    train_net(net, trainloader, valloader=valloader, epochs=50)
    test_net(net, testloader)


def main2d():

    noise_ratio = 0
    # noise_ratio = float(sys.argv[1])
    print('noise_ratio: ', noise_ratio)
    batch_size = 1
    trainloader = DataLoader(dataset=RandomDataset2d(10000, 256, noise_ratio),
                             batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=RandomDataset2d(2000, 256, noise_ratio),
                           batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=RandomDataset2d(2000, 256, noise_ratio), batch_size=batch_size,
                            shuffle=False, num_workers=2)
    net = Net2d()
    train_net(net, trainloader, valloader=valloader, epochs=20)
    test_net(net, testloader)

    #params = list(net.parameters())
    # print(len(params))
    # print(params[0].size())


if __name__ == "__main__":
    # print(os.path.basename(__file__))
    main2d()

    #load model parameters
    # net = Net2d()
    # net.load_state_dict(torch.load(PATH))

