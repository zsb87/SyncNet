import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import pickle as cp
from mydataset import mydataset

# MD2K_202_video_sensor.data: tuple of tuple: ((X_train, y_train),(X_test, y_test))
#   X_train, X_test: hstack both video features and sensor features


def load_dataset(filename):
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()
    X_train, y_train = data[0]
    X_test, y_test = data[1]
    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_test, y_test

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # visual branch
        self.conv1a = nn.Conv1d(256, 64, 5)
        self.conv2a = nn.Conv1d(64, 512, 25)
        self.conv3a = nn.Conv1d(512, 1024, 25)
        # sensor branch
        self.conv1b = nn.Conv1d(256, 64, 5)
        self.conv2b = nn.Conv1d(64, 512, 25)
        self.conv3b = nn.Conv1d(512, 1024, 25)

    def forward(self, input_data):
        x_v = input_data[:,0:40960].reshape(1,256,160)
        x_s = input_data[:,40960:].reshape(1,256,160)
        # visual branch
        # Max pooling over a 4 window
        tmp_a = self.conv1a(x_v)
       # print('after conv:')
       # print(tmp_a.size())
        tmp_b = F.relu(tmp_a)
       # print('after relu')
        #print(tmp_b.size())
        x_v = F.max_pool1d(tmp_b, 4, stride=2)
        #print('after max pooling:')
        #print(x_v.size())
        x_v = F.max_pool1d(F.relu(self.conv2a(x_v)), 4, stride=2)
        #print('after conv, relu, max pooling:')
        #print(x_v.size())
        x_v = F.relu(self.conv3a(x_v))
        #print('after conv relu:')
        #print(x_v.size())
        x_v = F.max_pool1d(x_v, x_v.size()[2]).squeeze()
        # sensor branch
        x_s = F.max_pool1d(F.relu(self.conv1b(x_s)), 4, stride=2)
        x_s = F.max_pool1d(F.relu(self.conv2b(x_s)), 4, stride=2)
        x_s = F.relu(self.conv3b(x_s))
        x_s = F.max_pool1d(x_s, x_s.size()[2]).squeeze()
        #print(x_v.size())
        #print(x_s.size())
        x_out = torch.dot(x_v, x_s)
        #x_temp = 1-x_out
        #x_out = torch.Tensor([x_temp,x_out])
        return x_out

net = Net()
print(net)

def train_net(net, trainloader, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.L1Loss()#CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=0.001)
    correct = 0
    total = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).squeeze().unsqueeze(0).type(torch.cuda.FloatTensor)
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.unsqueeze(0)
           # print(outputs)
           # print(labels)
            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            optimizer.step()
            predicted = outputs.data > 0.5
            predicted = predicted.type(torch.cuda.ByteTensor)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print('epoch: ', epoch, ', loss: ', running_loss/667)
        print('total:', total)
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    
    print('Finished Training')    
    
def test_net(net, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    net.to(device)
    loss = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).squeeze().unsqueeze(0)
            outputs = net(inputs)
            print(outputs.data,labels)
            #_, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            temp = outputs.data - labels.type(torch.cuda.FloatTensor)
            predicted = outputs.data > 0.5
            predicted = predicted.type(torch.cuda.ByteTensor)
            correct += (predicted == labels).sum().item()
            loss += torch.abs(temp)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print('loss = ', loss/total)
    
def main():
    
    batch_size = 1
    
    # X_train, y_train, X_test, y_test = load_dataset('MD2K_202_video_sensor.data')
    X_train, y_train, X_test, y_test = load_dataset('dummy_test.data')

    trainloader = DataLoader(dataset=mydataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True)
    testloader = DataLoader(mydataset(X_test, y_test), batch_size=1,
                                         shuffle=False, num_workers=2)
    net = Net()
    train_net(net, trainloader, 100)
    test_net(net, testloader)
    
    #params = list(net.parameters())
    #print(len(params))
    #print(params[0].size())    
if __name__ == "__main__":
    main()
