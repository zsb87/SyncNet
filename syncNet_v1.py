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
        self.conv1a = nn.Conv1d(4096, 64, 5)
        self.conv2a = nn.Conv1d(64, 512, 25)
        self.conv3a = nn.Conv1d(512, 1024, 25)
        # sensor branch
        self.conv1b = nn.Conv1d(4096, 64, 5)
        self.conv2b = nn.Conv1d(64, 512, 25)
        self.conv3b = nn.Conv1d(512, 1024, 25)

    def forward(self, input_data):
        x_v = input_data[:,0:4096].t().unsqueeze(0)
        x_s = input_data[:,4096:].t().unsqueeze(0)
        # visual branch
        # Max pooling over a 4 window
        tmp_a = self.conv1a(x_v)
        tmp_b = F.relu(tmp_a)
        x_v = F.max_pool1d(tmp_b, 4, stride=2)
        x_v = F.max_pool1d(F.relu(self.conv2a(x_v)), 4, stride=2)
        x_v = F.relu(self.conv3a(x_v))
        print(x_v.size())
        x_v = F.max_pool1d(x_v, x_v.size()[1]).squeeze()
        # sensor branch
        x_s = F.max_pool1d(F.relu(self.conv1b(x_s)), 4, stride=2)
        x_s = F.max_pool1d(F.relu(self.conv2b(x_s)), 4, stride=2)
        x_s = F.relu(self.conv3b(x_s))
        x_s = F.max_pool1d(x_s, x_s.size()[1]).squeeze()
        x_out = torch.dot(x_v, x_s)
        return x

net = Net()
print(net)

def train_net(net, trainloader, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
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
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')    
    
def test_net(net, testloader):
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

    # m = nn.Conv1d(16, 33, 3, stride=2)
    # input = torch.randn(1,16,50)
    # output = m(input)
    # print(output.size())
    # exit()
    
    batch_size = 200
    
    # X_train, y_train, X_test, y_test = load_dataset('MD2K_202_video_sensor.data')
    X_train, y_train, X_test, y_test = load_dataset('dummy_test.data')

    # print(X_train)

    trainloader = DataLoader(dataset=mydataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True)
    testloader = DataLoader(mydataset(X_test, y_test), batch_size=4,
                                         shuffle=False, num_workers=2)
    net = Net()
    train_net(net, trainloader, 1)
    test_net(net, testloader)
    
    #params = list(net.parameters())
    #print(len(params))
    #print(params[0].size())    
if __name__ == "__main__":
    main()
