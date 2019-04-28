import os
import torch
import torch.nn as nn
import torch.nn.functional as function
import torch.optim as optim
from torchvision import datasets, transforms

kwargs = {'num_workers': 1, 'pin_memory': True}

train_data = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data', train=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data', train=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)

class neuralNetwork(nn.Module):
    def __init__(self):
        super(neuralNetwork, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*4*4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, input):
        #Feature Learning
            #Sequence 1
        input = function.relu(self.conv_layer1(input))
        input = self.pool(input)
            #Sequence 2
        input = function.relu(self.conv_layer2(input))
        input = self.pool(input)
        #Classification
        input = input.view(-1, 32*4*4)
        input = function.relu(self.fc1(input))
        input = function.dropout(input, training=self.training)
        input = function.relu(self.fc2(input))
        input = self.fc3(input)
        return function.log_softmax(input, dim=1)

network = neuralNetwork()
network.cuda()

optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.8)
criterion = function.nll_loss

#Train the neural network
def train(epoch):
    network.train()
    for batch_id, (data, target) in enumerate (train_data):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch = batch_id * len(data)
        total = len(train_data.dataset)
        done = 100. * batch_id / len(train_data)
        rate = loss.item()
        print(f"Train Epoch: {epoch} [{batch}/{total} ({done:.0f}%)]\tLoss: {rate:.6f}")

#Evaluation
def test():
    network.eval()
    loss = 0
    correct = 0
    for data, target in test_data:
        data = data.cuda()
        target = target.cuda()
        out = network(data)
        loss += function.nll_loss(out, target, reduction='sum').item()
        predicted = out.argmax(dim=1, keepdim=True) #Information, welches Bild erkannt wurde
        correct += predicted.eq(target.data.view_as(predicted)).cpu().sum().item() #Vergleich von erkanntem Bild und dem tatsächlichen Bild
    loss = loss / len(test_data.dataset)
    print('Average Loss: ', loss)
    print('Genauigkeit: ', 100.*correct/len(test_data.dataset), '%')

if __name__ == '__main__':
    for epoch in range(1, 10):
        train(epoch)
    test()

