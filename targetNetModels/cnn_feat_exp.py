import torch
import torch.nn.functional as F


class CNN5_0(torch.nn.Module):
    def __init__(self):
        super(CNN5_0, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*80, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)    
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x, probs


class FE_CNN_0(torch.nn.Module):
    def __init__(self):
        super(FE_CNN_0, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn(self.conv0(x)))
        x = self.pool(x)

        return x # 3, 16, 16


class CNN5_1(torch.nn.Module):
    def __init__(self):
        super(CNN5_1, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*80, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x, probs
    

class FE_CNN_1(torch.nn.Module):
    def __init__(self):
        super(FE_CNN_1, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn(self.conv0(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)

        return x


class CNN5_2(torch.nn.Module):
    def __init__(self):
        super(CNN5_2, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*80, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x, probs


class FE_CNN_2(torch.nn.Module):
    def __init__(self):
        super(FE_CNN_2, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn(self.conv0(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)

        return x


class CNN5_3(torch.nn.Module):
    def __init__(self):
        super(CNN5_3, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*80, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)            
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x, probs


class FE_CNN_3(torch.nn.Module):
    def __init__(self):
        super(FE_CNN_3, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn(self.conv0(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(x)

        return x


class CNN5_4(torch.nn.Module):
    def __init__(self):
        super(CNN5_4, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*80, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x) 
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x, probs


class FE_CNN_4(torch.nn.Module):
    def __init__(self):
        super(FE_CNN_4, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn(self.conv0(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.pool(x)

        return x
    
    
class CNN5_5(torch.nn.Module):
    def __init__(self):
        super(CNN5_5, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*80, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)    
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x, probs
    
    
class FE_CNN_5(torch.nn.Module):
    def __init__(self):
        super(FE_CNN_5, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)  # self.conv1.weight.size(): torch.Size([16, 1, 3, 3])
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = torch.nn.Linear(1*1*80, 10, bias=True)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = torch.nn.functional.relu(self.bn(self.conv0(x)))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv4(x))
            x = self.pool(x)    
            x = x.view(x.size(0),-1)
            x = self.fc(x)
            probs = F.softmax(x, dim=1)
        return x, probs        