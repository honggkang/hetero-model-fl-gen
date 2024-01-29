import torch
import torch.nn.functional as F



class MLP2(torch.nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc5 = torch.nn.Linear(784, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs
   
    
class MLP3(torch.nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 196)
        self.fc5 = torch.nn.Linear(196, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs
    

class MLP3b(torch.nn.Module):
    def __init__(self):
        super(MLP3b, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 128)
        self.fc5 = torch.nn.Linear(128, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs


class MLP4(torch.nn.Module):
    def __init__(self):
        super(MLP4, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 256)
        self.fc4 = torch.nn.Linear(256, 196)
        self.fc5 = torch.nn.Linear(196, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs


class MLP4b(torch.nn.Module):
    def __init__(self):
        super(MLP4b, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 400)
        self.fc4 = torch.nn.Linear(400, 196)
        self.fc5 = torch.nn.Linear(196, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs


class MLP4c(torch.nn.Module):
    def __init__(self):
        super(MLP4c, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 128)
        self.fc4 = torch.nn.Linear(128, 196)
        self.fc5 = torch.nn.Linear(196, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs


class MLP5(torch.nn.Module):
    def __init__(self):
        super(MLP5, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 196)
        self.fc3 = torch.nn.Linear(196, 256)
        self.fc4 = torch.nn.Linear(256, 196)
        self.fc5 = torch.nn.Linear(196, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc3(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc3(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs


class MLP5b(torch.nn.Module):
    def __init__(self):
        super(MLP5b, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 320)
        self.fc3 = torch.nn.Linear(320, 320)
        self.fc4 = torch.nn.Linear(320, 196)
        self.fc5 = torch.nn.Linear(196, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc3(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc3(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs


class MLP5c(torch.nn.Module):
    def __init__(self):
        super(MLP5c, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 196)
        self.fc5 = torch.nn.Linear(196, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc3(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc3(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.dropout(x, training=self.training)            
            x = torch.nn.functional.relu(self.fc5(x))
            probs = F.softmax(x, dim=1)
        return x, probs


class MLP6(torch.nn.Module):
    def __init__(self):
        super(MLP6, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        self.fc2 = torch.nn.Linear(784, 256) # 16*16
        # self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 196)
        self.fc5 = torch.nn.Linear(196, 196) # 14*14
        self.fc6 = torch.nn.Linear(196, 10)
    
    def forward(self, x, start_layer=None):
        if start_layer == None:
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            # x = torch.nn.functional.relu(self.fc3(x))
            x = torch.nn.functional.relu(self.fc4(x))
            x = torch.nn.functional.relu(self.fc5(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = self.fc6(x)
            probs = F.softmax(x, dim=1)
        elif start_layer == 'feature':
            # x = x.view(-1, 196)
            x = torch.nn.functional.relu(self.fc2(x))
            # x = torch.nn.functional.relu(self.fc3(x))
            x = torch.nn.functional.relu(self.fc4(x))            
            x = torch.nn.functional.relu(self.fc5(x))
            x = torch.nn.functional.dropout(x, training=self.training)
            x = self.fc6(x)
            probs = F.softmax(x, dim=1)
        return x, probs


class FE_MLP(torch.nn.Module):
    def __init__(self):
        super(FE_MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 784)
        # self.fc2 = torch.nn.Linear(256, 256)
        # self.fc3 = torch.nn.Linear(256, 256)
        # self.fc4 = torch.nn.Linear(256, 196)
        # self.fc5 = torch.nn.Linear(128, 128)
        # self.fc6 = torch.nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.relu(self.fc2(x))
        # x = torch.nn.functional.relu(self.fc3(x))
        # x = torch.nn.functional.relu(self.fc4(x))
        # x = torch.nn.functional.relu(self.fc5(x))
        # x = torch.nn.functional.dropout(x, training=self.training)
        # x = self.fc6(x)
        return x