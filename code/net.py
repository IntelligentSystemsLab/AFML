import torch.nn as nn
import torch
import torch.nn.functional as F
from config_reader import LEN_INPUT,INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,NUM_LAYERS_GRU,NUM_LAYERS_LSTM,NUM_LAYERS_RNN

class GRUNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS_GRU, batch_first=True)  # utilize the GRU model in torch.nn
        self.fc1 = nn.Linear(LEN_INPUT+1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, OUTPUT_SIZE)

    def forward(self, input):
        # x is input, size (batch, seq, feature)
        x_0 = input[:,:-1,:]
        x_1 = input[:,-1:,:]
        self.backbone.flatten_parameters()
        x, _ = self.backbone(x_0)
        
        x = x.transpose(1, 2)
        x = torch.cat((x,x_1),dim = 2)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x
    
class LSTMNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS_LSTM, batch_first=True)
        self.fc1 = nn.Linear(LEN_INPUT+1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, OUTPUT_SIZE)

    def forward(self, input):
        x_0 = input[:,:-1,:]
        x_1 = input[:,-1:,:]
        self.backbone.flatten_parameters()
        x, _ = self.backbone(x_0)
        
        x = x.transpose(1, 2)
        x = torch.cat((x,x_1),dim = 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x

class RNNNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.backbone = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS_RNN, batch_first=True) 
        self.fc1 = nn.Linear(LEN_INPUT+1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, OUTPUT_SIZE)

    def forward(self, input):
        x_0 = input[:,:-1,:]
        x_1 = input[:,-1:,:]
        self.backbone.flatten_parameters()
        x, _ = self.backbone(x_0)
        
        x = x.transpose(1, 2)
        x = torch.cat((x,x_1),dim = 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x

class GRUNet_BN(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS_GRU, batch_first=True)  # utilize the GRU model in torch.nn
        self.fc1 = nn.Linear(LEN_INPUT+1, 32)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(16, OUTPUT_SIZE)

    def forward(self, input):
        # x is input, size (batch, seq, feature)
        x_0 = input[:,:-1,:]
        x_1 = input[:,-1:,:]
        self.backbone.flatten_parameters()
        x, _ = self.backbone(x_0)
        
        x = x.transpose(1, 2)
        x = torch.cat((x,x_1),dim = 2)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x
    
class LSTMNet_BN(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS_LSTM, batch_first=True)
        self.fc1 = nn.Linear(LEN_INPUT+1, 32)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(16, OUTPUT_SIZE)

    def forward(self, input):
        x_0 = input[:,:-1,:]
        x_1 = input[:,-1:,:]
        self.backbone.flatten_parameters()
        x, _ = self.backbone(x_0)
        
        x = x.transpose(1, 2)
        x = torch.cat((x,x_1),dim = 2)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x

class RNNNet_BN(nn.Module):
    def __init__(self,):
        super().__init__()
        self.backbone = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS_RNN, batch_first=True) 
        self.fc1 = nn.Linear(LEN_INPUT+1, 32)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(16, OUTPUT_SIZE)

    def forward(self, input):
        x_0 = input[:,:-1,:]
        x_1 = input[:,-1:,:]
        self.backbone.flatten_parameters()
        x, _ = self.backbone(x_0)
        
        x = x.transpose(1, 2)
        x = torch.cat((x,x_1),dim = 2)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x