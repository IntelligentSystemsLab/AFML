import torch
import os
from net import LSTMNet
import time
from Client import Client
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
from torch import nn
from config_reader import TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX,REFRESH_FREQUENCY,LEN_INPUT

class FedAvg_LSTM(nn.Module):
  def __init__(self, city,device,num,):
    super(FedAvg_LSTM, self).__init__()
    self.device = device
    self.city = city
    self.time_len = LEN_INPUT
    self.number = num
    self.deep_frequency = REFRESH_FREQUENCY
    self.net = LSTMNet().to(self.device)
    train_path = r"./dataset/{}_save/train".format(self.city)
    test_path = r"./dataset/{}_save/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fed_train"
    self.mode_2 = "fed_test"
    self.writer = SummaryWriter('./results/{}/l_{}_s_{}_b_{}_i_{}_n_{}/FedAvg_LSTM'.format(self.city,LEN_INPUT,TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX,self.number))

    for index,path in enumerate(train_path_set):
      model = LSTMNet().to(self.device)
      self.clients.append(Client(model,index,path,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = LSTMNet().to(self.device)
      self.test_clients.append(Client(model,index,path,self.device,self.mode_2))

  def forward(self):
    pass

  def Fed_Train(self,round):
    time_set = []
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
        self.clients[j].refresh(self.net)
        self.clients[j].Local_Fed_Train()
        time_set.append(self.clients[j].time)
    
    time_start = time.time()
    with torch.no_grad():
        a = 0
        for id,j in enumerate(id_train_0):
            b = 0
            for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
                if (w is None or id == 0):
                    w_tem = Variable(torch.zeros_like(w)).to(self.device)
                    w.data.copy_(w_tem.data)
                if w_t is None:
                    w_t = Variable(torch.zeros_like(w)).to(self.device)
                w.data.add_(w_t.data)
        for w in self.net.parameters():
            w.data.div_(len(id_train_0))
    time_end = time.time()
    max_time = max(time_set)
    time_total = time_end - time_start + max_time
    self.writer.add_scalar('Time', time_total, round)
    self.writer.add_scalar('Num', len(id_train_0), round)
    
  def Fed_Test(self,round,):
    id_test = list(range(len(self.test_clients)))
    mae_list = []
    rmse_list = []
    mape_list = []
    r2_list = []
    for a,id in enumerate(id_test):
        self.test_clients[id].refresh(self.net)
        test_mae,test_rmse,test_mape,test_r2 = self.test_clients[id].Client_Test()
        mae_list.append(test_mae)
        rmse_list.append(test_rmse)
        mape_list.append(test_mape)
        r2_list.append(test_r2)
    mae_list = np.array(mae_list)
    rmse_list = np.array(rmse_list)
    mape_list = np.array(mape_list)
    r2_list = np.array(r2_list)
    mae_mean = mae_list.mean()
    rmse_mean = rmse_list.mean()
    mape_mean = mape_list.mean()
    r2_mean = r2_list.mean()
    print("Round {}:\n".format(round))
    print('all MAE: %.2f' % (mae_mean))
    print('all RMSE: %.2f' % (rmse_mean))
    print('all MAPE: %.2f' % (mape_mean))
    print('all R2: %.2f' % (r2_mean))

    self.writer.add_scalar('MAE', mae_mean, round)
    self.writer.add_scalar('RMSE', rmse_mean, round)
    self.writer.add_scalar('MAPE', mape_mean, round)
    self.writer.add_scalar('R2', r2_mean, round)