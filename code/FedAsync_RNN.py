import torch
import os
from net import RNNNet
import time
from Client import Client
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
from torch import nn
from config_reader import TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX,REFRESH_FREQUENCY,LEN_INPUT

class FedAsync_RNN(nn.Module):
  def __init__(self, city,device,num,):
    super(FedAsync_RNN, self).__init__()
    self.device = device
    self.city = city
    self.time_len = LEN_INPUT
    self.number = num
    self.deep_frequency = REFRESH_FREQUENCY
    self.net = RNNNet().to(self.device)
    train_path = r"./dataset/{}_save/train".format(self.city)
    test_path = r"./dataset/{}_save/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set][:10]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fed_train"
    self.mode_2 = "fed_test"
    self.writer = SummaryWriter('./results/{}/l_{}_s_{}_b_{}_i_{}_n_{}/FedAsync_RNN'.format(self.city,LEN_INPUT,TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX,self.number))

    for index,path in enumerate(train_path_set):
      model = RNNNet().to(self.device)
      self.clients.append(Client(model,index,path,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = RNNNet().to(self.device)
      self.test_clients.append(Client(model,index,path,self.device,self.mode_2))

  def forward(self):
    pass

  def Fed_Train(self,round):
    time_set = []
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].Local_FedAsy_Train()
        time_set.append(self.clients[j].time)
      else:
        continue
    
    time_start = time.time()
    time_1 = []
    id_train = []
    for id in id_train_0:
      time_1.append(self.clients[id].time)
    time_1 = np.array(time_1)
    min_time = time_1.min()

    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - min_time,0)
      if self.clients[id].time <= 0:
        id_train.append(id)

    weight = []
    for id,j in enumerate(id_train):
      weight.append(np.power(self.clients[j].time_record+1,-0.5))
    weight = np.array(weight)
    with torch.no_grad():
      a = 0
      for id,j in enumerate(id_train):
        b = 0
        for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
          if (w is None or id == 0):
            w_tem = Variable(torch.zeros_like(w)).to(self.device)
            w.data.copy_(w_tem.data)
          if w_t is None:
            w_t = Variable(torch.zeros_like(w)).to(self.device)
          w.data.mul_(1-weight[id])
          w.data.add_(w_t.data*weight[id])
    time_end = time.time()
    time_total = time_end - time_start + min_time
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