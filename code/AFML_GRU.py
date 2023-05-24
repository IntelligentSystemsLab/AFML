import torch
import os
from net import GRUNet_BN
import time
from Client import Client
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
from torch import nn
from config_reader import TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX,REFRESH_FREQUENCY,LEN_INPUT

class AFML_GRU(nn.Module):
  def __init__(self, city,device,num,):
    super(AFML_GRU, self).__init__()
    self.device = device
    self.city = city
    self.time_len = LEN_INPUT
    self.number = num
    self.deep_frequency = REFRESH_FREQUENCY
    self.net = GRUNet_BN().to(self.device)
    train_path = r"./dataset/{}_save/train".format(self.city)
    test_path = r"./dataset/{}_save/test".format(self.city)
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.clients = []
    self.test_clients = []
    self.mode_1 = "reptile_train"
    self.mode_2 = "reptile_test"
    self.writer = SummaryWriter('./results/{}/l_{}_s_{}_b_{}_i_{}_n_{}/AFML_GRU'.format(self.city,LEN_INPUT,TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX,self.number))

    for index,path in enumerate(train_path_set):
      model = GRUNet_BN().to(self.device)
      self.clients.append(Client(model,index,path,self.device,self.mode_1))
    for index,path in enumerate(test_path_set):
      model = GRUNet_BN().to(self.device)
      self.test_clients.append(Client(model,index,path,self.device,self.mode_2))

  def forward(self):
    pass

  def Fed_Train(self,round):
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].Local_Rep_Train()
        self.clients[j].epoch = round
      else:
        continue
    
    time_start = time.time()
    id_train = []
    size_all = 0
    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - 40,0)
      if self.clients[id].time <= 0:
        id_train.append(id)
        size_all += self.clients[id].size

    weight = []
    for id,j in enumerate(id_train):
      weight.append(np.power(np.exp(1),self.clients[j].epoch-round))
    weight = np.array(weight)
    weight = weight / weight.sum()
    
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

                w.data.add_(w_t.data*weight[id])
                b += 1
                if b >= 12 & round%self.deep_frequency != 0:
                    break

    time_end = time.time()
    time_total = time_end - time_start + 20
    self.writer.add_scalar('Time', time_total, round)
    self.writer.add_scalar('Num', len(id_train), round)
    
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