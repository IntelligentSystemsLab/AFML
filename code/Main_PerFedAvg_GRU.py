import pandas as pd
import torch
import os
from PerFedAvg_GRU import PerFedAvg_GRU
from config_reader import CITY,LEN_INPUT,TRAINING_EPOCHS,RESULT_SAVE_NUMBER_START,RESULT_SAVE_NUMBER_END,TEST_FREQUENCY,TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX

def main(model_name,device_num,epoch,num):
    folder = r"./model/{}".format(CITY)
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = r"./model/{}/l_{}_s_{}_b_{}_i_{}_n_{}".format(CITY,LEN_INPUT,TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX,num)
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = os.path.join(folder,model_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    fed_net = PerFedAvg_GRU(CITY,device_num,num)
    for i in range(1,1+epoch):
        print("{} round training.".format(i))
        if i == 1:
            fed_net.Fed_Test(0,) 
            torch.save({'model': fed_net.state_dict()},os.path.join(folder,"model_epoch_{}.pth".format(0)))
        fed_net.Fed_Train(i)
        if i%TEST_FREQUENCY == 0:
            fed_net.Fed_Test(i,) 
            torch.save({'model': fed_net.state_dict()},os.path.join(folder,"model_epoch_{}.pth".format(i)))

if __name__ == '__main__':
    model_name = 'PerFedAvg_GRU'
    device_num = 2
    epoch = TRAINING_EPOCHS
    for i in range(RESULT_SAVE_NUMBER_START,RESULT_SAVE_NUMBER_END):
        main(model_name,device_num,epoch,i)
