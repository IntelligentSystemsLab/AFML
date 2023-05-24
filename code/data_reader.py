# coding = utf-8

import pandas as pd
import numpy as np
import csv
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def import_data(path,device):
    path_list = []
    name_dict = dict()
    data_dict = dict()
    
    
    path_list.extend(glob.glob(path+'/*.xlsx'))
    for n, npath in enumerate(path_list):
        name_dict[n] = npath[(len(path)+1):(npath.rfind('xlsx')-1)]
        temp = pd.read_excel(npath)
        data = temp['busy'].values.astype('float64')
        data = torch.tensor(data).float()
        data = data.to(device)
        data_dict[n] = data

    return name_dict, data_dict

class MyData(Dataset):
    def __init__(self, data, seq_len,time_len,device):
        self.sample_list = dict()
        self.label_list = dict()
        self.device = device
        num = 12*24-seq_len-time_len
        for n in range(12*24-seq_len-time_len,len(data) - seq_len - time_len):
            sample_0 = data[n:n+seq_len]
            sample_1 = data[n+seq_len+time_len-12*24].reshape(-1,)
            sample = torch.cat((sample_0,sample_1))
            label = data[n+seq_len+time_len]
            self.sample_list[n-num] = sample
            self.label_list[n-num] = label

    def __len__(self):
        return int(len(self.sample_list))

    def __getitem__(self, item):
        sample = self.sample_list[item]
        sample = torch.reshape(sample, (-1, 1))
        label = self.label_list[item]
        sample = sample.to(self.device)
        label = label.to(self.device)
        return sample, label
