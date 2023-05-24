import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from data_reader import MyData
from copy import deepcopy,copy
from torch.autograd import Variable 
from torch.utils.data import DataLoader
import pandas as pd
import time
from scipy.signal import savgol_filter
from config_reader import LEARNING_RATE,META_LEARNING_RATE,REP_LEARNING_RATE,REPTILE_INNER_STEP,TEST_UPDATE_STEP,BATCH_SIZE,\
    BATCH_INDEX,LEN_INPUT,TEST_BATCH_INDEX,TEST_BATCH_SIZE,TRAIN_BATCH_SIZE
from metrics import masked_mape_np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class Client(nn.Module):
    def __init__(self,model,id,path,device,mode):
        super(Client, self).__init__()
        self.id = id
        self.update_step = BATCH_INDEX
        self.update_step_test = TEST_UPDATE_STEP
        self.rep_inner_step = REPTILE_INNER_STEP
        self.net = deepcopy(model)
        self.base_lr = LEARNING_RATE
        self.meta_lr = META_LEARNING_RATE
        self.rep_lr = REP_LEARNING_RATE
        self.mode = mode
        self.layer_idx = 0
        self.eta = 1.0
        data = pd.read_excel(path, engine='openpyxl')
        data = data["utility"]
        # if self.mode == "reptile_train":
        # data = savgol_filter(data, 21, 3, mode= 'nearest')
        # elif self.mode == "fed_train":
        data = savgol_filter(data, 21, 3, mode= 'nearest')
        data = data.astype('float16')
        data = torch.as_tensor(data, device=device).float().to(device)
        
        self.time = 0
        self.time_record = 0
        self.epoch = 0
        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True
        if self.mode == "fed_train":
            support_size = int(len(data)*0.8)
            support_target = data[:support_size]
            support_set = MyData(support_target, seq_len=LEN_INPUT,time_len = LEN_INPUT,device = device)
            self.support_loader = DataLoader(
            support_set, batch_size=BATCH_SIZE, shuffle=False)
        elif self.mode == "reptile_train":
            support_size = int(len(data)*0.8)
            support_target = data[:support_size]
            support_set = MyData(support_target, seq_len=LEN_INPUT,time_len = LEN_INPUT,device = device)
            self.support_loader = DataLoader(
            support_set, batch_size=BATCH_SIZE, shuffle=False)
        elif self.mode == "PerFed_train":
            support_size = int(len(data)*0.7)
            support_target = data[:support_size]
            support_set = MyData(support_target, seq_len=LEN_INPUT,time_len = LEN_INPUT,device = device)
            self.support_loader = DataLoader(
            support_set, batch_size=BATCH_SIZE, shuffle=False)
        else:
            support_size = int(len(data)*0.8)
            query_size = int(len(data)*0.9)
            index_size = int(len(data)*0.1)
            support_target = data[support_size:query_size]
            support_set = MyData(support_target, seq_len=LEN_INPUT,time_len = LEN_INPUT,device = device)
            self.support_loader = DataLoader(
            support_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

        query_size = int(len(data)*0.8)

        if self.mode == "fed_train":
            pass
        elif self.mode == "reptile_train":
            pass
        elif self.mode == "PerFed_train":
            query_target = data[support_size:query_size]
            query_set = MyData(query_target, seq_len=LEN_INPUT,time_len = LEN_INPUT,device = device)
            self.query_loader = DataLoader(
                query_set, batch_size = BATCH_SIZE, shuffle=False)
        else:
            query_size = int(len(data)*0.9)
            query_target = data[query_size:]
            query_set = MyData(query_target, seq_len=LEN_INPUT,time_len = LEN_INPUT,device = device)
            self.query_loader = DataLoader(
                query_set, batch_size=TEST_BATCH_SIZE, shuffle=False)
        self.optim = torch.optim.Adam(self.net.parameters(), lr = self.base_lr)
        self.meta_optim = torch.optim.SGD(self.net.parameters(), lr = self.meta_lr)
        self.rep_optim = torch.optim.SGD(self.net.parameters(), lr = self.rep_lr)
        self.size = query_size
        self.device = device
        self.lamda = 1.0
        self.loss_function = torch.nn.L1Loss().to(self.device)
        
    def forward(self):
        pass

    def Local_Rep_Train(self):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 200
        # 模型大小
        # server_model.summary()
        model_size =973
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for _ in range(self.rep_inner_step):
            net_tem = deepcopy(self.net)
            meta_optim_tem = torch.optim.Adam(net_tem.parameters(), lr = self.base_lr)
            for index,support in enumerate(self.support_loader):
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                meta_optim_tem.zero_grad()
                output = net_tem(support_x,)
                output = torch.flatten(output)
                loss = self.loss_function(output,support_y)
                loss.backward()
                meta_optim_tem.step()
                if index >= BATCH_INDEX:
                    break
            
            # tensor_l2_b = torch.zeros((1,)).cuda(self.device)
            self.rep_optim.zero_grad()
            for w, w_t in zip(self.net.parameters(), net_tem.parameters()):
                if w.grad is None:
                    w.grad = Variable(torch.zeros_like(w)).to(self.device)
                w.grad.data.add_(w.data - w_t.data)
                # tensor_l2_b = torch.cat((tensor_l2_b,w.grad.data.reshape(-1)))
            
            # gamma = torch.norm(tensor_l2_b)
            # # print(gamma)
            # for w in self.net.parameters():
            #     w.grad.data.mul_(gamma/0.01)
            self.rep_optim.step()

        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all
    
    def Local_Fomaml_Train(self):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 200
        # 模型大小
        model_size =973
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for _ in range(1):
            for batch_index, batch_data in enumerate(self.support_loader):
                support_x, support_y = batch_data
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output,support_y) 
                loss.backward()
                self.optim.step()
                if batch_index > BATCH_INDEX:
                    break
                
            for batch_index, batch_data in enumerate(self.query_loader):
                query_x, query_y = batch_data
                if torch.cuda.is_available():
                    query_x = query_x.cuda(self.device)
                    query_y = query_y.cuda(self.device)
                self.meta_optim.zero_grad()
                output = self.net(query_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, query_y)
                loss.backward()
                self.meta_optim.step()
                break

        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

    def Local_Fed_Train(self):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 200
        model_size =973
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for _ in range(self.update_step):
            for index,support in enumerate(self.support_loader):
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                self.optim.step()
                if index > BATCH_INDEX:
                    break

        self.optim.zero_grad()
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

    def Local_FedProx_Train(self):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 200
        model_size =973
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        mu = 0.001
        global_model = deepcopy(self.net)
        for _ in range(self.update_step):
            for index,support in enumerate(self.support_loader):
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                proximal_term = 0.0
                for w, w_t in zip(self.net.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, support_y) + (mu / 2) *  proximal_term
                loss.backward()
                self.optim.step()
                if index > BATCH_INDEX:
                    break

        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

    def Local_PFedMe_Train(self):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 200
        model_size =973
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        model_tem = deepcopy(self.net)
        optim_tem = torch.optim.Adam(model_tem.parameters(), lr = self.base_lr)
        for _ in range(self.update_step):
            for index,support in enumerate(self.support_loader):
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                optim_tem.zero_grad()
                output = model_tem(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                optim_tem.step()
                if index > BATCH_INDEX:
                    break
        
        for tem_0, tem in zip(model_tem.parameters(), self.net.parameters()):
            tem.data = tem.data - self.lamda * self.base_lr * (tem.data - tem_0.data)
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

    def Local_FedAsy_Train(self):
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 200
        model_size =973
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for _ in range(self.update_step):
            for index,support in enumerate(self.support_loader):
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                self.optim.step()
                if index > BATCH_INDEX:
                    break
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end - start) * 100
        t_all = t_c + tmp_tranmission

        self.time = t_all
    
    def Local_FedAsyED_Train(self,):
        self.old_model = deepcopy(self.net)
        self.tem_model = deepcopy(self.net)
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 200
        model_size =973
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for _ in range(self.update_step):
            for index,support in enumerate(self.support_loader):
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                self.optim.step()
                # if index > BATCH_INDEX:
                #     break

        for w,w_t,w_s in zip(self.net.parameters(),self.old_model.parameters(), self.tem_model.parameters()):
            if w.grad is None:
                w.grad = Variable(torch.zeros_like(w)).to(self.device)
            w_s.data.add_(w.data - w_t.data)
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 10000
        t_c = np.array(end - start) * 100
        t_all = t_c + tmp_tranmission

        self.time = t_all

    def refresh(self,model):
        for w,w_t in zip(self.net.parameters(),model.parameters()):
            w.data.copy_(w_t.data)
    
    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 
        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 
        Returns:
            None.
        """

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()


        # temp local model only for weight learning
        model_t = deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight

        for x, y in self.support_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            output = model_t(x)
            loss_value = self.loss_function(output, y) # modify according to the local objective
            loss_value.backward()

            # update weight in this batch
            for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                    params_gp, self.weights):
                weight.data = torch.clamp(
                    weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

            # update temp local model in this batch
            for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                    params_gp, self.weights):
                param_t.data = param + (param_g - param) * weight

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()

    def Client_Test(self):
        for _ in range(1,self.update_step_test+1):
            for index,support in enumerate(self.support_loader):
                self.optim.zero_grad()
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                output = self.net(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                self.optim.step()
                if index >= TEST_BATCH_INDEX:
                    break

        prediction = []
        ground_truth = []
        with torch.no_grad():
            for query in self.query_loader:
                query_x, query_y = query
                ground_truth.extend(list(query_y.cpu().numpy()))
                if torch.cuda.is_available():
                    query_x = query_x.cuda(self.device)
                    query_y = query_y.cuda(self.device)
                output = self.net(query_x)
                output = torch.flatten(output)
                prediction.extend(list(output.cpu().numpy()))
            
            data_target_tensor = np.array(ground_truth)
            prediction = np.array(prediction)
            mae = mean_absolute_error(data_target_tensor, prediction)
            rmse = mean_squared_error(data_target_tensor, prediction) ** 0.5
            mape = masked_mape_np(data_target_tensor, prediction, 0)  
            r2 = r2_score(data_target_tensor, prediction)

        return mae,rmse,mape,r2
    
    def Client_Test_new(self):
        mae_list = []
        rmse_list = []
        mape_list = []
        r2_list = []

        prediction = []
        ground_truth = []
        with torch.no_grad():
            for query in self.query_loader:
                query_x, query_y = query
                ground_truth.extend(list(query_y.cpu().numpy()))
                if torch.cuda.is_available():
                    query_x = query_x.cuda(self.device)
                    query_y = query_y.cuda(self.device)
                output = self.net(query_x)
                output = torch.flatten(output)
                prediction.extend(list(output.cpu().numpy()))
            
            data_target_tensor = np.array(ground_truth)
            prediction = np.array(prediction)
            mae = mean_absolute_error(data_target_tensor, prediction)
            rmse = mean_squared_error(data_target_tensor, prediction) ** 0.5
            mape = masked_mape_np(data_target_tensor, prediction, 0)  
            r2 = r2_score(data_target_tensor, prediction)
            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)
            r2_list.append(r2)

        for _ in range(1,21):
            for index,support in enumerate(self.support_loader):
                self.optim.zero_grad()
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                output = self.net(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                self.optim.step()
                break

            prediction = []
            ground_truth = []
            with torch.no_grad():
                for query in self.query_loader:
                    query_x, query_y = query
                    ground_truth.extend(list(query_y.cpu().numpy()))
                    if torch.cuda.is_available():
                        query_x = query_x.cuda(self.device)
                        query_y = query_y.cuda(self.device)
                    output = self.net(query_x)
                    output = torch.flatten(output)
                    prediction.extend(list(output.cpu().numpy()))
                
                data_target_tensor = np.array(ground_truth)
                prediction = np.array(prediction)
                mae = mean_absolute_error(data_target_tensor, prediction)
                rmse = mean_squared_error(data_target_tensor, prediction) ** 0.5
                mape = masked_mape_np(data_target_tensor, prediction, 0)  
                r2 = r2_score(data_target_tensor, prediction)
                mae_list.append(mae)
                rmse_list.append(rmse)
                mape_list.append(mape)
                r2_list.append(r2)

        return mae_list,rmse_list,mape_list,r2_list