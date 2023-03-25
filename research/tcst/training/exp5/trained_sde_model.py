'''
This code loads in a model to calculate the evolution of system state parameters (c10/c12) given an input temperature ramp rate, pressure ramp rate, 
and the values of the order parameters c10 and c12 at time 0. 

Additional files: 
(model state dictionary)
4fold_3_2_layer_model.pt

Optional additional files to run the example below
(State parameter trajectory files)
op_0.npy 
op_1.npy
(Ramp rate file)
hypercube_200.npy

'''

import numpy as np
import torch
from torch import nn, optim
import torchsde
from enum import Enum
import os

params_list = np.load("hyperparameter_v3.npy", allow_pickle=True)
LEARN_RATE = params_list[7][0]
BATCH_SIZE = params_list[7][1]
MODEL_NUMBER = params_list[7][2]

class SDE_3(nn.Module):

    def __init__(self):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
        self.network_f = nn.Sequential(
            nn.Linear(5, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 2)
        )

        self.network_g = nn.Sequential(
            nn.Linear(5, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 2)
        )

        self.r = None
    
    def get_temp_and_stress(self, t, start_val):
        temp_val = t[0]
        # Example: temperature should be set to 
        # 10 * ramp_rate + 120
        res = self.r[:, 0:2] * temp_val + self.start_val
        return res

    # drift
    def f(self, t, y):
        batch_size = y.shape[0]
        t = torch.reshape(t, [-1, 1])
        t = t.repeat((batch_size, 1))

        temp = self.get_temp_and_stress(t, self.start_val)
        #import pdb; pdb.set_trace()
        input_vec = torch.cat([y, temp, t], axis=1)
        #import pdb; pdb.set_trace()
        return self.network_f.forward(input_vec)
    
    # diffusion
    def g(self, t, y):
        """
        Output of g: should be a single tensor of size
        (batch_size, d)
        """
        batch_size = y.shape[0]
        t = torch.reshape(t, [-1, 1])
        t = t.repeat((batch_size, 1))

        temp = self.get_temp_and_stress(t, self.start_val)
        input_vec = torch.cat([y, temp, t], axis=1)
        return self.network_g.forward(input_vec)


def data_loader():
    ramp_rate_data = "hypercube_sample_4.npy"
    ramp_rate_np = np.load(ramp_rate_data)

    params = [10, 12]
    data_buffer = []
    for traj_idx in range(3):
        # (2, )
        ramp_rate_cur = ramp_rate_np[traj_idx]
        # (501, 2)
        ramp_rate_cur_tiled = np.tile(ramp_rate_cur, (501, 1))

        traj_state = []
        for param in params:
            filename = "op_2" + str(traj_idx) +  ".npy"
            # (501, )
            order_param_data = np.load(filename)[:, (param-1)].flatten()
            traj_state.append(order_param_data)
        
        # (2, 501)
        traj_state = np.array(traj_state)
        # (501, 2)
        traj_state = traj_state.T

        # (501, 4)
        full_traj_data = np.concatenate([traj_state, ramp_rate_cur_tiled], axis=1)
        data_buffer.append(full_traj_data)
    
    # (200, 501, 4)t.sh
    data_buffer = torch.tensor(np.array(data_buffer), dtype=torch.float32)

    return data_buffer


def build_loss(y_pred, y_gt):
    loss = nn.MSELoss()
    output = loss(y_pred, y_gt)
    return output

def moment_loss(y_pred, y_gt, batch_size):
    pred_mean = torch.mean(y_pred, axis=0)
    gt_mean = torch.mean(y_gt, axis=0)
    mean_arr = gt_mean - pred_mean
    mean_loss = torch.sum(mean_arr*mean_arr)
    pred_std = torch.std(y_pred, axis=0)*torch.std(y_pred, axis=0)
    gt_std = torch.std(y_gt, axis=0)*torch.std(y_gt, axis=0)
    std_arr = gt_std - pred_std
    std_loss = torch.sum(std_arr*std_arr)

    return torch.sum(mean_loss + std_loss) / batch_size

def get_next_step_predictions(model, data_buffer, rows, ts, batch_size):
    """
    - model: SDE object
    - data_buffer: (num_examples, num_time_steps+1, num_features)
    - rows: (batch_size, ), List of rows to sample from data_buffer.
    - ts: (num_time_steps, )
    - batch_size: (int).
    """
    y_gt = data_buffer[rows, 1:, :2]
    y_pred_list = []

    for time_step in range(len(ts)-1):
        y0 = data_buffer[rows, time_step, :2]
        y0 = torch.reshape(y0, [batch_size, -1])
        r = data_buffer[rows, 0, 2:4]
        r = torch.reshape(r, [-1, 2])
        start_val = data_buffer[rows, 0, 4:6]
        model.start_val = start_val
        model.r = r
        
        y_pred = torchsde.sdeint(model, y0, ts[time_step:time_step+2], dt=1e-1, method='euler')
        y_pred = y_pred.permute((1, 0, 2))
        
        y_pred_list.append(y_pred[:, 1:, :])
    
    y_pred_tensor = torch.cat(y_pred_list, axis=1)

    return y_gt, y_pred_tensor


def main():
    # load in example data
    eval_data = data_loader()
    # initialize neural network
    sde = SDE_3()
    t_size = 500 # set number of predictive time steps
    ts = torch.linspace(0, 200, t_size)
    # state path to model information file
    PATH_TO_MODEL = os.getcwd() + '/27hyperparameter_num_7_model.pt'
    # load model parameters
    sde.load_state_dict(torch.load(PATH_TO_MODEL))
    # switch items to GPU if avaliable
    if torch.cuda.is_available():
        print("Using GPU.")
        gpu = torch.device('cuda')
        sde = sde.to(gpu)
        eval_data = eval_data.to(gpu)
        ts = ts.to(gpu)
    else:
        print("Not using GPU.")
    # set model to evaluation mode
    sde.eval()
    batch_size = 1
    rows=0

    y_gt, y_pred_tensor = get_next_step_predictions(sde, eval_data, rows, ts, batch_size)
    
if __name__ == '__main__':
    main()