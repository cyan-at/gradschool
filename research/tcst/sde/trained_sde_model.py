#!/usr/bin/env python

'''
This code loads in a model to calculate the evolution of system state parameters (c10/c12) given an input temperature ramp rate, pressure ramp rate, 
and the values of the order parameters c10 and c12 at time 0. 

Additional files: 
(model state dictionary)
4fold_3_2_layer_model.pt

Optional additional files to run the example below
(State parameter trajectory files)
op_1.npy 
op_2.npy
(Ramp rate file)
hypercube_200.npy

'''

import numpy as np
import torch
from torch import nn, optim
import torchsde
from enum import Enum
import os

import argparse

import torch.nn.functional as F

class CustomSequential(nn.Sequential):
    def forward(self, input):
        for module in self:
            # input = module(input)
            if type(module) == nn.Linear:
                input = module(input)

                # input = F.linear(input,
                #     module.weight.detach(),
                #     module.bias.detach())

                # import ipdb; ipdb.set_trace()
                # print(module.weight[0:3, :5])
            else:
                # print(type(module))
                input = module(input)
        return input

class SDE(nn.Module):
    '''
    Initialize neural network module
    '''

    def __init__(self):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        '''
        Network for drift
        '''

        self.network_fs = [
            nn.Linear(5, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 2)
        ]
    
        self.network_f = CustomSequential(
            self.network_fs[0],
            nn.Tanh(),
            self.network_fs[1],
            nn.Tanh(),
            self.network_fs[2]
        )
        '''
        Network for diffusion
        '''
        self.network_gs = [
            nn.Linear(5, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 2)
        ]
        self.network_g = CustomSequential(
            self.network_gs[0],
            nn.Tanh(),
            self.network_gs[1],
            nn.Tanh(),
            self.network_gs[2]
        )

        # import ipdb; ipdb.set_trace()

        '''
        Vector for ramp rates
        '''

        self.r = None

    # drift
    def f(self, t, y): # ~D1
        t = torch.reshape(t, [-1, 1])
        # need to cat the ramp rates on the input vector for y
        input_vec = torch.cat([y,self.r, t], axis=1)

        # print(self.network_f.forward(input_vec).shape)

        return self.network_f.forward(input_vec)
    
    # diffusion
    def g(self, t, y): # ~D2
        """
        Output of g: should be a single tensor of size
        (batch_size, d)
        """
        t = torch.reshape(t, [-1, 1])
        # need to cat the ramp rates on the input vector for g
        input_vec = torch.cat([y, self.r, t], axis=1)

        # print("g", self.network_g.forward(input_vec))

        return self.network_g.forward(input_vec)


class SDE2(SDE):
    '''
    Initialize neural network module
    '''

    def __init__(self, control_data):
        super(SDE2, self).__init__()
        
        self.control_data = control_data

    def query_u(self, t, y):
        if torch.abs(t - 0) < 1e-8:
            t_key = 't0'
        elif torch.abs(t - 1.0) < 1e-8:
            t_key = 'tT'
        else:
            t_key = 'tt'
        t_control_data = self.control_data[t_key]

        query = y[0].detach().cpu().numpy()

        if t_control_data['grid'].shape[1] == 2 + 1:
            t2 = t.detach().cpu().numpy()
            query = np.append(query, t2)

        if 'grid_tree' in t_control_data:
            _, closest_grid_idx = t_control_data['grid_tree'].query(
                np.expand_dims(query, axis=0),
                k=1)
        else:
            closest_grid_idx = np.linalg.norm(
                query - t_control_data['grid'], ord=1, axis=1).argmin()

        u1 = t_control_data['0'][closest_grid_idx]
        u2 = t_control_data['1'][closest_grid_idx]

        u_tensor = torch.tensor(np.array([u1, u2]), dtype=torch.float32)
        u_tensor = u_tensor.reshape([-1, 2])

        return u_tensor

    # drift
    def f(self, t, y): # ~D1
        u_tensor = self.query_u(t, y)

        t = torch.reshape(t, [-1, 1])
        # need to cat the ramp rates on the input vector for y
        input_vec = torch.cat([y, u_tensor, t], axis=1)

        # print(self.network_f.forward(input_vec).shape)

        return self.network_f.forward(input_vec)
    
    # diffusion
    def g(self, t, y): # ~D2
        """
        Output of g: should be a single tensor of size
        (batch_size, d)
        """
        u_tensor = self.query_u(t, y)

        t = torch.reshape(t, [-1, 1])

        # need to cat the ramp rates on the input vector for g
        input_vec = torch.cat([y, u_tensor, t], axis=1)

        # print("g", self.network_g.forward(input_vec))

        return self.network_g.forward(input_vec)

def data_loader():
    ramp_rate_data = os.getcwd() + "/hypercube_200.npy"
    ramp_rate_np = np.load(ramp_rate_data)

    params = [10, 12]
    data_buffer = []
    for traj_idx in range(2):
        # (2, )
        ramp_rate_cur = ramp_rate_np[traj_idx]
        # (501, 2)
        ramp_rate_cur_tiled = np.tile(ramp_rate_cur, (501, 1))

        traj_state = []
        for param in params:
            filename = os.getcwd() + "/op_" + str(traj_idx) +  ".npy"
            # (501, )
            order_param_data = np.load(filename)[:, (param-1)].flatten()
            traj_state.append(order_param_data)

        # import ipdb; ipdb.set_trace()
        
        # (2, 501)
        traj_state = np.array(traj_state)
        # (501, 2)
        traj_state = traj_state.T

        # (501, 4)
        full_traj_data = np.concatenate([traj_state, ramp_rate_cur_tiled], axis=1)
        data_buffer.append(full_traj_data)
    
    # 92, 501, 4)
    data_buffer = torch.tensor(np.array(data_buffer), dtype=torch.float32)

    return data_buffer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelpt',
        type=str, default='./4fold_3_2_layer_model.pt')

    args = parser.parse_args()

    # load in example data
    eval_data = data_loader()
    # initialize neural network
    sde = SDE()
    t_size = 500 # set number of predictive time steps
    ts = torch.linspace(0, 50, t_size)
    # state path to model information file
    # load model parameters
    sde.load_state_dict(torch.load(args.modelpt))
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

    '''
    BCC:
    C10:0.41235
    C12:0.37605
    FCC: 
    C10:0.012857
    C12:0.60008
    SC:
    C10:0.41142
    C12: 0.69550
    '''
    bcc = np.array([0.41235, 0.37605])
    fcc = np.array([0.012857, 0.60008])
    sc = np.array([0.41142, 0.69550])

    all_trajs = {}

    for traj in range(2):
        y0 = eval_data[traj % 2, 0, :2] # call sections of data loader corresponding to c10/c12 for first time step
        y0 = torch.reshape(y0, [1, -1]) # reshape y) for correct input
        print("y0", y0)
        r = eval_data[traj % 2, 0, 2:4] # call data loader for ramp rates 
        print("r", r)
        r = torch.reshape(r, [-1, 2]) # reshape ramp rates for model input
        sde.r = r # assign r as sde.r for correct cat


        # import ipdb; ipdb.set_trace()

        y_pred = torchsde.sdeint(sde, y0, ts, method='euler').squeeze() # calculate predictions

        # call data loader corresponding to c10/c12, this is the ground truth values of c10/c12, only here for comparison, not for running the model
        y_gt = eval_data[0, 1:, :2]

        # import ipdb; ipdb.set_trace();

        y_pred_down = y_pred.cpu().detach().numpy()
        y_gt_down = y_gt.cpu().detach().numpy()

        all_trajs[traj] = {
            'pred' : y_pred_down,
            'gt' : y_gt_down,
            'bcc' : bcc * np.ones_like(y_pred_down),
            'fcc' : fcc * np.ones_like(y_pred_down),
            'sc' : sc * np.ones_like(y_pred_down),
        }

    np.save('all_trajs.npy', all_trajs)
    
if __name__ == '__main__':
    main()