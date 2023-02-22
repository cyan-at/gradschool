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
    
        self.network_f = nn.Sequential(
            nn.Linear(4, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        '''
        Network for diffusion
        '''
        self.network_g = nn.Sequential(
            nn.Linear(4, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )

        '''
        Vector for ramp rates
        '''

        self.r = None

    # drift
    def f(self, t, y):
        t = torch.reshape(t, [-1, 1])
        # need to cat the ramp rates on the input vector for y
        input_vec = torch.cat([y,self.r, t], axis=1)
        return self.network_f.forward(input_vec)
    
    # diffusion
    def g(self, t, y):
        """
        Output of g: should be a single tensor of size
        (batch_size, d)
        """
        t = torch.reshape(t, [-1, 1])
        # need to cat the ramp rates on the input vector for g
        input_vec = torch.cat([y, self.r, t], axis=1)
        return self.network_g.forward(input_vec)


def data_loader():
    ramp_rate_data = os.getcwd() + "/hypercube_200.npy"
    ramp_rate_np = np.load(ramp_rate_data)

    params = [12]
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
    # load in example data
    eval_data = data_loader()
    # initialize neural network
    sde = SDE()
    t_size = 500 # set number of predictive time steps
    ts = torch.linspace(0, 200, t_size)
    # state path to model information file
    PATH_TO_MODEL = os.getcwd() + '/14fold_4_2_layer_model.pt'
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
    for traj in range(2):
        y0 = eval_data[traj % 2, 0, :1] # call sections of data loader corresponding to c10/c12 for first time step
        y0 = torch.reshape(y0, [1, -1]) # reshape y) for correct input
        r = eval_data[traj % 2, 0, 1:3] # call data loader for ramp rates 
        r = torch.reshape(r, [-1, 2]) # reshape ramp rates for model input
        sde.r = r # assign r as sde.r for correct cat

        # import ipdb; ipdb.set_trace()
        bm = torchsde.BrownianInterval(
            t0=float(0.0),
            t1=float(200),
            size=y0.shape,
            device=gpu,
        )  # We need space-time Levy area to use the SRK solver

        y_pred = torchsde.sdeint(sde, y0, ts, dt=1e-1, bm=bm, method='euler').squeeze() # calculate predictions
        # call data loader corresponding to c10/c12, this is the ground truth values of c10/c12, only here for comparison, not for running the model
        y_gt = eval_data[traj % 2, 1:, :1].squeeze()

        print(r)

        # import ipdb; ipdb.set_trace()
        y_pred_down = y_pred.cpu().detach().numpy()
        y_gt_down = y_gt.cpu().detach().numpy()
        # tbl = np.vstack((y_gt_down, y_pred_down)).T

        dat = {
            'gt' : y_gt_down,
            'pred' : y_pred_down
        }
        np.save('sde_model_%d.npy' % (traj), dat)

if __name__ == '__main__':
    main()