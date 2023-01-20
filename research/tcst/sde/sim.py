#!/usr/bin/env python

from trained_sde_model import *

import numpy as np

cuda0 = torch.device('cuda:0')
cpu = torch.device('cpu')
device = cuda0

def do_integration(
    control_data, d, T_0, T_t, mu_0, sigma_0, args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelpt',
        type=str, default='./4fold_3_2_layer_model.pt')

    parser.add_argument('--M',
        type=int,
        default=100,
        required=False)

    args = parser.parse_args()

    #####################################

    # initialize neural network
    sde = SDE()
    # state path to model information file
    # load model parameters
    sde.load_state_dict(torch.load(args.modelpt))
    if torch.cuda.is_available():
        print("Using GPU.")
        sde = sde.to(device)
    # set model to evaluation mode
    sde.eval()

    #####################################

    t_size = 500 # set number of predictive time steps
    ts = torch.linspace(0, 1, t_size)
    ts = ts.to(device)

    #####################################

    # load in example data
    eval_data = data_loader()
    eval_data = eval_data.to(device)

    traj_idx = 0
    y0 = eval_data[traj_idx % 2, 0, :2]
    y0 = torch.reshape(y0, [1, -1]) # reshape y) for correct input

    r = eval_data[traj_idx % 2, 0, 2:4]
    r_device = r.detach().cpu().numpy()
    r = torch.reshape(r, [-1, 2]) # reshape ramp rates for model input
    sde.r = r # assign r as sde.r for correct cat

    # y0_device = y0.detach().cpu().numpy()
    # initial_sample = np.random.multivariate_normal(
    #     y0_device,
    #     np.eye(y0.shape[0])*1.0,
    #     args.M)

    y_pred = torchsde.sdeint(
        sde, y0, ts, method='euler').squeeze() # calculate predictions

    import ipdb; ipdb.set_trace()

    # T_t = 10
    # t_size = T_t * 500 # set number of predictive time steps
    # ts = torch.linspace(0, T_t, t_size)
