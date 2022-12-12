#!/usr/bin/env python3

from scipy.stats import truncnorm, norm
import numpy as np, os, time, sys

import torch

os.environ['DDE_BACKEND'] = "pytorch" # v2
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf"
# https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path
# this directory has /nvvm/libdevice/libdevice.10.bc
print(os.environ['DDE_BACKEND'])
import deepxde as dde

from deepxde import backend as bkd
from deepxde.backend import backend_name
from deepxde.utils import get_num_args, run_if_all_none

import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    print("no tkagg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

cuda0 = torch.device('cuda:0')
cpu = torch.device('cpu')
device = cuda0

def pdf3d(x,y,z,rv):
    return rv.pdf(np.hstack((x, y, z)))

def slice(matrix, i, j, mode):
    if mode == 0:
        return matrix[:, i, j]
    elif mode == 1:
        return matrix[j, :, i]
    else:
        return matrix[i, j, :]

def slice2d(matrix, i, mode):
    if mode == 0:
        return matrix[:, i]
    else:
        return matrix[i, :]

def get_pdf_support(matrix, xs):
    d = len(xs)

    if d == 3:
        # flatten z down into xy cell
        # flatten x down to y row
        # flatten y into number
        buffer0 = np.zeros(len(xs[0]))
        buffer1 = np.zeros(len(xs[1]))
        for j in range(len(xs[1])):
            for i in range(len(xs[0])):
                buffer0[i] = np.trapz(
                    slice(matrix, i, j, 2),
                    x=xs[2])
                # buffer0[i] = trapz out z
                # buffer[i][j] = matrix[i][j][:]
            # buffer0 is now filled with accumulated z
            buffer1[j] = np.trapz(
                buffer0,
                x=xs[0])
            # buffer1[j] = trapz out x
            # buffer[j] = matrix[:][j][:]
        return np.trapz(buffer1, x=xs[1])
    elif d == 2:
        # flatten x into y row
        # flatten y into number
        buffer1 = np.zeros(len(xs[1]))
        for i in range(len(xs[1])):
            buffer1[i] = np.trapz(
                slice2d(matrix, i, 0),
                x=xs[0])
            # buffer1[i] = trapz out x
            # buffer1[i] = matrix[:, i]
        return np.trapz(buffer1, x=xs[1])

def get_pdf_support_torch(
    tensor,
    xtensors,
    dt=torch.float32):
    d = len(xtensors)
    n = len(xtensors[0])

    tensor_matrix = torch.reshape(
        tensor, tuple([n]*d))

    if d == 3:
        # flatten z down into xy cell
        # flatten x down to y row
        # flatten y into number    
        buffer0 = torch.zeros(len(xtensors[0]))
        buffer1 = torch.zeros(len(xtensors[1]))
        for j in range(len(xtensors[1])):
            for i in range(len(xtensors[0])):
                buffer0[i] = torch.trapz(
                    slice(tensor_matrix, i, j, 2)
                    , x=xtensors[2])
            buffer1[j] = torch.trapz(
                buffer0,
                x=xtensors[0])
        return torch.trapz(buffer1, x=xtensors[1])
    elif d == 2:
        buffer1 = torch.zeros(len(xtensors[1]), dtype=dt)
        for i in range(len(xtensors[1])):
            buffer1[i] = torch.trapz(
                slice2d(tensor_matrix, i, 0),
                x=xtensors[0])
        return np.trapz(buffer1, x=xs[1])

def get_marginal_pmf(matrix, xs, mode):
    d = len(xs)
    if d == 3:
        # mode == 0, smoosh out 'x', then 'y' => z marginal
        # mode == 1, smoosh out 'y', then 'z' => x marginal
        # mode == 2, smoosh out 'z', then 'x' => y marginal
        marginal = np.array([
            np.sum(
                np.array([
                    np.sum(
                        slice(matrix, i, j, mode)
                    ) # smoosh out axis=mode
                    for i in range(len(xs[(mode + 1) % d]))
                ])
            ) # x2 slice for one x1 => R
        for j in range(len(xs[(mode + 2) % d]))])
        return marginal
    elif d == 2:
        # mode == 0, flatten 'x' => y marginal
        # mode == 1, flatten 'y' => x marginal
        marginal = np.array([
            np.sum(
                slice2d(matrix, i, mode)
            )
            for i in range(len(xs[mode]))
        ])
        return marginal

def get_pmf_stats(pmf, mesh_vectors, linspaces, normalize=True):
    # pmf has SUM=1, so normalize as such
    pmf_normed = pmf
    if normalize:
        pmf_support = np.sum(pmf)
        pmf_normed = pmf / pmf_support

    d = len(linspaces)
    n = len(linspaces[0])
    pmf_cube = pmf_normed.reshape(tuple([n]*len(linspaces)))

    mus = np.array([0.0]*d)
    marginals = []
    deltas = []

    for i in range(d):
        # finding mu / E(x1) of discrete distribution / pmf
        # is implemented as a dot product
        marginal_pmf = get_marginal_pmf(
            pmf_cube, linspaces, (i+1) % d)
        mus[i] = np.dot(marginal_pmf, linspaces[i])

        marginals.append(marginal_pmf)
        deltas.append(mesh_vectors[i] - mus[i])

    cov_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov_matrix[i][j] = np.sum(pmf_normed * deltas[i] * deltas[j])

    return mus, cov_matrix, pmf_cube, marginals

def get_marginal_pmf_torch(
    tensor_matrix,
    xtensors,
    mode):
    d = len(xtensors)
    if d == 3:
        # mode == 0, smoosh out 'x', then 'y' => z marginal
        # mode == 1, smoosh out 'y', then 'z' => x marginal
        # mode == 2, smoosh out 'z', then 'x' => y marginal

        q = len(xtensors[(mode + 2) % d])
        marginal = torch.zeros(q)
        for j in range(q):
            t = len(xtensors[(mode + 1) % d])
            sums = torch.zeros(t)
            for i in range(t):
                sums[i] = torch.sum(slice(tensor_matrix, i, j, mode))
            marginal[j] = torch.sum(sums)

        # import ipdb; ipdb.set_trace()
        return marginal
    elif d == 2:
        # mode == 0, flatten 'x' => y marginal
        # mode == 1, flatten 'y' => x marginal
        q = len(xtensors[mode])
        marginal = torch.zeros(q)
        for i in range(q):
            marginal[i] = torch.sum(
                slice2d(matrix, i, mode))
        return marginal

def get_pmf_stats_torch(
    pmf,
    mesh_vectors,
    linspaces,
    dt):
    pmf_support = torch.sum(pmf)

    pmf_normed = pmf / pmf_support

    d = len(linspaces)
    n = len(linspaces[0])
    pmf_cube = pmf_normed.reshape(tuple([n]*len(linspaces)))

    mus = torch.zeros((d))
    marginals = []
    deltas = []

    for i in range(d):
        # finding mu / E(x1) of discrete distribution / pmf
        # is implemented as a dot product
        marginal_pmf = get_marginal_pmf_torch(
            pmf_cube, linspaces, (i+1) % d)
        mus[i] = torch.matmul(
            marginal_pmf,
            linspaces[i]
        )

        marginals.append(marginal_pmf)
        deltas.append(mesh_vectors[i] - mus[i])

    cov_matrix = torch.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov_matrix[i][j] = torch.sum(pmf_normed * deltas[i] * deltas[j])

    return mus, cov_matrix

def get_multivariate_truncated_pdf(x_T, y_T, z_T, mu, sigma, state_min, state_max, N, f, cache_name):
    state = np.hstack((x_T, y_T, z_T))
    if not os.path.exists(cache_name):
        import julia
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        jl.eval('import Pkg; Pkg.add("Primes"); Pkg.add("Parameters"); Pkg.add("Distributions");')
        jl.eval('')
        jl.eval('include("/home/cyan3/Dev/jim/TMvNormals.jl/src/TMvNormals.jl"); using .TMvNormals; using LinearAlgebra')
        make_d = jl.eval('(mu, sigma, state_min, state_max) -> d = TMvNormal(mu*ones(3),sigma*I(3),[state_min, state_min, state_min],[state_max, state_max, state_max])')
        trunc_pdf = jl.eval('(d, x) -> pdf(d, x)')
        trunc_mean = jl.eval('d -> mean(d)')
        trunc_cov = jl.eval('d -> cov(d)')

        trunc_rv = make_d(mu, sigma, state_min, state_max)

        #################################################

        trunc_pdf = np.array([
            trunc_pdf(trunc_rv, state[i, :]) for i in range(state.shape[0])
            ]).astype(f).reshape(N**3, 1)
        np.savetxt(cache_name, trunc_pdf)
    else:
        trunc_pdf = np.loadtxt(cache_name).astype(f).reshape(N**3, 1)

    return trunc_pdf

def sinkhorn_torch(K,
    c_tensor,
    a_tensor,
    b_tensor,
    u_vec,
    v_vec,
    p_opt,
    device,
    delta=1e-1,
    lam=1e-6):
    if u_vec.grad is not None:
        u_vec.grad.zero_()

    if v_vec.grad is not None:
        v_vec.grad.zero_()

    if p_opt.grad is not None:
        p_opt.grad.zero_()

    u_vec = torch.ones(a_tensor.shape[0], dtype=torch.float32).to(device)
    v_vec = torch.ones(b_tensor.shape[0], dtype=torch.float32).to(device)

    u_trans = torch.matmul(K, v_vec) + lam  # add regularization to avoid divide 0
    v_trans = torch.matmul(K.T, u_vec) + lam  # add regularization to avoid divide 0

    err_1 = torch.sum(torch.abs(u_vec * u_trans - a_tensor))
    err_2 = torch.sum(torch.abs(v_vec * v_trans - b_tensor))

    # import ipdb; ipdb.set_trace()

    while True:
        if (err_1 + err_2).item() > delta:
            u_vec = torch.div(a_tensor, u_trans)
            v_trans = torch.matmul(K.T, u_vec) + lam

            v_vec = torch.div(b_tensor, v_trans)
            u_trans = torch.matmul(K, v_vec) + lam

            err_1 = torch.sum(
                torch.abs(u_vec * u_trans - a_tensor))
            err_2 = torch.sum(
                torch.abs(v_vec * v_trans - b_tensor))

            # print("err_1 + err_2", (err_2 + err_1).item() > delta)

            # import ipdb; ipdb.set_trace()
        else:
            # print("DONE!")
            break

    print(v_vec)

    p_opt = torch.linalg.multi_dot([
        torch.diag(v_vec),
        K,
        torch.diag(u_vec)])

    w = torch.dot(c_tensor, p_opt.view(-1))
    # print(w)

    return w

N = 15

# must be floats
state_min = -5.0
state_max = 5.0

mu_0 = 2.0
sigma_0 = 0.5

mu_T = 0.0
sigma_T = 0.5

j1, j2, j3 =1.0,1.0,2.0 # axis-symmetric case
q_statepenalty_gain = 0 # 0.5

T_0=0. #initial time
T_t=5. #Terminal time

id_prefix = "wass_3d"
de = 1

f = 'float32'
dt = torch.float32

samples_between_initial_and_final = 12000 # 10^4 order, 20k = out of memory
initial_samples = 500 # some 10^3 order

num_epochs = 100000

epsilon=.001

########################################################

X_IDX = 0
Y_IDX = 1
Z_IDX = 2
T_IDX = 3
PSI_IDX = 4
RHO_OPT_IDX = 5

'''
def plot_rho_bc(label, test_ti, mu, sigma, ax):
    ind = np.lexsort((test_ti[:,X_IDX], test_ti[:,T_IDX]))
    test_ti = test_ti[ind]

    test_ti[:,RHO_OPT_IDX] = np.where(
      test_ti[:,RHO_OPT_IDX] < 0, 0, test_ti[:,RHO_OPT_IDX])
    s1 = np.trapz(test_ti[:,RHO_OPT_IDX],
        axis=0,
        x=test_ti[:,X_IDX])
    test_ti[:, RHO_OPT_IDX] /= s1 # to pdf
    s2 = np.sum(test_ti[:,RHO_OPT_IDX])
    test_ti[:, RHO_OPT_IDX] /= s2 # to pmf

    s1 = np.trapz(test_ti[:,RHO_OPT_IDX],
        axis=0,
        x=test_ti[:,X_IDX])
    s2 = np.sum(test_ti[:,RHO_OPT_IDX])

    true_rho=pdf1d(test_ti[:, X_IDX], mu, sigma).reshape(test_ti.shape[0],1)
    true_rho /= np.trapz(true_rho, axis=0, x=test_ti[:,X_IDX])
    true_rho = true_rho / np.sum(np.abs(true_rho))

    ax.plot(
        test_ti[:, X_IDX],
        test_ti[:, RHO_OPT_IDX],
        c='b',
        linewidth=1,
        label='test %s' % (label))
    ax.plot(
        test_ti[:, X_IDX],
        true_rho,
        c='r',
        linewidth=1,
        label='true %s' % (label))

    return s1, s2
'''

def euler_pde_1(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    # dy1_y = dde.grad.jacobian(y1, x, j=1)
    # dy1_z = dde.grad.jacobian(y1, x, j=2)
    dy1_y = 0.0
    dy1_z = 0.0
    dy1_t = dde.grad.jacobian(y1, x, j=1)

    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    # dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    # dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)
    dy1_yy = 0.0
    dy1_zz = 0.0

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    # dy2_y = dde.grad.jacobian(y2, x, j=1)
    # dy2_z = dde.grad.jacobian(y2, x, j=2)
    # dy2_y = 0.0
    dy2_z = 0.0
    dy2_t = dde.grad.jacobian(y2, x, j=1)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    # dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    # dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)
    dy2_yy = 0.0
    dy2_zz = 0.0

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=1.0*1.0*(j2-j3)/j1
    # f1=x[:, 1:2]*1.0*(j2-j3)/j1
    f2=x[:, 0:1]*1.0*(j3-j1)/j2
    # f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    # f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    # f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    f3=x[:, 0:1]*1.0*(j1-j2)/j3

    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y+dy1_z*dy1_z) - (dy1_x*f1 + dy1_y*f2 + dy1_z*f3) - epsilon*(dy1_xx+dy1_yy+dy1_zz)

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y+d_f3dy1_y2_z)+epsilon*(dy2_xx+dy2_yy+dy2_zz),
    ]

def euler_pde_2(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_y = dde.grad.jacobian(y1, x, j=1)
    # dy1_z = dde.grad.jacobian(y1, x, j=2)
    # dy1_y = 0.0
    dy1_z = 0.0
    dy1_t = dde.grad.jacobian(y1, x, j=2)

    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    # dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)
    # dy1_yy = 0.0
    dy1_zz = 0.0

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    dy2_y = dde.grad.jacobian(y2, x, j=1)
    # dy2_z = dde.grad.jacobian(y2, x, j=2)
    # dy2_y = 0.0
    dy2_z = 0.0
    dy2_t = dde.grad.jacobian(y2, x, j=2)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    # dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)
    # dy2_yy = 0.0
    dy2_zz = 0.0

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=x[:, 1:2]*1.0*(j2-j3)/j1
    f2=x[:, 0:1]*1.0*(j3-j1)/j2
    # f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    # f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    # f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    # d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y) - (dy1_x*f1 + dy1_y*f2) - epsilon*(dy1_xx+dy1_yy)

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y)+epsilon*(dy2_xx+dy2_yy),
    ]

def euler_pde_3(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_y = dde.grad.jacobian(y1, x, j=1)
    dy1_z = dde.grad.jacobian(y1, x, j=2)
    dy1_t = dde.grad.jacobian(y1, x, j=3)
    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    dy2_y = dde.grad.jacobian(y2, x, j=1)
    dy2_z = dde.grad.jacobian(y2, x, j=2)
    dy2_t = dde.grad.jacobian(y2, x, j=3)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y+dy1_z*dy1_z) - (dy1_x*f1 + dy1_y*f2 + dy1_z*f3) - epsilon*(dy1_xx+dy1_yy+dy1_zz)

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y+d_f3dy1_y2_z)+epsilon*(dy2_xx+dy2_yy+dy2_zz),
    ]

U1 = dde.Variable(torch.ones(samples_between_initial_and_final + initial_samples + 2*500), dtype=torch.float32)
def euler_pde_4(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_y = dde.grad.jacobian(y1, x, j=1)
    dy1_z = dde.grad.jacobian(y1, x, j=2)
    dy1_t = dde.grad.jacobian(y1, x, j=3)
    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    dy2_y = dde.grad.jacobian(y2, x, j=1)
    dy2_z = dde.grad.jacobian(y2, x, j=2)
    dy2_t = dde.grad.jacobian(y2, x, j=3)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y+dy1_z*dy1_z) - (dy1_x*f1 + dy1_y*f2 + dy1_z*f3) - epsilon*(dy1_xx+dy1_yy+dy1_zz)

    dpsi_x = dde.grad.jacobian(psi, x, j=0)

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y+d_f3dy1_y2_z)+epsilon*(dy2_xx+dy2_yy+dy2_zz),
        U1 - dpsi_x,
        0.1 / U1, # scaled, not that important
    ]

def euler_pde_5(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_y = dde.grad.jacobian(y1, x, j=1)
    dy1_z = dde.grad.jacobian(y1, x, j=2)
    dy1_t = dde.grad.jacobian(y1, x, j=3)
    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    dy2_y = dde.grad.jacobian(y2, x, j=1)
    dy2_z = dde.grad.jacobian(y2, x, j=2)
    dy2_t = dde.grad.jacobian(y2, x, j=3)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y+dy1_z*dy1_z) - (dy1_x*f1 + dy1_y*f2 + dy1_z*f3) - epsilon*(dy1_xx+dy1_yy+dy1_zz)

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y+d_f3dy1_y2_z)+epsilon*(dy2_xx+dy2_yy+dy2_zz),
        0.01/(torch.min(torch.abs(dy1_x)) + 1e-10),
        0.01/(torch.min(torch.abs(dy1_y)) + 1e-10),
        0.01/(torch.min(torch.abs(dy1_z)) + 1e-10),
    ]

def euler_pde_6(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_y = dde.grad.jacobian(y1, x, j=1)
    dy1_z = dde.grad.jacobian(y1, x, j=2)
    dy1_t = dde.grad.jacobian(y1, x, j=3)
    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    dy2_y = dde.grad.jacobian(y2, x, j=1)
    dy2_z = dde.grad.jacobian(y2, x, j=2)
    dy2_t = dde.grad.jacobian(y2, x, j=3)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y+dy1_z*dy1_z) - (dy1_x*f1 + dy1_y*f2 + dy1_z*f3) - epsilon*(dy1_xx+dy1_yy+dy1_zz)

    dpsi_x = dde.grad.jacobian(psi, x, j=0)
    dpsi_y = dde.grad.jacobian(psi, x, j=1)
    dpsi_z = dde.grad.jacobian(psi, x, j=2)

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y+d_f3dy1_y2_z)+epsilon*(dy2_xx+dy2_yy+dy2_zz),
    ]

def euler_pde_7(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_y = dde.grad.jacobian(y1, x, j=1)
    # dy1_z = dde.grad.jacobian(y1, x, j=2)
    # dy1_y = 0.0
    dy1_z = 0.0
    dy1_t = dde.grad.jacobian(y1, x, j=2)

    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    # dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)
    # dy1_yy = 0.0
    dy1_zz = 0.0

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    dy2_y = dde.grad.jacobian(y2, x, j=1)
    # dy2_z = dde.grad.jacobian(y2, x, j=2)
    # dy2_y = 0.0
    dy2_z = 0.0
    dy2_t = dde.grad.jacobian(y2, x, j=2)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    # dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)
    # dy2_yy = 0.0
    dy2_zz = 0.0

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=x[:, 1:2]*1.0*(j2-j3)/j1
    f2=x[:, 0:1]*1.0*(j3-j1)/j2
    # f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    # f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    # f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    # d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y) - (dy1_x*f1 + dy1_y*f2) - epsilon*(dy1_xx+dy1_yy)
    
    # print(torch.min(torch.abs(y1)))
    # print(torch.min(torch.abs(dy1_x)))

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y)+epsilon*(dy2_xx+dy2_yy),
        0.001/(torch.max(torch.abs(dy1_x)) + 1e-10), # [:N**2]
        0.001/(torch.max(torch.abs(dy1_y)) + 1e-10), # [:N**2]
    ]

def euler_pde_8(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_y = dde.grad.jacobian(y1, x, j=1)
    # dy1_z = dde.grad.jacobian(y1, x, j=2)
    # dy1_y = 0.0
    dy1_z = 0.0
    dy1_t = dde.grad.jacobian(y1, x, j=2)

    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    # dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)
    # dy1_yy = 0.0
    dy1_zz = 0.0

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    dy2_y = dde.grad.jacobian(y2, x, j=1)
    # dy2_z = dde.grad.jacobian(y2, x, j=2)
    # dy2_y = 0.0
    dy2_z = 0.0
    dy2_t = dde.grad.jacobian(y2, x, j=2)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    # dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)
    # dy2_yy = 0.0
    dy2_zz = 0.0

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=x[:, 1:2]*1.0*(j2-j3)/j1
    f2=x[:, 0:1]*1.0*(j3-j1)/j2
    # f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    # f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    # f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    # d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y) - (dy1_x*f1 + dy1_y*f2) - epsilon*(dy1_xx+dy1_yy)

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y)+epsilon*(dy2_xx+dy2_yy),
        0.01/(torch.sum(torch.abs(dy1_x)) + 1e-10),
        0.01/(torch.sum(torch.abs(dy1_y)) + 1e-10),
    ]

S3 = dde.Variable(1.0)
sa_a, sa_b, sa_c, sa_d, sa_f= 10., 2.1, 0.75, .0045, 0.0005
sa_K, sa_T=1.38066*10**-23, 293.
def self_assembly(x, y):
    """Self assembly system.
    dy1_t = 1/2*(y3^2)-dy1_x*D1-dy1_xx*D2
    dy2_t = -dD1y2_x +dD2y2_xx
    y3=dy1_x*dD1_y3+dy1_xx*dD2_y3
    All collocation-based residuals are defined here
    Including a penalty function for negative solutions
    """
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_t = dde.grad.jacobian(y1, x, j=1)
    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_xx = dde.grad.hessian(y1, x, j=0)

    D2=sa_d*torch.exp(-(x[:, 0:1]-sa_b-sa_c*y3)*(x[:, 0:1]-sa_b-sa_c*y3))+sa_f
    F=sa_a*sa_K*sa_T*(x[:, 0:1]-sa_b-sa_c*y3)*(x[:, 0:1]-sa_b-sa_c*y3)
    D1=-2*(x[:, 0:1]-sa_b-sa_c*y3)*((sa_d*torch.exp(-(x[:, 0:1]-sa_b-sa_c*y3)*(x[:, 0:1]-sa_b-sa_c*y3)))+sa_a*D2)
    dy2_t = dde.grad.jacobian(y2, x, j=1)
    dD1y2_x=dde.grad.jacobian(D1*y2, x, j=0)
    dD2y2_xx = dde.grad.hessian(D2*y2, x,  j=0)
    dD1_y3=dde.grad.jacobian(D1, y3)
    dD2_y3=dde.grad.jacobian(D2, y3)

    return [
        dy1_t-.5*(S3*y3*S3*y3)+D1*dy1_x+D2*dy1_xx,
        dy2_t+dD1y2_x-dD2y2_xx,
        S3*y3-dy1_x*dD1_y3-dy1_xx*dD2_y3,
#         neg_loss,
#         neg_loss_y2,
    ]

euler_pdes = {
    1 : euler_pde_1,
    2 : euler_pde_2,
    3 : euler_pde_3,
    4 : euler_pde_4,
    5 : euler_pde_5,
    6 : euler_pde_6,
    7 : euler_pde_7,
    8 : euler_pde_8,
    9 : self_assembly,
}

######################################

class EarlyStoppingFixed(dde.callbacks.EarlyStopping):
    def __init__(self, path, min_delta=0, patience=0, baseline=None, monitor="loss_train"):
        super(EarlyStoppingFixed, self).__init__(min_delta, patience, baseline, monitor)
        self.path = path

    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            # must meet baseline first
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.model.train_state.epoch
                self.model.stop_training = True
        else:
            self.wait = 0
                
    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))
        
        self.model.save(self.path, verbose=True)

    def get_monitor_value(self):
        if self.monitor == "train loss" or self.monitor == "loss_train":
            data = self.model.train_state.loss_train
        elif self.monitor == "test loss" or self.monitor == "loss_test":
            data = self.model.train_state.loss_test
        else:
            raise ValueError("The specified monitor function is incorrect.", self.monitor)

        result = max(data)
        if min(data) < 1e-50:
            print("likely a numerical error")
            # numerical error
            return 1.0

        return result

class ModelCheckpoint2(dde.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        verbose=0,
        save_better_only=False,
        period=1,
        extra_comment="",
        monitor="train loss",
    ):
        super().__init__(
            filepath,
            verbose,
            save_better_only,
            period,
            monitor)
        self.extra_comment = extra_comment

    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current, self.best) and current < 1e-1:
            save_path = self.model.save(self.filepath, verbose=0)
            print(
                "Epoch {}: {} improved from {:.2e} to {:.2e}, saving model to {} ...\n".format(
                    self.model.train_state.epoch,
                    self.monitor,
                    self.best,
                    current,
                    save_path,
                ))

            comment = "Epoch {}: {} improved from {:.2e} to {:.2e}, {}".format(
                    self.model.train_state.epoch,
                    self.monitor,
                    self.best,
                    current,
                    self.extra_comment,
                )

            test_path = save_path.replace(".pt", "-%d.dat" % (
                self.model.train_state.epoch))
            test = np.hstack((
                self.model.train_state.X_test,
                self.model.train_state.y_pred_test))
            np.savetxt(test_path, test,
                header="x, y_pred",
                comments=comment,
            )
            print("saved test data to ", test_path)

            self.best = current

    def get_monitor_value(self):
        if self.monitor == "train loss" or self.monitor == "loss_train":
            data = self.model.train_state.loss_train
        elif self.monitor == "test loss" or self.monitor == "loss_test":
            data = self.model.train_state.loss_test
        else:
            raise ValueError("The specified monitor function is incorrect.", self.monitor)

        result = max(data)
        if min(data) < 1e-50:
            print("likely a numerical error")
            # numerical error
            return 1.0

        return result

class NonNeg_LastLayer_Model(dde.Model):
    def _train_sgd(self, iterations, display_every):
        print("NonNeg_LastLayer_Model training")
        for i in range(iterations):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            # clamped_weights = self.net.linears[-1].weight[1].clamp(0.0, 1.0)

            clamped_weights = self.net.linears[-1].weight[1].clamp_min(0.0)
            self.net.linears[-1].weight.data[1] = clamped_weights

            if self.stop_training:
                break

model_types = {
    0 : dde.Model,
    1 : NonNeg_LastLayer_Model
}

class CustomGeometryXTime1(dde.geometry.GeometryXTime):
    def __init__(self, geometry, timedomain, nt_override=None):
        self.geometry = geometry
        self.timedomain = timedomain
        self.dim = geometry.dim + timedomain.dim

        self.nt_override = nt_override

    def uniform_points(self, n, boundary=True):
        """Uniform points on the spatio-temporal domain.

        Geometry volume ~ bbox.
        Time volume ~ diam.
        """
        nx = int(
            np.ceil(
                (
                    n
                    * np.prod(self.geometry.bbox[1] - self.geometry.bbox[0])
                    / self.timedomain.diam
                )
                ** 0.5
            )
        )
        nt = int(np.ceil(n / nx))

        if self.nt_override is not None:
            print("overriding")
            nt = self.nt_override
            nx = int(np.ceil(n / nt))

        print(n, nx, nt)

        x = self.geometry.uniform_points(nx, boundary=boundary)
        nx = len(x)

        if boundary:
            t = self.timedomain.uniform_points(nt, boundary=True)
        else:
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype=config.real(np),
            )[:, None]
        xt = []
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti[0]))))
        xt = np.vstack(xt)
        if n != len(xt):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(xt))
            )
        return xt

class CustomGeometryXTime2(dde.geometry.GeometryXTime):
    def __init__(self, geometry, timedomain):
        self.geometry = geometry
        self.timedomain = timedomain
        self.dim = geometry.dim + timedomain.dim

    def uniform_points(self, n, boundary=True):
        """Uniform points on the spatio-temporal domain.

        Geometry volume ~ bbox.
        Time volume ~ diam.
        """
        nx = int(
            np.ceil(
                (
                    n
                    * np.prod(self.geometry.bbox[1] - self.geometry.bbox[0])
                    / self.timedomain.diam
                )
                ** 0.5
            )
        )

        # order here implies memory weight
        nx = initial_samples
        nt = int(np.ceil(n / nx))

        print(n, nx, nt)

        x = self.geometry.uniform_points(nx, boundary=boundary)
        nx = len(x)

        if boundary:
            t = self.timedomain.uniform_points(nt, boundary=True)
        else:
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype=config.real(np),
            )[:, None]
        xt = []
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti[0]))))
        xt = np.vstack(xt)
        if n != len(xt):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(xt))
            )
        return xt

class WASSPDE(dde.data.TimePDE):
    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if dde.backend.backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            outputs_pde = outputs
        elif dde.backend.backend_name == "jax":
            # JAX requires pure functions
            outputs_pde = (outputs, aux[0])

        f = []
        if self.pde is not None:
            if get_num_args(self.pde) == 2:
                f = self.pde(inputs, outputs_pde)
            elif get_num_args(self.pde) == 3:
                if self.auxiliary_var_fn is None:
                    if aux is None or len(aux) == 1:
                        raise ValueError("Auxiliary variable function not defined.")
                    f = self.pde(inputs, outputs_pde, unknowns=aux[1])
                else:
                    f = self.pde(inputs, outputs_pde, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * (len(f) + len(self.bcs))
        elif len(loss_fn) != len(f) + len(self.bcs):
            raise ValueError(
                "There are {} errors, but only {} losses.".format(
                    len(f) + len(self.bcs), len(loss_fn)
                )
            )

        bcs_start = np.cumsum([0] + self.num_bcs)
        bcs_start = list(map(int, bcs_start))

        # import ipdb; ipdb.set_trace()

        losses = []
        for i, fi in enumerate(f):
            if len(fi.shape) > 0:
                error = fi[bcs_start[-1] :]
                losses.append(loss_fn[i](bkd.zeros_like(error), error))
            else:
                losses.append(fi)

        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.

            fun = loss_fn[len(f) + i]

            # import ipdb; ipdb.set_trace();

            if "WASS_batch" in fun.__name__:
                # print("hello!")
                y_pred = outputs[beg:end, bc.component : bc.component + 1]
                # y_true = bc.values[bc.batch_indices]
                losses.append(fun(
                    bc.batch_indices,
                    y_pred,
                ))
            elif "WASS" in fun.__name__:
                y_pred = outputs[beg:end, bc.component : bc.component + 1]
                y_true = bc.values
                losses.append(fun(
                    y_true,
                    y_pred,
                ))
            else:
                error = bc.error(self.train_x, inputs, outputs, beg, end)
                losses.append(fun(
                    bkd.zeros_like(error),
                    error))

        # print(len(losses))

        return losses

from deepxde.nn import activations, initializers
from deepxde import config
class ScaleLayer(torch.nn.Module):
   def __init__(self, init_value=1.0):
        super().__init__()
        t = torch.FloatTensor([init_value]).to(device).requires_grad_(True)
        # it is NECESSARY to declare the tensor WITH GRADIENT, BEFORE declaring it as a parameter
        self.scale = torch.nn.Parameter(t)

   def forward(self, input):
        print(self.scale)
        input[:, 1] *=self.scale
        return input

class ScaledFNN(dde.nn.NN):
    """Fully-connected neural network."""
    '''
    torch.nn.Module
    '''

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

        print(self.num_trainable_parameters())
        # self.rho_scale = ScaleLayer()
        t = torch.FloatTensor([1.0]).to(device).requires_grad_(True)
        # it is NECESSARY to declare the tensor WITH GRADIENT, BEFORE declaring it as a parameter
        self.rho_scale = torch.nn.Parameter(t)
        print(self.num_trainable_parameters())

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))

        x = self.linears[-1](x)

        # import ipdb; ipdb.set_trace()

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        # x = self.rho_scale(x)

        # print(self.rho_scale)
        x[:, 1] *= self.rho_scale
        # must be done 'in-place' or 'out-of-place' consistently above

        return x

class Counter(object):
    def __init__(self):
        self.count = 0

    def on_press_saveplot(self, event, png_name):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == 'x':
            # visible = xl.get_visible()
            # xl.set_visible(not visible)
            # fig.canvas.draw()

            fname = png_name.replace(".png", "_%d.png" % (
                self.count))

            plt.savefig(
                fname,
                dpi=500,
                bbox_inches='tight')
            print("saved figure", fname)

            self.count += 1

######################################

def WASS_0(y_true, y_pred, sinkhorn, rho_tensor, C):
    p2 = torch.abs(torch.sum(y_pred) - 1)

    y_pred = torch.where(y_pred < 0, 0, y_pred)
    dist, _, _ = sinkhorn(C, y_pred.reshape(-1), rho_tensor)
    # print("Sinkhorn distance: {:.3f}".format(dist.item()))

    return dist + p2 # + p1

def WASS_batch_0(y_true, y_pred, device, sinkhorn, rho, state):
    rhoT_temp_tensor = torch.from_numpy(
        rho[y_true],
    ).to(device).requires_grad_(False)

    C_temp_device = torch.from_numpy(
        cdist(state[y_true, :], state[y_true, :], 'sqeuclidean'))
    C_temp_device = C_temp_device.to(device).requires_grad_(False)

    p2 = torch.abs(torch.sum(y_pred) - 1)

    y_pred = torch.where(y_pred < 0, 0, y_pred)

    # import ipdb; ipdb.set_trace()

    dist, _, _ = sinkhorn(
        C_temp_device,
        y_pred.reshape(-1),
        rhoT_temp_tensor)

    return dist + p2 # + p1

######################################

# this one gets stuck in local minima
def WASS_1(y_true, y_pred, sinkhorn, rho_tensor, C):
    p1 = (y_pred<0).sum() # negative terms

    p2 = 1 / torch.var(y_pred)

    p3 = torch.abs(torch.sum(y_pred) - 1)

    y_pred = torch.where(y_pred < 0, 0, y_pred)
    dist, _, _ = sinkhorn(C, y_pred.reshape(-1), rho_tensor)
    # print("Sinkhorn distance: {:.3f}".format(dist.item()))

    return 10 * p1 + p2 + p3 + dist

def WASS_2(y_true, y_pred, sinkhorn, rho_tensor, C):
    p1 = -torch.sum(y_pred[y_pred < 0])
    y_pred = torch.where(y_pred < 0, 0, y_pred)

    # p2 = torch.sum(y_pred[y_pred > 1.0])
    # p2 = torch.sum(y_pred) / 1.0
    p2 = (torch.sum(y_pred)-1)**2

    # normalizing can introduce nans
    # y_pred = y_pred / torch.sum(y_pred)

    dist, _, _ = sinkhorn(C, y_pred.reshape(-1), rho_tensor)
    # print("Sinkhorn distance: {:.3f}".format(dist.item()))

    return 10 * p1 + p2 + dist

def WASS_3(y_true, y_pred, sinkhorn, rho_tensor, C):
    p1 = -torch.sum(y_pred[y_pred < 0])
    y_pred = torch.where(y_pred < 0, 0, y_pred)

    # p2 = torch.sum(y_pred[y_pred > 1.0])
    # p2 = torch.sum(y_pred) / 1.0
    # p2 = (torch.sum(y_pred)-1)**2

    # normalizing can introduce nans
    # y_pred = y_pred / torch.sum(y_pred)

    dist, _, _ = sinkhorn(C, y_pred.reshape(-1), rho_tensor)
    # print("Sinkhorn distance: {:.3f}".format(dist.item()))

    return 10 * p1 + dist

def WASS_4(y_true, y_pred, sinkhorn, rho_tensor, C, n, dx):
    p1 = -torch.sum(y_pred[y_pred < 0])
    y_pred = torch.where(y_pred < 0, 0, y_pred)
    
    y_pred_matrix = y_pred.reshape(n, n)

    buffer1 = torch.zeros(n, dtype=dt)
    for i in range(n):
        buffer1[i] = torch.trapz(
            slice2d(y_pred_matrix, i, 0),
            dx=dx)
    t = torch.trapz(buffer1, dx=dx)
    p2 = (t - 1)**2
    # print("t", t)
    
    dist, _, _ = sinkhorn(C, y_pred.reshape(-1), rho_tensor)

    return 10*p1 + p2 + dist


def WASS_batch_1(y_true, y_pred, device, sinkhorn, rho, state):
    # import ipdb; ipdb.set_trace()
    p1 = -torch.sum(y_pred[y_pred < 0])

    # p1 = (y_pred<0).sum() # negative terms

    y_pred = torch.where(y_pred < 0, 0, y_pred)

    p2 = torch.abs(torch.sum(y_pred) - 1)

    rhoT_temp_tensor = torch.from_numpy(
        rho[y_true],
    ).to(device).requires_grad_(False)

    C_temp_device = torch.from_numpy(
        cdist(state[y_true, :], state[y_true, :], 'sqeuclidean'))
    C_temp_device = C_temp_device.to(device).requires_grad_(False)

    # import ipdb; ipdb.set_trace()

    dist, _, _ = sinkhorn(
        C_temp_device,
        y_pred.reshape(-1),
        rhoT_temp_tensor)

    return 20 * p1 + 10 * p2 + dist

def WASS_batch_2(y_true, y_pred, device, sinkhorn, rho, state):
    # import ipdb; ipdb.set_trace()
    p1 = -torch.sum(y_pred[y_pred < 0])

    # p1 = (y_pred<0).sum() # negative terms

    y_pred = torch.where(y_pred < 0, 0, y_pred)

    p2 = torch.abs(torch.sum(y_pred) - 500 / 3375)

    rhoT_temp_tensor = torch.from_numpy(
        rho[y_true],
    ).to(device).requires_grad_(False)

    C_temp_device = torch.from_numpy(
        cdist(state[y_true, :], state[y_true, :], 'sqeuclidean'))
    C_temp_device = C_temp_device.to(device).requires_grad_(False)

    # import ipdb; ipdb.set_trace()

    dist, _, _ = sinkhorn(
        C_temp_device,
        y_pred.reshape(-1),
        rhoT_temp_tensor)

    return 20 * p1 + dist

def WASS_batch_3(y_true, y_pred, device, sinkhorn, rho, state, expected_sum):
    # import ipdb; ipdb.set_trace()
    p1 = -torch.sum(y_pred[y_pred < 0])

    # p1 = (y_pred<0).sum() # negative terms

    y_pred = torch.where(y_pred < 0, 0, y_pred)

    # p2 = torch.abs(torch.sum(y_pred) - expected_sum) # l1-norm
    # p2 = torch.norm(torch.sum(y_pred) - expected_sum)

    # non-negative output must sum to expected_sum
    # so the greatest element in output must be 
    # LESS
    p2 = torch.sum(y_pred[y_pred > expected_sum])

    rhoT_temp_tensor = torch.from_numpy(
        rho[y_true],
    ).to(device).requires_grad_(False)

    C_temp_device = torch.from_numpy(
        cdist(state[y_true, :], state[y_true, :], 'sqeuclidean'))
    C_temp_device = C_temp_device.to(device).requires_grad_(False)

    # import ipdb; ipdb.set_trace()

    dist, _, _ = sinkhorn(
        C_temp_device,
        y_pred.reshape(-1),
        rhoT_temp_tensor)

    return 20 * p1 + p2 + dist

######################################

X1_index = 0
X2_index = 1
X3_index = 2

def euler_maru(
    y0,
    t_span,
    mu_func,
    dt,
    dw_func,
    sigma_func,
    args=()
    ):
    '''
    https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method

    # #recall
    '''
    ts = np.arange(t_span[0], t_span[1] + dt, dt)
    ys = np.zeros((len(ts), len(y0)))

    ys[0, :] = y0

    for i in range(1, ts.size):
        t = t_span[0] + (i - 1) * dt
        y = ys[i - 1, :]
        ys[i, :] = y + np.array(mu_func(t, y, *args)) * dt + sigma_func(y, t) * dw_func(dt)

    return ts, ys

def dynamics(t, state, j1, j2, j3, control_data, affine):
    statedot = np.zeros_like(state)
    # implicit is that all state dimension NOT set
    # have 0 dynamics == do not change in value

    alpha1 = (j2 - j3) / j1
    alpha2 = (j3 - j1) / j2
    alpha3 = (j1 - j2) / j3

    ########################################

    if len(state) == 3:
        statedot[X1_index] = alpha1 * state[X2_index] * state[X3_index]
        statedot[X2_index] = alpha2 * state[X3_index] * state[X1_index]
        statedot[X3_index] = alpha3 * state[X1_index] * state[X2_index]
    elif len(state) == 2:
        statedot[X1_index] = alpha1 * state[X2_index]
        statedot[X2_index] = alpha2 * state[X1_index]

    ########################################

    if control_data is None:
        return statedot
    # else:
    #     print("t", t)
    #     statedot[X1_index] += np.random.uniform(-0.2, 0.5)
    #     statedot[X2_index] += np.random.uniform(-0.2, 0.5)
    #     statedot[X3_index] += np.random.uniform(-0.2, 0.5)
    #     return statedot

    if np.abs(t - T_0) < 1e-8:
        t_key = 't0'
    elif np.abs(t - T_t) < 1e-8:
        t_key = 'tT'
    else:
        t_key = 'tt'

    t_control_data = control_data[t_key]

    query = state
    # if t_key == 'tt':
    if t_control_data['grid'].shape[1] == len(state) + 1:
        query = np.append(query, t)

    # if np.abs(t - T_0) < 1e-8:
    #     print("t_key", t_key)
    #     print("state", query)

    # grid_l2_norms = np.linalg.norm(query - t_control_data['grid'], ord=2, axis=1)
    # closest_grid_idx = grid_l2_norms.argmin()

    closest_grid_idx = np.linalg.norm(query - t_control_data['grid'], ord=1, axis=1).argmin()
    # print("query",
    #     query,
    #     closest_grid_idx,
    #     t_control_data['grid'][closest_grid_idx],
    #     t_control_data['0'][closest_grid_idx],
    #     t_control_data['1'][closest_grid_idx])

    v_x = t_control_data['0'][closest_grid_idx]
    v_y = t_control_data['1'][closest_grid_idx]

    if affine is not None:
        v_x = affine(v_x)
        v_y = affine(v_y)

    # print("v_x", v_x)
    # print("v_y", v_y)

    statedot[X1_index] = statedot[X1_index] + v_x
    statedot[X2_index] = statedot[X2_index] + v_y

    if '2' in t_control_data:
        v_z = t_control_data['2'][closest_grid_idx]
        if affine is not None:
            v_z = affine(v_z)
        statedot[X3_index] = statedot[X3_index] + v_z

    return statedot

from concurrent.futures import ThreadPoolExecutor

def hash_func(v_scale, bias):
    return "%.3f_%.3f" % (v_scale, bias)

class Integrator(object):
    def __init__(self, initial_sample, t_span, args, dynamics):
        self.initial_sample = initial_sample
        self.t_span = t_span
        self.args = args
        self.dynamics = dynamics
        self.j1, self.j2, self.j3 = [float(x) for x in args.system.split(",")]

    def task(self, i, target, control_data, affine):
        # print("starting {}".format(i))

        if self.args.noise:
            _, tmp = euler_maru(
                self.initial_sample[i, :],
                self.t_span,
                self.dynamics,
                (self.t_span[-1] - self.t_span[0])/(self.args.integrate_N),
                lambda delta_t: np.random.normal(
                    loc=0.0,
                    scale=np.sqrt(delta_t)),
                lambda y, t: 0.06,
                (
                    self.j1, self.j2, self.j3,
                    control_data,
                    affine
                ))
        else:
            _, tmp = euler_maru(
                self.initial_sample[i, :],
                self.t_span,
                self.dynamics,
                (self.t_span[-1] - self.t_span[0])/(self.args.integrate_N),
                lambda delta_t: 0.0,
                # np.random.normal(loc=0.0, scale=np.sqrt(delta_t)),
                lambda y, t: 0.0,
                # 0.06,
                (
                    self.j1, self.j2, self.j3,
                    control_data,
                    affine
                ))
        target[i, :, :] = tmp.T
        return i
