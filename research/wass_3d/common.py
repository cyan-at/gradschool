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
        buffer0 = torch.zeros(len(xtensors[0]), dtype=dt)
        buffer1 = torch.zeros(len(xtensors[1]), dtype=dt)
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
                for i in range(len(xs[(mode + 1) % d]))])
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
        marginal = torch.cat(
            torch.sum(
                torch.cat(
                    torch.sum(
                        slice(tensor_matrix, i, j, mode)
                    ) # smoosh out axis=mode
                    for i in range(len(xs[(mode + 1) % d]))
                )
            ) # x2 slice for one x1 => R
            for j in range(len(xs[(mode + 2) % d]))
        )
        return marginal
    else:
        # mode == 0, flatten 'x' => y marginal
        # mode == 1, flatten 'y' => x marginal
        marginal = torch.cat(
            torch.sum(
                slice2d(matrix, i, mode)
            )
            for i in range(len(xs[mode]))
        )
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
        mus[i] = np.dot(marginal_pmf, linspaces[i])

        marginals.append(marginal_pmf)
        deltas.append(mesh_vectors[i] - mus[i])

    cov_matrix = torch.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov_matrix[i][j] = np.sum(pmf_normed * deltas[i] * deltas[j])

    return mu, cov_matrix

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

j1, j2, j3 =1,1,2 # axis-symmetric case
q_statepenalty_gain = 0 # 0.5

T_0=0. #initial time
T_t=20. #Terminal time

id_prefix = "wass_3d"
de = 1

f = 'float32'
dt = torch.float32

samples_between_initial_and_final = 12000 # 10^4 order, 20k = out of memory
initial_and_final_samples = 2000 # some 10^3 order

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

euler_pdes = {
    1 : euler_pde_1,
    2 : euler_pde_2,
    3 : euler_pde_3
}

######################################

ck_path = "%s/%s_model" % (os.path.abspath("./"), id_prefix)
class EarlyStoppingFixed(dde.callbacks.EarlyStopping):
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
        
        self.model.save(ck_path, verbose=True)

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
earlystop_cb = EarlyStoppingFixed(baseline=1e-3, patience=0)

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
modelcheckpt_cb = ModelCheckpoint2(
    ck_path, verbose=True, save_better_only=True, period=1)

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
        error_f = [fi[bcs_start[-1] :] for fi in f]
        losses = [
            loss_fn[i](bkd.zeros_like(error), error) for i, error in enumerate(error_f)
        ]
        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.

            fun = loss_fn[len(error_f) + i]
            if "WASS_batch" in fun.__name__:
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
        return losses