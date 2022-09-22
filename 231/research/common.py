#!/usr/bin/env python3

from scipy.stats import truncnorm, norm
import numpy as np, os, time, sys

import torch

def pdf3d(x,y,z,rv):
    return rv.pdf(np.hstack((x, y, z)))

def slice(matrix_3d, i, j, mode):
    if mode == 0:
        return matrix_3d[j, i, :]
    elif mode == 1:
        return matrix_3d[i, j, :]
    else:
        return matrix_3d[i, :, j]

def get_pdf_support_torch(
    tensor,
    xtensors,
    mode):
    n = len(xtensors[0])
    tensor_3d = torch.reshape(
        tensor, (n, n, n))

    buffer0 = torch.zeros(len(xtensors[1]), dtype=dt)
    buffer1 = torch.zeros(len(xtensors[0]), dtype=dt)
    for j in range(len(xtensors[0])):
        # collapse 1 dimension away into buffer0
        for i in range(len(xtensors[1])):
            buffer0[i] = torch.trapz(
                slice(tensor_3d, i, j, mode)
                , x=xtensors[2])
        # collapse 2 dimensions into 1 scalar
        buffer1[j] = torch.trapz(
            buffer0,
            x=xtensors[1])
    return torch.trapz(buffer1, x=xtensors[0])

def get_marginal_pmf(matrix_3d, xs, mode):
    marginal = np.array([
      np.sum(
          np.array([
            np.sum(
                slice(matrix_3d, i, j, mode)
            )
            for i in range(len(xs[1]))])
        ) # x2 slice for one x1 => R
    for j in range(len(xs[0]))])
    return marginal

def get_pmf_stats(pmf, x_T, y_T, z_T, x1, x2, x3):
    # pmf has SUM=1, so normalize as such
    pmf_support = np.sum(pmf)

    pmf_normed = pmf / pmf_support

    n = len(x1)
    pmf_cube_normed = pmf_normed.reshape(n, n, n)

    # finding mu / E(x1) of discrete distribution / pmf
    # is implemented as a dot product
    x1_marginal_pmf = get_marginal_pmf(
        pmf_cube_normed, [x1, x2, x3], 0)
    mu1 = np.dot(x1_marginal_pmf, x1)

    x2_marginal_pmf = get_marginal_pmf(
        pmf_cube_normed, [x2, x3, x1], 1)
    mu2 = np.dot(x2_marginal_pmf, x2)

    x3_marginal_pmf = get_marginal_pmf(
        pmf_cube_normed, [x3, x1, x2], 2)
    mu3 = np.dot(x3_marginal_pmf, x3)

    mu = np.array([mu1, mu2, mu3])

    dx = x_T - mu1
    dy = y_T - mu2
    dz = z_T - mu3

    cov_xx = np.sum(pmf_normed * dx * dx)
    cov_xy = np.sum(pmf_normed * dx * dy)
    cov_xz = np.sum(pmf_normed * dx * dz)
    cov_yy = np.sum(pmf_normed * dy * dy)
    cov_yz = np.sum(pmf_normed * dy * dz)
    cov_zz = np.sum(pmf_normed * dz * dz)

    cov_matrix = np.array([
        [cov_xx, cov_xy, cov_xz],
        [cov_xy, cov_yy, cov_yz],
        [cov_xz, cov_yz, cov_zz]
    ])

    return mu, cov_matrix, pmf_cube_normed

def fill_marginal_pmf_torch(
    tensor_3d,
    xtensors,
    mode,
    buffer0,
    buffer1):
    for j in range(len(xtensors[0])):
        # collapse 1 dimension away into buffer0
        for i in range(len(xtensors[1])):
            buffer0[i] = torch.sum(slice(tensor_3d, i, j, mode))
        # collapse 2 dimensions into 1 scalar
        buffer1[j] = torch.sum(buffer0)

def get_pmf_stats_torch(
    pmf,
    x_T, y_T, z_T,
    x1, x2, x3,
    dt):
    pmf_support = torch.sum(pmf)

    pmf_normed = pmf / pmf_support

    n = len(x1)
    pmf_cube_normed = torch.reshape(
        pmf_normed, (n, n, n))

    # finding mu / E(x1) of discrete distribution / pmf
    # is implemented as a dot product
    buffer0 = torch.zeros(len(x2), dtype=dt)
    x1_marginal_pmf = torch.zeros(len(x1), dtype=dt)
    fill_marginal_pmf_torch(pmf_cube_normed,
        [x1, x2, x3], 0,
        buffer0,
        x1_marginal_pmf)
    mu1 = torch.dot(x1_marginal_pmf, x1)

    buffer1 = torch.zeros(len(x3), dtype=dt)
    x2_marginal_pmf = torch.zeros(len(x2), dtype=dt)
    fill_marginal_pmf_torch(pmf_cube_normed,
        [x2, x3, x1], 1,
        buffer1,
        x2_marginal_pmf)
    mu2 = torch.dot(x2_marginal_pmf, x2)

    buffer2 = torch.zeros(len(x1), dtype=dt)
    x3_marginal_pmf = torch.zeros(len(x3), dtype=dt)
    fill_marginal_pmf_torch(pmf_cube_normed,
        [x3, x1, x2], 2,
        buffer2,
        x3_marginal_pmf)
    mu3 = torch.dot(x3_marginal_pmf, x3)

    mu = torch.cat((
        mu1.reshape(1),
        mu2.reshape(1),
        mu3.reshape(1)
    ))

    dx = x_T - mu1
    dy = y_T - mu2
    dz = z_T - mu3

    cov_xx = torch.sum(pmf_normed * dx * dx).reshape(1)
    cov_xy = torch.sum(pmf_normed * dx * dy).reshape(1)
    cov_xz = torch.sum(pmf_normed * dx * dz).reshape(1)
    cov_yy = torch.sum(pmf_normed * dy * dy).reshape(1)
    cov_yz = torch.sum(pmf_normed * dy * dz).reshape(1)
    cov_zz = torch.sum(pmf_normed * dz * dz).reshape(1)

    cov_matrix = torch.stack((
        torch.cat((cov_xx, cov_xy, cov_xz)),
        torch.cat((cov_xy, cov_yy, cov_yz)),
        torch.cat((cov_xz, cov_yz, cov_zz))
    ))

    return mu, cov_matrix

def get_multivariate_truncated_norm(x_T, y_T, z_T, mu, sigma, state_min, state_max, N, f, cache_name):
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

N = 50

# must be floats
state_min = -5.0
state_max = 5.0

mu_0 = 2.0
sigma_0 = 1.0

mu_T = 0.0
sigma_T = 1.5

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
T_IDX = 1
Y3_IDX = 2
RHO_OPT_IDX = 3

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