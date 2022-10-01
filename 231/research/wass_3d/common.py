#!/usr/bin/env python3

from scipy.stats import truncnorm, norm
import numpy as np, os, time, sys

import torch

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

def sinkhorn_torch(K, c_tensor, a_tensor, b_tensor,
    device,
    delta=1e-1,
    lam=1e-5):    
    u_vec = torch.ones(a_tensor.shape[0], dtype=torch.float32).to(device)
    v_vec = torch.ones(b_tensor.shape[0], dtype=torch.float32).to(device)

    import ipdb; ipdb.set_trace()

    u_trans = torch.matmul(K, v_vec) + lam  # add regularization to avoid divide 0
    v_trans = torch.matmul(K.T, u_vec) + lam  # add regularization to avoid divide 0


    err_1 = torch.sum(torch.abs(u_vec * u_trans - a_tensor))
    err_2 = torch.sum(torch.abs(v_vec * v_trans - b_tensor))

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
        else:
            # print("DONE!")
            break

    p_opt = torch.linalg.multi_dot([
        torch.diag(v_vec),
        K,
        torch.diag(u_vec)])

    return torch.dot(c_tensor, p_opt.view(-1))

N = 20

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