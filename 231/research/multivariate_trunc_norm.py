#!/usr/bin/env python3

# https://stackoverflow.com/questions/61707037/are-expected-mean-var-of-sample-equal-to-population-mean-var
# the mu / cov from discrete SAMPLES will not be the population
# so you should derive wasserstein off of discrete SAMPLE statistics, and not population ones
# and even if you increase N it does not improve the accuracy of mu

# truncnorm vs norm impacts the INTEGRAL, the area under the PDF
# truncnorm gives you a pdf where you don't need to
# normalize by integral because it is already ~1
# that's all there is to this

# question: does normalizing a discrete distribution by the sum so the sum == 1
# does it change the mu estimate?

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

import scipy
from scipy.stats import multivariate_normal

import os
from common import *

import torch
import torch.linalg as linalg

torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)


rho0_name = 'rho0_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_0, sigma_0,
    state_min, state_max,
    N)
rhoT_name = 'rhoT_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_T, sigma_T,
    state_min, state_max,
    N)

#################################################

f = 'float32'
dt = torch.float32

x1 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
x2 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
x3 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
[X,Y,Z] = np.meshgrid(x1,x2,x3)
x_T=X.reshape(N**3,1)
y_T=Y.reshape(N**3,1)
z_T=Z.reshape(N**3,1)
state = np.hstack((x_T, y_T, z_T))

if not os.path.exists(rho0_name) or not os.path.exists(rhoT_name):
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

    trunc_rv0 = make_d(mu_0, sigma_0, state_min, state_max)
    trunc_rvT = make_d(mu_T, sigma_T, state_min, state_max)

    #################################################

    trunc_rho0 = np.array([
        trunc_pdf(trunc_rv0, state[i, :]) for i in range(state.shape[0])
        ]).astype(f).reshape(N**3, 1)
    np.savetxt(rho0_name, trunc_rho0)

    #################################################

    trunc_rhoT = np.array([
        trunc_pdf(trunc_rvT, state[i, :]) for i in range(state.shape[0])
        ]).astype(f).reshape(N**3, 1)
    np.savetxt(rhoT_name, trunc_rhoT)
else:
    trunc_rho0 = np.loadtxt(rho0_name).astype(f).reshape(N**3, 1)
    trunc_rhoT = np.loadtxt(rhoT_name).astype(f).reshape(N**3, 1)

'''
#################################################

# as a trunnorm, trun_rho0 already has a pdf support of 1
rho0_mu, rho0_cov_matrix, _ = get_pmf_stats(
    trunc_rho0, x_T, y_T, z_T, x1, x2, x3)

print("rho0_mu: ", rho0_mu)
print("rho0_cov_matrix:")
print(rho0_cov_matrix)

#################################################

# as a trunnorm, trun_rho0 already has a pdf support of 1
rhoT_mu, rhoT_cov_matrix, _ = get_pmf_stats(
    trunc_rhoT, x_T, y_T, z_T, x1, x2, x3)

print("rho0_mu: ", rhoT_mu)
print("rho0_cov_matrix:")
print(rhoT_cov_matrix)

#################################################

m1 = torch.from_numpy(
    rho0_mu
).requires_grad_(True)
sig1 = torch.from_numpy(
    rho0_cov_matrix
).requires_grad_(False)
m2 = torch.from_numpy(
    rhoT_mu
).requires_grad_(False)
sig2 = torch.from_numpy(
    rhoT_cov_matrix
).requires_grad_(False)
w = torch.norm(m1 - m2, p=2) + torch.trace(sig1 + sig2 - 2*torch.sqrt(torch.sqrt(sig1) * sig2 * torch.sqrt(sig1)))
print("true Wasserstein  :", w)
w.backward()
print("m1.grad", m1.grad)

#################################################
'''

x1_tensor = torch.from_numpy(
    x1).requires_grad_(False)
x2_tensor = torch.from_numpy(
    x2).requires_grad_(False)
x3_tensor = torch.from_numpy(
    x3).requires_grad_(False)

x_T_tensor = torch.from_numpy(
    x_T).requires_grad_(False)
y_T_tensor = torch.from_numpy(
    y_T).requires_grad_(False)
z_T_tensor = torch.from_numpy(
    z_T).requires_grad_(False)

trunc_rho0_tensor = torch.from_numpy(
    trunc_rho0
).requires_grad_(True)

m1, sig1 = get_pmf_stats_torch(
    trunc_rho0_tensor,
    x_T_tensor, y_T_tensor, z_T_tensor,
    x1_tensor, x2_tensor, x3_tensor, dt)

print("m1\n", m1)
print("sig1\n", sig1)

trunc_rhoT_tensor = torch.from_numpy(
    trunc_rhoT
).requires_grad_(False)

m2, sig2 = get_pmf_stats_torch(
    trunc_rhoT_tensor,
    x_T_tensor, y_T_tensor, z_T_tensor,
    x1_tensor, x2_tensor, x3_tensor, dt)

print("m2\n", m2)
print("sig2\n", sig2)

e = torch.ones((3,3)) * 1e-9 # regularizer to prevent nan in matrix sqrt, nan in gradient
w1 = torch.sqrt(sig1 + e)
w = torch.norm(m1 - m2, p=2) + torch.trace(sig1 + sig2 - 2*torch.sqrt(w1 * sig2 * w1 + e))
print("true Wasserstein  :", w)

w.backward()
print("trunc_rho0_tensor.grad", trunc_rho0_tensor.grad)
# print("trunc_rhoT_tensor.grad", trunc_rho0_tensor.grad)

# import ipdb; ipdb.set_trace();
