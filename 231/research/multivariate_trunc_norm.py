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

rho0_name = 'rho0_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_0, sigma_0,
    state_min, state_max,
    N)
rhoT_name = 'rhoT_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_T, sigma_T,
    state_min, state_max,
    N)

#################################################

x1 = np.transpose(np.linspace(state_min, state_max, N))
x2 = np.transpose(np.linspace(state_min, state_max, N))
x3 = np.transpose(np.linspace(state_min, state_max, N))
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
        ]).reshape(N**3, 1)
    np.savetxt(rho0_name, trunc_rho0)

    #################################################

    trunc_rhoT = np.array([
        trunc_pdf(trunc_rvT, state[i, :]) for i in range(state.shape[0])
        ])
    np.savetxt(rhoT_name, trunc_rhoT)
else:
    trunc_rho0 = np.loadtxt(rho0_name).reshape(N**3, 1)
    trunc_rhoT = np.loadtxt(rhoT_name).reshape(N**3, 1)

#################################################

# as a trunnorm, trun_rho0 already has a pdf support of 1
rho0_mu, rho0_cov_matrix, _ = get_pmf_stats(
    trunc_rho0, x1, x2, x3, x_T, y_T, z_T)

print("rho0_mu: ", rho0_mu)
print("rho0_cov_matrix:")
print(rho0_cov_matrix)

#################################################

# as a trunnorm, trun_rho0 already has a pdf support of 1
rhoT_mu, rhoT_cov_matrix, _ = get_pmf_stats(
    trunc_rhoT, x1, x2, x3, x_T, y_T, z_T)

print("rho0_mu: ", rhoT_mu)
print("rho0_cov_matrix:")
print(rhoT_cov_matrix)

#################################################

# trunc_rhoT_cube = trunc_rhoT.reshape(N, N, N)

'''
rv0 = multivariate_normal([mu_0, mu_0, mu_0], sigma_0 * np.eye(3))
rho_0=pdf3d(x_T,y_T,z_T,rv0).reshape(len(x_T),1)
rho_0_cube = rho_0.reshape(N, N, N)
support = get_pdf_support(rho_0_cube, [x1, x2, x3], 0)
print("support", support)

rho0_x1_marginal = get_marginal(
    rho_0_cube, [x1, x2, x3], 0, False)
mu_1 = np.trapz(rho0_x1_marginal * x1, x=x1)
print("mu_1", mu_1)
'''

# trunc_rho_T = np.array([trunc_pdf(trunc_rvT, state[i, :]) for i in range(state.shape[0])])


# trunc_support = get_pdf_support(trunc_rho0_cube, [x1, x2, x3], 0)
# # rho_0 /= support
# # rho_0_cube /= support
# print("trunc_support", trunc_support)

# trunc_rho0_x1_marginal = get_marginal(
#     trunc_rho0_cube, [x1, x2, x3], 0)

# # trunc_rho0_x1_marginal /= np.sum(trunc_rho0_x1_marginal)

# trunc_mu_1 = np.trapz(trunc_rho0_x1_marginal * x1, x=x1)
# print("trunc_mu_1", trunc_mu_1)

# print(trunc_mean(trunc_rv0))

import math
import torch
import torch.linalg as linalg

fact = 0.5 # n = 3

m1 = torch.from_numpy(
    rho0_mu
).requires_grad_(True)
sig1 = torch.from_numpy(
    rho0_cov_matrix
).requires_grad_(True)
m2 = torch.from_numpy(
    rhoT_mu
).requires_grad_(True)
sig2 = torch.from_numpy(
    rhoT_cov_matrix
).requires_grad_(True)
w = torch.norm(m1 - m2, p=2) + torch.trace(sig1 + sig2 - 2*torch.sqrt(torch.sqrt(sig1) * sig2 * torch.sqrt(sig1)))
print("true Wasserstein  :", w)
w.backward()
print("m1.grad", m1.grad)



