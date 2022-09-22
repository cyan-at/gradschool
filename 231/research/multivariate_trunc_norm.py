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

import julia
from julia.api import Julia

import numpy as np
import scipy
from scipy.stats import multivariate_normal

from common import *

def slice(matrix_3d, i, j, mode):
    if mode == 0:
        return matrix_3d[j, i, :]
    elif mode == 1:
        return matrix_3d[i, j, :]
    else:
        return matrix_3d[i, :, j]

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

x1 = np.transpose(np.linspace(state_min, state_max, N))
x2 = np.transpose(np.linspace(state_min, state_max, N))
x3 = np.transpose(np.linspace(state_min, state_max, N))
[X,Y,Z] = np.meshgrid(x1,x2,x3)
x_T=X.reshape(N**3,1)
y_T=Y.reshape(N**3,1)
z_T=Z.reshape(N**3,1)
state = np.hstack((x_T, y_T, z_T))

#################################################

trunc_rho0 = np.array([
    trunc_pdf(trunc_rv0, state[i, :]) for i in range(state.shape[0])
    ])
np.savetxt('rho0_%.3f_%.3f__%.3f_%.3f__%d' % (
    mu_0, sigma_0,
    state_min, state_max,
    N), trunc_rho0)

#################################################

trunc_rhoT = np.array([
    trunc_pdf(trunc_rvT, state[i, :]) for i in range(state.shape[0])
    ])
np.savetxt('rhoT_%.3f_%.3f__%.3f_%.3f__%d' % (
    mu_T, sigma_T,
    state_min, state_max,
    N), trunc_rhoT)

#################################################

# trunc_rho0_cube = trunc_rho0.reshape(N, N, N)

# as a trunnorm, trun_rho0 already has a pdf support of 1
# # deserializing the trunc_rho0 as a PMF

# pmf_support = np.sum(trunc_rho_0_cube)
# trunc_rho0_cube_normed = trunc_rho_0_cube / pmf_support

# # finding mu / E(x1) of discrete distribution / pmf
# # is implemented as a dot product
# x1_marginal_pmf = get_marginal_pmf(
#     trunc_rho0_cube_normed, [x1, x2, x3], 0)
# mu1 = np.dot(x1_marginal_pmf, x1)

# x2_marginal_pmf = get_marginal_pmf(
#     trunc_rho0_cube_normed, [x2, x3, x1], 1)
# mu2 = np.dot(x2_marginal_pmf, x2)

# x3_marginal_pmf = get_marginal_pmf(
#     trunc_rho0_cube_normed, [x3, x1, x2], 2)
# mu3 = np.dot(x3_marginal_pmf, x3)

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


# trunc_support = get_pdf_support(trunc_rho_0_cube, [x1, x2, x3], 0)
# # rho_0 /= support
# # rho_0_cube /= support
# print("trunc_support", trunc_support)

# trunc_rho0_x1_marginal = get_marginal(
#     trunc_rho_0_cube, [x1, x2, x3], 0)

# # trunc_rho0_x1_marginal /= np.sum(trunc_rho0_x1_marginal)

# trunc_mu_1 = np.trapz(trunc_rho0_x1_marginal * x1, x=x1)
# print("trunc_mu_1", trunc_mu_1)

# print(trunc_mean(trunc_rv0))




import ipdb; ipdb.set_trace();



