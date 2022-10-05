#!/usr/bin/env python3

from scipy.stats import truncnorm, norm
import numpy as np

def pdf1d(x, mu, sigma):
    a, b = (state_min - mu) / sigma, (state_max - mu) / sigma
    rho_x=truncnorm.pdf(x, a, b,
        loc = mu, scale = sigma)

    # do NOT use gaussian norm, because it is only area=1
    # from -inf, inf, will not be for finite state/grid
    # rho_x = norm.pdf(x, mu, sigma)
    return rho_x

N = 50

# must be floats
state_min = 0.0
state_max = 6.0

mu_0 = 2.0
sigma_0 = 1.5

mu_T = 4.0
sigma_T = 1.0

j1, j2, j3 =1,1,2 # axis-symmetric case
q_statepenalty_gain = 0 # 0.5

T_0=0. #initial time
T_t=200. #Terminal time

id_prefix = "wass_1d"
de = 1

########################################################

X_IDX = 0
T_IDX = 1
Y3_IDX = 2
RHO_OPT_IDX = 3

def plot_rho_bc(label, test_ti, mu, sigma, ax):
    ind = np.lexsort((test_ti[:,X_IDX], test_ti[:,T_IDX]))
    test_ti = test_ti[ind]

    p1 = np.where(test_ti[:,RHO_OPT_IDX] < 0)[0]
    print("# neg:", len(p1))

    pdf = np.trapz(test_ti[:,RHO_OPT_IDX],
        axis=0,
        x=test_ti[:,X_IDX])
    print("pdf: ", pdf)

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