#!/usr/bin/env python3

from scipy.stats import truncnorm, norm
import numpy as np

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

def get_pmf_stats(pmf, x1, x2, x3, x_T, y_T, z_T):
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