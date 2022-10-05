#!/usr/bin/env python3

from common import *

from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm
from scipy.linalg import norm

from scipy import sparse
import cvxpy as cp

from scipy.stats import multivariate_normal

from scipy.optimize import linprog

import matplotlib.pyplot as plt 
from matplotlib import cm

torch.set_printoptions(precision=3, sci_mode=False)

def sinkhorn(K, cvector, a_vec, b_vec, 
    delta=1e-1, lam=1e-6):
    a_vec = a_vec.reshape(-1)
    b_vec = b_vec.reshape(-1)

    u_vec = np.ones(np.shape(a_vec)[0])
    v_vec = np.ones(np.shape(b_vec)[0])

    u_trans = np.dot(K, v_vec) + lam  # add regularization to avoid divide 0
    v_trans = np.dot(K.T, u_vec) + lam  # add regularization to avoid divide 0

    err_1 = np.sum(np.abs(u_vec * u_trans - a_vec))
    err_2 = np.sum(np.abs(v_vec * v_trans - b_vec))

    # print("u.shape", u_vec.shape)
    # print("v.shape", v_vec.shape)

    # print("u_trans", u_trans)
    # print("v_trans", v_trans)

    # import ipdb; ipdb.set_trace()

    while True:
        if err_1 + err_2 > delta:
            u_vec = np.divide(a_vec, u_trans)
            v_trans = np.dot(K.T, u_vec) + lam

            v_vec = np.divide(b_vec, v_trans)
            u_trans = np.dot(K, v_vec) + lam

            err_1 = np.sum(np.abs(u_vec * u_trans - a_vec))
            err_2 = np.sum(np.abs(v_vec * v_trans - b_vec))
            # print("err_1 + err_2", err_2 + err_1)

            # import ipdb; ipdb.set_trace()
        else:
            break

    # print(v_vec)

    p_opt = np.linalg.multi_dot([
        np.diag(v_vec),
        K,
        np.diag(u_vec)])

    return np.dot(cvector, p_opt.reshape(-1))

def stat_wass(a, b, mesh_vectors, linspaces):
    mu_a, cov_a, _, _ = get_pmf_stats(
        a, mesh_vectors, linspaces)

    mu_b, cov_b, _, _ = get_pmf_stats(
        b, mesh_vectors, linspaces)

    c = cov_a * cov_b * cov_a
    a = sqrtm(c)
    b = np.trace(cov_a + cov_b - 2*a)
    w = norm(mu_a - mu_b, 2)

    return w + b

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--N', type=int, default=10, help='')
parser.add_argument('--js', type=str, default="1,1,2", help='')
parser.add_argument('--q', type=float, default=0.0, help='')
args = parser.parse_args()

N = args.N
j1, j2, j3 = [float(x) for x in args.js.split(",")] # axis-symmetric case
q_statepenalty_gain = args.q # 0.5

# sigma_T = args.sigma_T

d = 2
M = N**d

linspaces = []
for i in range(d):
    linspaces.append(np.transpose(np.linspace(state_min, state_max, N)))

meshes = np.meshgrid(*linspaces)
mesh_vectors = []
for i in range(d):
    mesh_vectors.append(meshes[i].reshape(M,1))
state = np.hstack(tuple(mesh_vectors))

# rho0_name = 'rho0_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
#     mu_0, sigma_0,
#     state_min, state_max,
#     N)
# trunc_rho0_pdf = get_multivariate_truncated_pdf(x_T, y_T, z_T, mu_0, sigma_0, state_min, state_max, N, f, rho0_name)

# rhoT_name = 'rhoT_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
#     mu_T, sigma_T,
#     state_min, state_max,
#     N)
# trunc_rhoT_pdf = get_multivariate_truncated_pdf(x_T, y_T, z_T, mu_T, sigma_T, state_min, state_max, N, f, rhoT_name)
# rho0 = trunc_rho0_pdf
# rhoT = trunc_rhoT_pdf

rv0 = multivariate_normal([mu_0]*d, sigma_0 * np.eye(d))
rvT = multivariate_normal([mu_T]*d, sigma_T * np.eye(d))
rho0=rv0.pdf(state).reshape(state.shape[0],1)
rhoT=rvT.pdf(state).reshape(state.shape[0],1)

# rho0 /= get_pdf_support(
#     rho0.reshape(tuple([N]*d)),
#     linspaces)
# rhoT /= get_pdf_support(
#     rhoT.reshape(tuple([N]*d)),
#     linspaces)
# rho0 /= np.sum(rho0)
# rhoT /= np.sum(rhoT)

C = cdist(state, state, 'sqeuclidean')
cvector = C.reshape((M)**2)

bvector = np.concatenate((rho0, rhoT), axis=0)

########################################################
# statistical wass

wass1 = stat_wass(rho0, rhoT, mesh_vectors, linspaces)
print("wass1", wass1)

########################################################
# cvxpy wass

A = np.concatenate(
    (
        np.kron(
            np.ones((1,M)),
            sparse.eye(M).toarray()
        ),
        np.kron(
            sparse.eye(M).toarray(),
            np.ones((1,M))
        )
    ), axis=0)
x = cp.Variable(
    cvector.shape[0],
    nonneg=True
)
pred = cp.Parameter((A.shape[0],))
problem = cp.Problem(
    cp.Minimize(cvector.T @ x),
    [
        A @ x == pred,
    ],
)
assert problem.is_dpp()
pred.value = np.concatenate((rho0, rhoT), axis=0).reshape(-1)

problem.solve(verbose=True, solver=cp.ECOS)
wass2 = 0.0
if (x.value is not None):
    wass2 = np.dot(cvector, x.value)
    print("wass2", wass2)
# wass2_sqrt = np.sqrt(wass2)
# print("wass2_sqrt", wass2_sqrt)

########################################################
# linprog wass
# linprog has a numerical requirement: N needs to be sufficiently high to solve:
# for 2D, N >15, =20 is good

wass3 = None
try:
    wass3 = linprog(cvector,
        A_eq=A,
        b_eq=np.concatenate((rho0, rhoT), axis=0),
        bounds=[(0, None)],
        options={"disp": True})
    wass3 = wass3.fun
except:
    wass3 = 0.0
if wass3 is None:
    wass3 = 0.0
print("wass3", wass3)

########################################################
#sinkhorn wass

reg = 10e-2 # gamma, 10e-2, 5e-2
K = np.exp(-C / reg - 1)
wass4 = sinkhorn(
    K,
    cvector,
    rho0, rhoT, delta=1e-3, lam=1e-10)
print("wass4", wass4)

C_tensor = torch.from_numpy(
    -C / reg - 1
).requires_grad_(False)
c_tensor = torch.from_numpy(
    cvector).requires_grad_(False)
c_tensor = c_tensor.type(torch.FloatTensor)

rho0_tensor = torch.from_numpy(
    rho0
).requires_grad_(False).view(-1)
rhoT_tensor = torch.from_numpy(
    rhoT
).requires_grad_(True).view(-1)
rho0_tensor = rho0_tensor.type(torch.FloatTensor)
rhoT_tensor = rhoT_tensor.type(torch.FloatTensor)

M = torch.exp(C_tensor)
M = M.type(torch.FloatTensor)

# import ipdb; ipdb.set_trace()

u_vec0 = torch.ones(rho0_tensor.shape[0], dtype=torch.float32).requires_grad_(True)
v_vec0 = torch.ones(rho0_tensor.shape[0], dtype=torch.float32).requires_grad_(True)
p_opt0 = torch.zeros_like(M).requires_grad_(True)

cuda0 = torch.device('cuda:0')
cpu = torch.device('cpu')
device = cpu

wass4_torch = sinkhorn_torch(M,
    c_tensor,
    rho0_tensor,
    rhoT_tensor,
    u_vec0,
    v_vec0,
    p_opt0,
    device,
    delta=1e-2,
    lam=1e-12)

########################################################

print("wass4_torch", wass4_torch)

title_str = "N=%d\n" % (N)
title_str += "wass1 (stats)=%.3f\n" % (wass1)
# title_str += "wass2 (cvxpy)=%.3f\n" % (wass2)
# title_str += "wass3 (linprog)=%.3f\n" % (wass3)
title_str += "wass4 (sinkhorn)=%.3f\n" % (wass4)

# fig = plt.figure()
# ax = fig.add_subplot()

# ax.contourf(
#     meshes[0],
#     meshes[1],
#     rhoT.reshape(N, N),
#     cmap=cm.gray, alpha=0.5)

# ax.contourf(
#     meshes[0],
#     meshes[1],
#     rho0.reshape(N, N),
#     cmap=cm.jet, alpha=0.5)

# plt.tight_layout()
# plt.suptitle(title_str)

# plt.show()
