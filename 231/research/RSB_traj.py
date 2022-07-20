#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

#====================================================
# Make plots beautiful
#====================================================

def plot_params():
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.85 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 9
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8
    # learn how to configure:
    # http://matplotlib.sourceforge.net/users/customizing.html
    params = {'backend': 'ps',
              'axes.labelsize': 16,
              'legend.fontsize': tick_size,
              'legend.handlelength': 2.5,
              'legend.borderaxespad': 0,
              'axes.labelsize': label_size,
              'xtick.labelsize': tick_size,
              'ytick.labelsize': tick_size,
              'font.family': 'serif',
              'font.size': text_size,
              'font.serif': ['Computer Modern Roman'],
              'ps.usedistiller': 'xpdf',
              'text.usetex': True,
              'figure.figsize': fig_size,
              # include here any needed package for latex
              'text.latex.preamble': [r'\usepackage{amsmath}'],
              }
    return params

#====================================================
# Creating 3D arrow class
#====================================================

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

if __name__ == '__main__':
    params = plot_params()
    plt.rcParams.update(params)

    fig = plt.figure(1, figsize=params["figure.figsize"])  # figsize accepts only inches.
    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.15,
                        hspace=0.05, wspace=0.02)

    #====================================================
    # Actual plot
    #====================================================

    rho_0 = np.loadtxt('rho_0.txt')
    rho_1 = np.loadtxt('rho_1.txt')
    rho_1_unc = np.loadtxt('rho_1_unc.txt')

    rho_between = np.loadtxt('rho_between.txt')
    t_between = [0.25, 0.5, 0.75, 0.9]

    cntrl_traj = np.loadtxt('cont_traj.txt')

    nSamples = 50; 
    h = .001;
    t = np.arange(1001)
    # fig, ax = plt.subplots()


    t0  = np.zeros((nSamples,1))
    t1  = np.ones((nSamples,1))

    tt = np.zeros((1,1001))

    # plt.figure()
    ax = plt.subplot(projection='3d')

    # plot given initial state PDF

    ax.plot(rho_0[:,0], t0, rho_0[:,1], color='k',lw=1)
    ax.add_collection3d(plt.fill_between(rho_0[:,0], 0*rho_0[:,1], rho_0[:,1], color='gray', alpha=0.3), zs=0, zdir='y')

    # plot given terminal state PDF

    ax.plot(rho_1[:,0], t1, rho_1[:,1], color='k',lw=1)
    ax.add_collection3d(plt.fill_between(rho_1[:,0], 0*rho_1[:,1], rho_1[:,1], color='gray', alpha=0.3), zs=1, zdir='y')

    # plot uncontrolled terminal state PDF

    ax.plot(rho_1_unc[:,0], t1, rho_1_unc[:,1], color='r',lw=1)
    ax.add_collection3d(plt.fill_between(rho_1_unc[:,0], 0*rho_1_unc[:,1], rho_1_unc[:,1], color='red', alpha=0.15), zs=1, zdir='y')

    # plot controlled transient state PDFs

    for j in range(len(t_between)):
         t_now = t_between[j]*np.ones((nSamples,1))
         ax.plot(rho_between[:,0], t_now, rho_between[:,j+1], color='k',lw=1)
         ax.add_collection3d(plt.fill_between(rho_between[:,0], 0*rho_between[:,j+1], rho_between[:,j+1], color='gray', alpha=0.3), zs=t_between[j], zdir='y')

    # plot controlled state sample paths

    for i in range(20):
        ax.plot(cntrl_traj[i,:],t*h,t*0,lw=.3)


    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$", rotation="horizontal")
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    # ax.set_zlabel('label text', rotation=0)
    # ax.set_zlabel(r"$\rho(x,t)$")

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.legend(markerscale=2, scatterpoints=3,frameon=False)

    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the spines
    # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


    ax.view_init(elev=51, azim=-22) 

    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    plt.xticks((0, 1))
    plt.yticks((0, 1))
    ax.set_yticklabels(('$t=0$', '$t=1$'))
    ax.set_zticks([0,0.5,1.0,1.5,2.0])

    ax.get_xaxis().set_tick_params(direction='in')
    ax.get_yaxis().set_tick_params(direction='in')


    ax.text(0.1, -0.22, 1.85, r'$\rho_{0}(x)$',size=19)
    ax.text(0.5, 1.05, 1.5, r'$\rho_{1}(x)$',size=19)
    ax.text(0.5, 0.842, 2.4, r'$\rho_{1}^{\text{unc}}(x)$',size=19, color='r')

    a = Arrow3D([0.6, 0.55], [0.842, 0.841], 
                    [2.55, 1.95], mutation_scale=20, 
                    lw=1.5, arrowstyle="-|>", color="r")
    ax.add_artist(a)

    plt.show()

    # plt.savefig('RSB.png', dpi=300)



