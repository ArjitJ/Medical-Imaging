# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 00:56:53 2019

@author: Arjit Jain, Yash Sharma
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp2d
file = loadmat('../data/phantom128.mat')

img = file['imageAC']
img = np.array(img)


# def pred()

def myIntegration(Z, t, theta_deg, step=0.1):
    originy = Z.shape[0]*step/2
    originx = Z.shape[1]*step/2
    theta = np.deg2rad(theta_deg)
    new_X = np.arange(0, Z.shape[0]*step, step)
    new_Y = np.arange(0, Z.shape[1]*step, step)
    epsilon = step*1.5
    xx, yy = np.meshgrid(new_X, new_Y)
    foo = (xx - originx)*np.cos(theta) + (originy - yy) * np.sin(theta) - t
    indices = np.abs(foo) < epsilon
    integral = step*np.sum(Z[indices])
    return integral


def myRadonTrans(img, step, t=np.arange(-90, 95, 5), theta=np.arange(0, 180, 5)):
    X = np.arange(img.shape[0])
    Y = np.arange(img.shape[1])
    Z = img
    f = interp2d(X, Y, Z, kind='linear')
    new_X = np.arange(0, img.shape[0], step)
    new_Y = np.arange(0, img.shape[1], step)
    new_Z = np.copy(f(new_X, new_Y))
    tt, thethe = np.meshgrid(t, theta)
    tt = tt.reshape(-1, 1)
    thethe = thethe.reshape(-1, 1)
    integrator = np.vectorize(lambda x, y: myIntegration(new_Z, x, y, step))
    radon = integrator(tt, thethe).reshape(theta.size, t.size).T
    return radon

from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure()
for i, s in enumerate([0.5, 1, 3]):
    Rf = myRadonTrans(img, s)
    ax = fig.add_subplot(1, 3, i+1)
    im = ax.imshow(Rf)
    ax.set_title('step size : ' + str(s))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)
fig.set_size_inches(10, 5)    
plt.savefig('../results/differentStepSizes.png')
plt.close()

t2 = np.arange(-90, 91)
theta2 = np.array([0, 90])
Rf2 = myRadonTrans(img, 0.1, t2, theta2)
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(t2, Rf2[:, 0])
ax.set_title('theta = ' + str(theta2[0]))
ax = fig.add_subplot(2, 1, 2)
ax.plot(t2, Rf2[:, 1])
ax.set_title('theta = ' + str(theta2[1]))
plt.savefig('../results/1DTransform.png')

