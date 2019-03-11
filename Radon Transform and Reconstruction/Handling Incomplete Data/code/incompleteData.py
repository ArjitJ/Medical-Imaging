# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp2d
from skimage.transform import radon, iradon
phantom = h5py.File('../data/myPhantom.mat', 'r')
ct = h5py.File('../data/CT_Chest.mat', 'r')
phantom = np.array(phantom['imageAC'], dtype=float)
ct = np.array(ct['imageAC'], dtype=float)

def RRMSE(A, B):
    # Always use noiseless image as A
    Aa = np.abs(A)
    return np.sqrt(np.sum(np.sum(np.square(Aa - np.abs(B))))) / np.sqrt(np.sum(np.sum(np.square(Aa))))

RRMSEPhantom = []
RRMSECT = []
strings = ['../results/phantomRRMSE.png', '../results/CTRRMSE.png']
range_theta = np.arange(0, 180)
for theta in range_theta:
    theta_comp = np.arange(theta, theta + 151) % 180
    Rf = radon(phantom, theta_comp, circle=False)
    iRf = iradon(Rf, theta_comp, circle=False)
    rrmse = RRMSE(phantom, iRf)
    RRMSEPhantom.append(rrmse)
for theta in range_theta:
    theta_comp = np.sort(np.arange(theta, theta + 151) % 180)
    Rf = radon(ct, theta_comp, circle=False)
    iRf = iradon(Rf, theta_comp, circle=False)
    rrmse = RRMSE(ct, iRf)
    RRMSECT.append(rrmse)
minRRPhantom = np.min(RRMSEPhantom)
minThetaPhantom = np.argmin(RRMSEPhantom)
minRRCT = np.min(RRMSECT)
minThetaCT = np.argmin(RRMSECT)
plt.plot(RRMSEPhantom)
plt.savefig(strings[0])
plt.close()
plt.plot(RRMSECT)
plt.savefig(strings[1])
plt.close()
pTheta = np.sort(np.arange(minThetaPhantom, minThetaPhantom + 151) % 180)
Rf = radon(phantom, pTheta, circle=False)
iRf = iradon(Rf, pTheta, circle=False)
plt.imshow(iRf, cmap='gray')
plt.savefig('../results/phantom.png')
plt.close()
cTheta = np.sort(np.arange(minThetaCT, minThetaCT + 151) % 180)
Rf = radon(ct, cTheta, circle=False)
iRf = iradon(Rf, cTheta, circle=False)
plt.imshow(iRf, cmap='gray')
plt.savefig('../results/ct.png')
plt.close()
