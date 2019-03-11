# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp2d
file = loadmat('../data/phantom256.mat')

img = file['imageAC']
img = np.array(img, dtype=float)

from skimage.transform import radon, iradon
theta = np.arange(0, 180, 3)
radonT = radon(img, theta, circle=False)
iradonT = iradon(radonT, theta, circle=False)


def foo(w, L):
    return np.abs(w)


def ramlak(w, L):
    tmp = np.abs(w)
    return tmp<=L


def shepplogan(w, L):
    tmp = np.abs(w)
    sin = np.divide(np.sin((np.pi*2/L)*w), (np.pi*2/L)*(w+1E-10))
    return np.multiply(sin, tmp<=L)


def lowpasscos(w, L):
    tmp = np.abs(w)
    cos = np.cos((np.pi*2/L)*w)
    return np.multiply(cos, tmp<=L)


def filteredbkp(radonTrans, filterfunc, theta, Lscale=1, override=False, Lval=1):
    dffts = np.fft.fft(radonTrans, axis=0)
    freqs = np.fft.fftfreq(radonTrans.shape[0])
    n = radonTrans.shape[0]
    wmax = None
    if n%2 == 0:
        wmax = n/2 - 1
    else:
        wmax = (n-1)/2
    wmax /= n
    L = Lscale*wmax
    if override:
        L = Lval
    filtereddffts = np.multiply(dffts, filterfunc(freqs, L).reshape(-1, 1))
    ifilteredfft = np.fft.ifft(filtereddffts, axis=0)
    return iradon(ifilteredfft, theta, circle=False)


fig = plt.figure()
img1 = filteredbkp(radonT, ramlak, theta)
ax = fig.add_subplot(3, 3, 1)
ax.imshow(img1, cmap='gray')
ax.set_title('ramlak, W_max')
img1 = filteredbkp(radonT, shepplogan, theta)
ax = fig.add_subplot(3, 3, 2)
ax.imshow(img1, cmap='gray')
ax.set_title('shepp-logan, W_max')
img1 = filteredbkp(radonT, lowpasscos, theta)
ax = fig.add_subplot(3, 3, 3)
ax.imshow(img1, cmap='gray')
ax.set_title('low-pass, W_max')
img1 = filteredbkp(radonT, ramlak, theta, 0.5)
ax = fig.add_subplot(3, 3, 4)
ax.imshow(img1, cmap='gray')
ax.set_title('ramlak, W_max/2')
img1 = filteredbkp(radonT, shepplogan, theta, 0.5)
ax = fig.add_subplot(3, 3, 5)
ax.imshow(img1, cmap='gray')
ax.set_title('shepp-logan, W_max/2')
img1 = filteredbkp(radonT, lowpasscos, theta, 0.5)
ax = fig.add_subplot(3, 3, 6)
ax.imshow(img1, cmap='gray')
ax.set_title('low-pass, W_max/2')
ax = fig.add_subplot(3, 3, 7)
ax.imshow(img, cmap='gray')
ax.set_title('phantom(256)')
ax = fig.add_subplot(3, 3, 8)
ax.imshow(iradonT, cmap='gray')
ax.set_title('in-built')
fig.set_size_inches(18.5, 10.5)
plt.savefig('../results/allvals.png')
plt.close()
# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

# https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function


from scipy.signal import convolve2d


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


s0 = img
s1 = conv2(img, fspecial_gauss(11, 1))
s5 = conv2(img, fspecial_gauss(51, 5))
radon0 = radon(s0, theta, circle=False)
radon1 = radon(s1, theta, circle=False)
radon5 = radon(s5, theta, circle=False)
r0 = filteredbkp(radon0, ramlak, theta)
r1 = filteredbkp(radon1, ramlak, theta)
r5 = filteredbkp(radon5, ramlak, theta)


def RRMSE(A, B):
    # Always use noiseless image as A
    # Aa = np.abs(A)
    return np.sqrt(np.sum(np.square(A - B))) / np.sqrt(np.sum(np.square(A)))


fig = plt.figure()
ax = fig.add_subplot(3, 2, 1)
ax.imshow(s0, cmap='gray')
ax.set_title('S0')
ax = fig.add_subplot(3, 2, 2)
ax.imshow(r0, cmap='gray')
ax.set_title('R0, RRMSE: %0.4f' % RRMSE(s0, r0))

ax = fig.add_subplot(3, 2, 3)
ax.imshow(s1, cmap='gray')
ax.set_title('S1')
ax = fig.add_subplot(3, 2, 4)
ax.imshow(r1, cmap='gray')
ax.set_title('R1, RRMSE: %0.4f' % RRMSE(s1, r1))

ax = fig.add_subplot(3, 2, 5)
ax.imshow(s5, cmap='gray')
ax.set_title('S5')
ax = fig.add_subplot(3, 2, 6)
ax.imshow(r5, cmap='gray')
ax.set_title('R5, RRMSE: %0.4f' % RRMSE(s5, r5))
fig.set_size_inches(15, 11)
plt.savefig('../results/filteredBackProjections.png')
plt.close()
n = radonT.shape[0]
wmax = None
if n%2 == 0:
    wmax = n//2 - 1
else:
    wmax = (n-1)//2
    
vals0 = []
vals1 = []
vals5 = []
Ls = np.arange(1, wmax+1) / n  
for i in Ls:
    rt0 = filteredbkp(radon0, ramlak, theta, 1, True, i)
    rt1 = filteredbkp(radon1, ramlak, theta, 1, True, i)
    rt5 = filteredbkp(radon5, ramlak, theta, 1, True, i)
    vals0.append(RRMSE(s0, rt0))
    vals1.append(RRMSE(s1, rt1))
    vals5.append(RRMSE(s5, rt5))
    
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ax.plot(Ls, vals0)
ax = fig.add_subplot(2, 2, 2)
ax.plot(Ls, vals1)
ax = fig.add_subplot(2, 2, 3)
ax.plot(Ls, vals5)
plt.savefig('../results/allplots.png')
plt.close()
