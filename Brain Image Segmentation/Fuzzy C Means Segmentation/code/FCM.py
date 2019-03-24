# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import matplotlib.patches as mpatches

file = h5py.File('../data/SegmentBrain.mat', 'r')
imageData = np.array(file['imageData'], dtype=np.float64)
imageMask = np.array(file['imageMask'], dtype='int64')
np.random.seed(2)
K = 3
epsilon = 1e-18

def choice_cost(samples, T):
    x = np.square(samples.reshape([-1, 1]) - T.reshape([1, -1]))
    x = np.min(x, axis=1)
    return x

def k_means_pp_init(data, mask, k=3):
    T = np.zeros([k, ], dtype=float)
    flattened_data = np.ndarray.flatten(data)
    flattened_mask = np.ndarray.flatten(mask)
    T[0] = np.random.choice(flattened_data, p=flattened_mask/np.sum(flattened_mask))
    for i in range(1, k):
        costs = choice_cost(flattened_data, T[:i])
        masked_costs = np.multiply(flattened_mask, costs)
        masked_costs = masked_costs/np.sum(masked_costs)
        T[i] = np.random.choice(flattened_data, p=masked_costs)
    return T

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def getdistances(imageData, means, bias, weights):
    means = means.reshape(-1, )
    expanded_means = means[np.newaxis, np.newaxis, :]
    expanded_bias = bias[:, :, np.newaxis]
    expanded_imageData = imageData[:, :, np.newaxis]
    expanded_weights = weights[:, :, np.newaxis]
    temp = np.square(expanded_imageData - np.multiply(expanded_means, expanded_bias))
    return convolve(temp, expanded_weights).reshape(-1, means.shape[0]).T

def getmembership(distances, q):
    temp = np.power(1/(distances + epsilon), 1/(q-1))
    return np.divide(temp, np.sum(temp, axis=0)+epsilon).T

def getbias(weights, means, imageData, memberships, q):
    temp = np.power(memberships, q)
    num2 = np.dot(temp, means).reshape(imageData.shape)
    denom2 = np.dot(temp, np.square(means)).reshape(imageData.shape)
    num1 = convolve(np.multiply(imageData, num2), weights)
    denom1 = convolve(denom2, weights)
    return np.divide(num1, denom1+epsilon)

def getmeans(weights, bias, imageData, memberships, q):
    temp = np.power(memberships, q)
    num2 = convolve(bias, weights)
    num1 = np.dot(np.multiply(imageData, num2).reshape(1, -1), temp).T
    denom2 = convolve(np.square(bias), weights).reshape(1, -1)
    denom1 = np.dot(denom2, temp).T
    return np.divide(num1, denom1+epsilon)

def getcost(distances, memberships, q):
    mysum=0
    for i in range(distances.shape[1]):
        mysum = mysum + np.dot(np.power(memberships[i, :], q), distances[:, i])
    return mysum

q = 1.6
numIter = 20
neighbour = 4
imageMaskFlattened = imageMask.reshape(-1, 1)
initialBiasConstant = 2
sigma = 1
N = imageData.size
bias = np.zeros(imageData.shape) + initialBiasConstant
#bias = np.copy(imageMask)
distances = None
memberships = None
means_init = np.sort(imageData.reshape(-1, ) * imageMask.reshape(-1, ))
means_init = means_init[np.argwhere(means_init)]
means_init = [np.percentile(means_init, i) for i in np.arange(100/(K+1), 100, 100/(K+1))]
means = np.sort(means_init)
firstMean = means[0]
weights = fspecial_gauss(neighbour, sigma)
vals = []
#imageData *= imageMask
for i in range(numIter):
    distances = getdistances(imageData, means, bias, weights)
    memberships = getmembership(distances, q)
    memberships *= imageMaskFlattened
    means = getmeans(weights, bias, imageData, memberships, q)
    means[0] = firstMean
    bias = getbias(weights, means, imageData, memberships, q)
    vals.append(getcost(distances, memberships, q))
    
img = np.zeros(imageData.shape + (K,))
class1 = (np.argmax(memberships, 1) == 1).reshape(imageData.shape)
class2 = (np.argmax(memberships, 1) == 2).reshape(imageData.shape)
class3 = (np.argmax(memberships, 1) == 0).reshape(imageData.shape)

plt.plot(vals)
# plt.show()
plt.title("Objective function values against iterations")
plt.savefig("../results/vals.png")
plt.close()

plt.imshow(imageData, cmap='gray')
# plt.show()
plt.title("Corrupted Image")
plt.savefig("../results/corruptedImage.png")
plt.close()

memberships_new = memberships.reshape(256, 256, 3)
img[:, :, 0] = class1
img[:, :, 1] = class3
img[:, :, 2] = class2
plt.imshow(img*imageMask[:, :, np.newaxis])
# plt.show()
legend_elements = [mpatches.Patch(color='r', label="cerebrospinal fluid"),
                   mpatches.Patch(color='b', label="grey matter"),
                   mpatches.Patch(color='#00ff00', label="white matter")]
plt.legend(handles=legend_elements)
plt.title("Optimal Class Membership Image Estimates")
plt.savefig("../results/binaryClassMembership.png")
plt.close()

plt.imshow(memberships_new)
# plt.show()
plt.title("Optimal Class Membership Image Estimates")
plt.savefig("../results/ClassMembership.png")
plt.close()

fig = plt.figure()
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    ax.set_title('class: ' + str(i))
    ax.imshow(memberships_new[:, :, i], cmap='gray')

plt.savefig('../results/membershipsQ1.png')
plt.close()
plt.imshow(memberships_new[:,:,0], cmap='gray')
# plt.show()
plt.title("Class 1")
plt.savefig("../results/Class1.png")
plt.close()

plt.imshow(memberships_new[:,:,1], cmap='gray')
# plt.show()
plt.title("Class 2")
plt.savefig("../results/Class2.png")
plt.close()

plt.imshow(memberships_new[:,:,2], cmap='gray')
# plt.show()
plt.title("Class 3")
plt.savefig("../results/Class3.png")
plt.close()

plt.imshow(bias*imageMask, cmap='gray')
# plt.show()
plt.title("Bias")
plt.savefig("../results/Bias.png")
plt.close()

biasRemoved = np.zeros(imageData.shape)
for i in range(K):
    biasRemoved += memberships_new[:, :, i]*means[i]

plt.imshow(biasRemoved, cmap='gray')
# plt.show()
plt.title("Bias Removed")
plt.savefig("../results/BiasRemoved.png")
plt.close()

residual = imageData - (bias*biasRemoved)

plt.imshow(residual, cmap='gray')
# plt.show()
plt.title("Residual Image")
plt.savefig("../results/ResidualImage.png")
plt.close()
