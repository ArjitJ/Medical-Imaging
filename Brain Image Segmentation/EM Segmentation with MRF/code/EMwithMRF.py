# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

file = h5py.File('../data/SegmentBrainGmmEmMrf.mat', 'r')
imageData = np.array(file['imageData'])
imageMask = np.array(file['imageMask'])


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


def initialize_memberships(means, image_data_masked):
    membership = np.zeros((image_data_masked.shape[0], image_data_masked.shape[1], means.size), dtype='float64')
    for i in range(means.size):
        membership[:, :, i] = np.argmin(np.square(image_data_masked[:, :, np.newaxis] - means.reshape([1, 1, -1])), axis=2)==i
    return membership


def get_mean(membership, image_data_masked):
    tmpdata = image_data_masked.reshape([-1, 1])
    tmpm = np.zeros(tmpdata.shape) + membership.reshape([-1, K])
    tmpdata = np.zeros(tmpm.shape) + tmpdata
    return np.average(tmpdata, weights=tmpm, axis=0)


def get_variance(mean, membership, image_data_masked):
    num = np.square(image_data_masked[:, :, np.newaxis] - mean.reshape([1, 1, -1]))
    num = np.multiply(membership, num)
    num = np.sum(np.sum(num, axis=0), axis=0)
    denom = np.sum(np.sum(membership, axis=0), axis=0)
    return np.divide(num, denom)


def get_weights2(imageData, MRF, means, variances, beta):
    something = np.zeros((imageData.size, len(means)))
    lenX = MRF.shape[0]
    lenY = MRF.shape[1]
    for index in np.arange(MRF.size):
        i = index // lenX
        j = index % lenY
        if imageMask[i][j] == 0:
            continue
        prior = np.array([int(MRF[(i+1)%lenX][j]!=k and imageMask[(i+1)%lenX][j] == 1) \
                + int(MRF[(i-1)%lenX][j]!=k and imageMask[(i-1)%lenX][j] == 1) \
                + int(MRF[i][(j+1)%lenY]!=k and imageMask[i][(j+1)%lenY] == 1) \
                + int(MRF[i][(j-1)%lenY]!=k and imageMask[i][(j-1)%lenY] == 1) for k in range(K)])
        prior = prior*beta
        something[index, :] = prior
    return np.exp(-1*something).reshape(imageData.shape + (len(means), ))    


def get_weights3(imageData, means, variances, beta):
    new_imageData = imageData[:, :, np.newaxis]
    new_means = means[np.newaxis, np.newaxis, :]
    new_variances = variances[np.newaxis, np.newaxis, :]
    temp = np.divide(np.square(new_imageData - new_means), 2*new_variances)
    temp*= (1-beta)
    return np.exp(-1*temp)          


def get_weights_final(imageData, MRF, means, variances, beta):
    num1 = get_weights3(imageData, means, variances, beta)
    num2 = get_weights2(imageData, MRF, means, variances, beta)
    temp = np.multiply(num1, num2)
    temp = np.divide(temp, np.sum(temp, axis=2)[:, :, np.newaxis])
    return temp*imageMask[:, :, np.newaxis]


def get_prior(MRF, beta):
    something = np.zeros(imageData.size)
    lenX = MRF.shape[0]
    lenY = MRF.shape[1]
    for index in np.arange(MRF.size):
        i = index // lenX
        j = index % lenY
        k = MRF[i][j]
        if imageMask[i][j] == 0:
            continue
        prior = int(MRF[(i+1)%lenX][j]!=k and imageMask[(i+1)%lenX][j] == 1) \
                + int(MRF[(i-1)%lenX][j]!=k and imageMask[(i-1)%lenX][j] == 1) \
                + int(MRF[i][(j+1)%lenY]!=k and imageMask[i][(j+1)%lenY] == 1) \
                + int(MRF[i][(j-1)%lenY]!=k and imageMask[i][(j-1)%lenY] == 1)
        prior = prior*beta
        something[index] = prior
    return something.reshape(imageData.shape)


def get_likelihood(imageData, MRF, means, variances, beta):
    flattened_imageData = imageData.reshape(-1, 1)
    flattened_MRF = MRF.reshape(-1, 1)
    vfunc = np.vectorize(lambda x, y: ((x-means[y])**2/(2*variances[y])))
    temp = vfunc(flattened_imageData, flattened_MRF)
    return (1-beta)*temp.reshape(imageData.shape)*imageMask


def get_map(MRF, imageData, beta, means, variances, costs):
    newMRF = np.copy(MRF)
    lenX = imageData.shape[0]
    lenY = imageData.shape[1]
    for i in range(lenX):
        for j in range(lenY):
            if imageMask[i][j]==0:
                continue
            index = None
            val = None
            for k in range(len(means)):
                prior = int(MRF[(i+1)%lenX][j]!=k and imageMask[(i+1)%lenX][j] == 1) \
                        + int(MRF[(i-1)%lenX][j]!=k and imageMask[(i-1)%lenX][j] == 1) \
                        + int(MRF[i][(j+1)%lenY]!=k and imageMask[i][(j+1)%lenY] == 1) \
                        + int(MRF[i][(j-1)%lenY]!=k and imageMask[i][(j-1)%lenY] == 1)
                likelihood = np.square(imageData[i][j] - means[k])/variances[k]
                if val == None or val > beta*prior + (1-beta)*likelihood:
                    index = k
                    val = beta*prior + (1-beta)*likelihood
            newMRF[i][j] = index          
    newcosts = get_prior(newMRF, beta) + get_likelihood(imageData, newMRF, means, variances, beta)
    newcosts = np.sum(np.multiply(imageMask, newcosts))
    if newcosts < costs:
        return newMRF, newcosts
    else:
        return MRF, costs


def ICM(imageData, means, variances, beta, MRFgiven=False, MRF=None):
    new_imageData = imageData[:, :, np.newaxis]
    new_means = means[np.newaxis, np.newaxis, :]
    new_variances = variances[np.newaxis, np.newaxis, :]
    if MRFgiven==False:
        MRF = np.argmin(np.divide(np.square(new_imageData - new_means), 2*new_variances), axis=2)
    costs = get_prior(MRF, beta) + get_likelihood(imageData, MRF, means, variances, beta)
    costs = np.sum(np.multiply(imageMask, costs))
    newMRF, newcosts = get_map(MRF, imageData, beta, means, variances, costs)
    while not (newMRF == MRF)[np.argwhere(imageMask)].all():
        MRF, costs = newMRF, newcosts
        newMRF, newcosts = get_map(MRF, imageData, beta, means, variances, costs)
    return MRF


epsilon = 1e-8
K = 3
numIter = 20
beta = 0.69
np.random.seed(8)
something = imageMask[:, :, np.newaxis]
means = k_means_pp_init(imageData, imageMask, K)
memberships = initialize_memberships(means, imageData)
memberships *= something
class1 = imageData[memberships[:, :, 0] > epsilon]
class2 = imageData[memberships[:, :, 1] > epsilon]
class3 = imageData[memberships[:, :, 2] > epsilon]
means = np.array([np.mean(class1.reshape(1, -1)), np.mean(class2.reshape(1, -1)), np.mean(class3.reshape(1, -1))])
variances = np.array([np.var(class1.reshape(1, -1)), np.var(class2.reshape(1, -1)), np.var(class3.reshape(1, -1))])
# print(variances)
vals = []
MRFgiven = False
MRF = None
for i in range(numIter):
    # E step
    MRF = ICM(imageData, means, variances, beta, MRFgiven, MRF)
    costs = get_prior(MRF, beta) + get_likelihood(imageData, MRF, means, variances, beta)
    costs = np.sum(np.multiply(imageMask, costs))
    MRFgiven = True
    vals.append(costs)
    memberships = get_weights_final(imageData, MRF, means, variances, beta)
    # M step
    means = get_mean(memberships, imageData)
    variances = get_variance(means, memberships, imageData)

img = np.zeros([256, 256, K])
for i in range(K):
    k0 = np.argmax(memberships, axis=2) == i
    img[:, :, i] = k0.reshape(imageData.shape)
plt.imshow(img * imageMask[:, :, np.newaxis])
legend_elements = [mpatches.Patch(color='r', label="cerebrospinal fluid"),
                   mpatches.Patch(color='b', label="grey matter"),
                   mpatches.Patch(color='#00ff00', label="white matter")]
plt.legend(handles=legend_elements)
plt.title((r'$\beta : %0.3f $' % beta))
plt.savefig('../results/segmentedBrainQ2.png')
plt.close()
fig = plt.figure()
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    ax.set_title('class: ' + str(i))
    ax.imshow(memberships[:, :, i], cmap='gray')

plt.savefig('../results/membershipsQ2.png')
plt.close()
plt.plot(vals)
plt.title('Plot of decreasing value of negative log posterior')
plt.savefig('../results/logposterior.png')
plt.close()
