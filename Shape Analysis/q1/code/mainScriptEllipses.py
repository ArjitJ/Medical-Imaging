import numpy as np
import h5py
import matplotlib.pyplot as plt

file = h5py.File('../data/ellipses2D.mat', 'r')
# print(list(file.keys()))
pointSets = file['pointSets']
pointSets = np.array(pointSets)

numOfPointSets = pointSets.shape[0]
numOfPoints = pointSets.shape[1]
numOfDimensions = pointSets.shape[2]

plt.plot(pointSets[:, :, 0].T, pointSets[:, :, 1].T, 'o', markersize=1)
plt.title('Raw points')
plt.savefig('../results/ellipse_raw_points')
plt.close()
pointSetsCentered = pointSets - np.expand_dims(np.mean(pointSets, axis=1), axis=1)
# plt.scatter(pointSetsCentered[16, :, 0], pointSetsCentered[16, :, 1], s=5)
pointSetReshaped = pointSetsCentered.reshape(numOfPointSets, 64)
my_norm = np.linalg.norm(pointSetReshaped, axis=1)
pointSetReshaped = pointSetReshaped / np.expand_dims(my_norm, axis=1)
pointSetScaled = pointSetReshaped.reshape(numOfPointSets, numOfPoints, numOfDimensions)


def align_point_sets(v1, v2):
    newv = np.copy(v2)
    newv[:,[0, 1]] = newv[:,[1, 0]]
    newv[:,0] = -1*newv[:,0]
    b = sum(sum(np.multiply(v1,newv)))
    a = sum(sum(np.multiply(v1, v2)))
    return np.array([[a, b], [-1*b,a]])

# The transformation has been right multiplied and hence the transpose is taken accordingly


def meantotransform(z, pointSets):
    listoftransformations = []
    for i in range(numOfPointSets):
        listoftransformations.append(align_point_sets(z, pointSets[i, :, :]))
    return listoftransformations


def transformtomean(listoftransformations, pointSets):
    mymean = np.zeros((numOfPoints, numOfDimensions))
    for i in range(numOfPointSets):
        mymean += np.dot(pointSets[i, :, :], listoftransformations[i])
    mymean /= numOfPointSets
    mymean = mymean / np.linalg.norm(mymean)
    return mymean


# mean = np.mean(pointSetScaled, axis=0)
# mean1 = np.copy(mean)
np.random.seed(0)
mean = pointSetScaled[np.random.randint(0, numOfPointSets), :, :]

optimalTransforms = []
numberOfIterations = 100
for i in range(numberOfIterations):
    optimalTransforms = meantotransform(mean, pointSetScaled)
    mean = transformtomean(optimalTransforms, pointSetScaled)

# plt.scatter(mean[:, 0], mean[:, 1], s=5)
# plt.show()
# covarianceMatrix = np.cov()

# transformedPointSets = np.zeros([numOfPointSets, numOfPoints, numOfDimensions])
for i in range(numOfPointSets):
    pointSetScaled[i, :, :] = np.dot(pointSetScaled[i, :, :], optimalTransforms[i])

plt.plot(pointSetScaled[:, :, 0].T, pointSetScaled[:, :, 1].T, 'o', markersize=1)
plt.plot(mean[:, 0], mean[:, 1], 'o', c='black')
plt.title('Transformed dataset with mean (black)')
plt.savefig('../results/ellise_mean')
plt.close()
covarianceMatrix = np.cov(np.transpose(pointSetScaled.reshape([numOfPointSets, numOfDimensions*numOfPoints])))
eigValues, eigVectors = np.linalg.eig(covarianceMatrix)
idx = eigValues.argsort()[::-1]
eigValues = eigValues[idx]
plt.plot(eigValues, marker='.')
plt.title('All Eigen Values')
plt.savefig('../results/ellipse_eigenvalue')
plt.close()
eigVectors = eigVectors[:, idx]

numOfVariations = min(numOfDimensions*numOfPoints, numOfPointSets)
allModesOfVariation = np.zeros([2, numOfVariations, numOfPoints, numOfDimensions])

# allModesOfVariation[twodirections, number_of_modes, num_of_points, 2D]

for i in range(numOfVariations):
    allModesOfVariation[0, i, :, :] = mean - 2*np.reshape(eigVectors[:, i],
                                                          [numOfPoints, numOfDimensions])*(eigValues[i] ** 0.5)
    allModesOfVariation[1, i, :, :] = mean + 2*np.reshape(eigVectors[:, i],
                                                          [numOfPoints, numOfDimensions])*(eigValues[i] ** 0.5)

# First mode of variation - eccentricity

plt.scatter(pointSetScaled[:, :, 0].T, pointSetScaled[:, :, 1].T, c='grey', s=1)
plt.scatter(mean[:, 0], mean[:, 1], label='mean')
plt.scatter(allModesOfVariation[0, 0, :, 0], allModesOfVariation[0, 0, :, 1], label='mean-2*std')
plt.scatter(allModesOfVariation[1, 0, :, 0], allModesOfVariation[1, 0, :, 1], label='mean+2*std')
plt.title('First mode of variation')
plt.legend()
plt.savefig('../results/ellipse_first_variation')
plt.close()

# Second mode of variation - scaling

plt.scatter(pointSetScaled[:, :, 0].T, pointSetScaled[:, :, 1].T, c='grey', s=1)
plt.scatter(mean[:, 0], mean[:, 1], label='mean')
plt.scatter(allModesOfVariation[0, 1, :, 0], allModesOfVariation[0, 1, :, 1], label='mean-2*std')
plt.scatter(allModesOfVariation[1, 1, :, 0], allModesOfVariation[1, 1, :, 1], label='mean+2*std')
plt.title('Second mode of variation')
plt.legend()
plt.savefig('../results/ellipse_second_variation')
plt.close()
