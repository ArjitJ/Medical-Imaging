import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
def add_legend(foo):
    foo._facecolors2d = foo._facecolors3d
    foo._edgecolors2d = foo._edgecolors3d
file = h5py.File('../data/bone3D.mat', 'r')
# print(list(file.keys()))
pointSets = file['shapesTotal']
pointSets = np.array(pointSets)
facets = np.array(file['TriangleIndex'])

temp = np.arange(0, 252, 1)
for i in facets:
    i -= 1
tri = Triangulation(temp, temp, triangles=facets.T)

fig = plt.figure()
ax = Axes3D(fig)
for i in range(pointSets.shape[0]):
    ax.scatter3D(pointSets[i, :, 0], pointSets[i, :, 1], pointSets[i, :, 2])
plt.savefig('../results/bone_raw_points')
plt.close()    
fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.plot_trisurf(pointSets[0, :, 0], pointSets[0, :, 1], triangles=tri.triangles, Z=pointSets[0, :, 2])
plt.savefig('../results/bone_first_example_with_faces')
plt.close()

numOfPointSets = pointSets.shape[0]
numOfPoints = pointSets.shape[1]

pointSetsCentered = pointSets - np.expand_dims(np.mean(pointSets, axis=1), axis=1)
# plt.scatter(pointSetsCentered[16, :, 0], pointSetsCentered[16, :, 1])
pointSetReshaped = pointSetsCentered.reshape(numOfPointSets, 756)
my_norm = np.linalg.norm(pointSetReshaped, axis=1)
pointSetReshaped = pointSetReshaped / np.expand_dims(my_norm, axis=1)
pointSetScaled = pointSetReshaped.reshape(numOfPointSets, numOfPoints, 3)


def kabsch(x, y):
    # input should be in the form (n_dims, n_samples)
    e = 0.00000000001
    u, s, vh = np.linalg.svd(np.dot(x, y.T))
    r = np.dot(vh.T, u.T)
    det = np.linalg.det(r)
    if abs(det - 1) < e:
        return r
    else:
        i = np.identity(u.shape[1])
        i[-1, -1] = -1
        r = np.dot(i, u.T)
        r = np.dot(vh.T, r)
        return r


def align_point_sets(v1, v2):
    return kabsch(v2.T, v1.T)


def meantotransform(z, pointSets):
    listoftransformations = []
    for i in range(numOfPointSets):
        listoftransformations.append(align_point_sets(z, pointSets[i, :, :]))
    return listoftransformations


def transformtomean(listoftransformations, pointSets):
    mymean = np.zeros((numOfPoints, 3))
    for i in range(numOfPointSets):
        mymean += np.dot(pointSets[i, :, :], listoftransformations[i].T)
    mymean /= numOfPointSets
    mymean = mymean / np.linalg.norm(mymean)
    return mymean


#
#
mean = np.mean(pointSetScaled, axis=0)
mean1 = np.copy(mean)
np.random.seed(0)
mean = pointSetScaled[np.random.randint(0, numOfPointSets), :, :]

optimalTransforms = []
numberOfIterations = 50
for i in range(numberOfIterations):
    optimalTransforms = meantotransform(mean, pointSetScaled)
    mean = transformtomean(optimalTransforms, pointSetScaled)
fig4 = plt.figure()
ax4 = Axes3D(fig4)

ax4.plot_trisurf(mean[:, 0], mean[:, 1], triangles=tri.triangles, Z=mean[:, 2])
plt.savefig('../results/bones_mean')
plt.close()
# print(sum(abs(mean - mean1)))
# covarianceMatrix = np.cov()

transformedPointSets = np.zeros([numOfPointSets, numOfPoints, 3])
for i in range(numOfPointSets):
    pointSetScaled[i, :, :] = np.dot(pointSetScaled[i, :, :], optimalTransforms[i].T)
covarianceMatrix = np.cov(np.transpose(pointSetScaled.reshape([numOfPointSets, 3 * numOfPoints])))
eigValues, eigVectors = np.linalg.eig(covarianceMatrix)
eigValues, eigVectors = abs(eigValues.real), abs(eigVectors.real)
idx = eigValues.argsort()[::-1]
eigValues = eigValues[idx]
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.plot(eigValues, marker='.')
ax5.set_title('All Eigen Values')
plt.savefig('../results/bone_eigenvalues')
plt.close()
eigVectors = eigVectors[:, idx]
allModesOfVariation = np.zeros([2, 3 * numOfPoints, numOfPoints, 3])

## allModeOfVariations[twodirections, number_of_modes, num_of_points, 2D]
#
for i in range(3 * numOfPoints):
    allModesOfVariation[0, i, :, :] = mean - 2 * np.reshape(eigVectors[:, i], [numOfPoints, 3]) * (eigValues[i] ** 0.5)
    allModesOfVariation[1, i, :, :] = mean + 2 * np.reshape(eigVectors[:, i], [numOfPoints, 3]) * (eigValues[i] ** 0.5)
#
## First mode of variation - eccentricity
fig6 = plt.figure()
ax6 = Axes3D(fig6)
add_legend(ax6.plot_trisurf(allModesOfVariation[0, 0, :, 0], allModesOfVariation[0, 0, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[0, 0, :, 2], label='mean-2*std'))
add_legend(ax6.plot_trisurf(mean[:, 0], mean[:, 1],
                 triangles=tri.triangles,
                 Z=mean[:, 2], label='mean'))
add_legend(ax6.plot_trisurf(allModesOfVariation[1, 0, :, 0], allModesOfVariation[1, 0, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[1, 0, :, 2], label='mean+2*std'))
ax6.legend()			

plt.savefig('../results/bone_first_variation')
plt.close()
fig7 = plt.figure()
ax7 = Axes3D(fig7)


add_legend(ax7.plot_trisurf(allModesOfVariation[0, 1, :, 0], allModesOfVariation[0, 1, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[0, 1, :, 2], label='mean-2*std'))
add_legend(ax7.plot_trisurf(mean[:, 0], mean[:, 1],
                 triangles=tri.triangles,
                 Z=mean[:, 2], label='mean'))
add_legend(ax7.plot_trisurf(allModesOfVariation[1, 1, :, 0], allModesOfVariation[1, 1, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[1, 1, :, 2], label='mean+2*std'))
ax7.legend()	
plt.savefig('../results/bone_second_variation')
plt.close()
fig8 = plt.figure()
ax8 = Axes3D(fig8)


add_legend(ax8.plot_trisurf(allModesOfVariation[0, 2, :, 0], allModesOfVariation[0, 2, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[0, 2, :, 2], label='mean-2*std'))
add_legend(ax8.plot_trisurf(mean[:, 0], mean[:, 1],
                 triangles=tri.triangles,
                 Z=mean[:, 2], label='mean'))
add_legend(ax8.plot_trisurf(allModesOfVariation[1, 2, :, 0], allModesOfVariation[1, 2, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[1, 2, :, 2], label='mean+2*std'))
ax8.legend()	
plt.savefig('../results/bone_third_variation')
plt.close()

fig9 = plt.figure()
ax9 = Axes3D(fig9)


add_legend(ax9.plot_trisurf(allModesOfVariation[0, 3, :, 0], allModesOfVariation[0, 3, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[0, 3, :, 2], label='mean-2*std'))
add_legend(ax9.plot_trisurf(mean[:, 0], mean[:, 1],
                 triangles=tri.triangles,
                 Z=mean[:, 2], label='mean'))
add_legend(ax9.plot_trisurf(allModesOfVariation[1, 3, :, 0], allModesOfVariation[1, 3, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[1, 3, :, 2], label='mean+2*std'))
ax9.legend()	
plt.savefig('../results/bone_fourth_variation')
plt.close()

fig10 = plt.figure()
ax10 = Axes3D(fig10)


add_legend(ax10.plot_trisurf(allModesOfVariation[0, 4, :, 0], allModesOfVariation[0, 4, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[0, 4, :, 2], label='mean-2*std'))
add_legend(ax10.plot_trisurf(mean[:, 0], mean[:, 1],
                 triangles=tri.triangles,
                 Z=mean[:, 2], label='mean'))
add_legend(ax10.plot_trisurf(allModesOfVariation[1, 4, :, 0], allModesOfVariation[1, 4, :, 1],
                 triangles=tri.triangles,
                 Z=allModesOfVariation[1, 4, :, 2], label='mean+2*std'))
ax10.legend()	
plt.savefig('../results/bone_fifth_variation')
plt.close()
