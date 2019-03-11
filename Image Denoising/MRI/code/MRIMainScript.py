import numpy as np
import h5py
import matplotlib.pyplot as plt

file = h5py.File('../data/ImageDenoisingBrainNoisy.mat', 'r')
imageNoisy = file['imageNoisy']

def mycomplex(a):
    return a[0] + 1j * a[1]

imageNoisy = np.vectorize(mycomplex)(np.array(imageNoisy))

noiseUL = imageNoisy[:125, :36].reshape(-1, 1)
noiseBL = imageNoisy[:29, 228:].reshape(-1, 1)
noiseBR = imageNoisy[160:, 230:].reshape(-1, 1)
noiseUR = imageNoisy[163:, 30].reshape(-1, 1)
bgnoise = np.concatenate((noiseBL, noiseBR, noiseUL, noiseUR))
rbgnoise = np.real(bgnoise)
ibgnoise = np.imag(bgnoise)
#print("std real", np.std(rbgnoise))
print("Noise level", np.std(ibgnoise))

def cost(image, func):
    a1 = np.roll(image, -1, 0)
    a2 = np.roll(image, 1, 0)
    a3 = np.roll(image, -1, 1)
    a4 = np.roll(image, 1, 1)
    return sum(sum((func(a1 - image) + func(a2 - image) + func(a3 - image) + func(a4 - image))))


def grad(image, func):
    val = 0
    a1 = np.roll(image, -1, 0)
    a2 = np.roll(image, 1, 0)
    a3 = np.roll(image, -1, 1)
    a4 = np.roll(image, 1, 1)
    val += np.multiply(image - a1, func(a1 - image))
    val += np.multiply(image - a2, func(a2 - image))
    val += np.multiply(image - a3, func(a3 - image))
    val += np.multiply(image - a4, func(a4 - image))
    return 2 * val


def g1(X):
    return np.square(np.abs(X))


def h1(X):
    return 1


def g2(X, gamma):
    return np.multiply(np.abs(X) <= gamma, 1 / 2 * np.square(np.abs(X))) + np.multiply(np.abs(X) > gamma,
                                                                                       gamma * np.abs(X) - 1 / 2 * (
                                                                                               gamma ** 2))


def h2(X, gamma):
    return 1 / 2 * (np.abs(X) <= gamma) + (gamma / 2) * np.multiply(np.abs(X) > gamma, np.reciprocal(np.abs(X)))


def g3(X, gamma):
    return gamma * np.abs(X) - (gamma ** 2) * np.log(1 + (np.abs(X) / gamma))


def h3(X, gamma):
    return (gamma / 2) * np.reciprocal(np.abs(X) + gamma)


def complex_model(X, Y, alpha, g, h, tau=0.1, epsilon=1e-6):
    objective_function_values = []
    oldval = alpha * sum(sum(np.square(np.abs(X - Y)))) + (1 - alpha) * cost(X, g)
    objective_function_values.append(oldval)
    i = 1
    while True:
        grads = np.zeros((256, 256), dtype=complex)
        grads += alpha * 2 * (X - Y) + (1 - alpha) * grad(X, h)
        X -= tau * grads
        newval = alpha * sum(sum(np.square(np.abs(X - Y)))) + (1 - alpha) * cost(X, g)
        if tau < 1e-10 or abs(newval - oldval) / oldval < epsilon:
            break
        if newval < oldval:
            tau = tau*1.1
        else:
            tau = tau/2
        oldval = newval
        objective_function_values.append(oldval)
        i += 1
    return objective_function_values

I1 = np.copy(imageNoisy)
I2 = np.copy(imageNoisy)
I3 = np.copy(imageNoisy)

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(np.abs(imageNoisy), cmap='gray')
ax1.set_title("Noisy, RRMSE ")#+ '%.3f' % RRMSE(imageNoiseless, imageNoisy))
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

alphaQuad = 0.43
modelQuad = np.copy(imageNoisy)
objQuad = complex_model(modelQuad, imageNoisy, alphaQuad, g1, h1)

ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(np.abs(modelQuad), cmap='gray')
ax2.set_title("Quadratic")#, RRMSE "# + '%.3f' % RRMSE(imageNoiseless, modelQuad))
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

alphaHuber = 0.05
gammaHuber = 0.005
modelHuber = np.copy(imageNoisy)
objHuber = complex_model(modelHuber, imageNoisy, alphaHuber, lambda X: g2(X, gammaHuber), lambda X: h2(X, gammaHuber))

ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(np.abs(modelHuber), cmap='gray')
ax3.set_title("Huber")#, RRMSE " + '%.3f' % RRMSE(imageNoiseless, modelHuber))
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

alphaAdaptive = 0.09
gammaAdaptive = 0.009
modelAdaptive = np.copy(imageNoisy)
objAdaptive = complex_model(modelAdaptive, imageNoisy, alphaAdaptive, lambda X: g3(X, gammaAdaptive), lambda X: h3(X, gammaAdaptive))

ax4 = fig.add_subplot(2, 2, 4)
ax4.imshow(np.abs(modelAdaptive), cmap='gray')
ax4.set_title("Adaptive")#, RRMSE " + '%.3f' % RRMSE(imageNoiseless, modelAdaptive))
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)

plt.savefig('../results/brain.png')
plt.close()
    
#for i in np.arange(0.05, 0.1, 0.01):
#    for j in np.arange(0.005, 0.01, 0.001):
#        temp=np.copy(imageNoisy)
#        complex_model(temp, imageNoisy, i, lambda X: g3(X, j), lambda X: h3(X, j))
#        plt.imshow(np.abs(temp), cmap='gray')
#        plt.savefig(str(i)+str(j)+'.png')
#        plt.close()
        
def saveObjPlots(obj, name):
    plt.plot(obj)
    plt.savefig('../results/'+name+'ObjectivePlot.png')
    fig.clear()
    plt.close()
    
for i in [(objQuad, 'brainQuadratic'), (objHuber, 'brainHuber'), (objAdaptive, 'brainAdaptive')]:
    saveObjPlots(i[0], i[1])   
    
from scipy.io import savemat
savemat('../results/brainQuadratic', {'image': modelQuad})
savemat('../results/brainHuber', {'image': modelHuber})
savemat('../results/brainAdaptive', {'image':modelAdaptive})
    
        