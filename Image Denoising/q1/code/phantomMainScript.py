import numpy as np
import h5py
import matplotlib.pyplot as plt

file = h5py.File('../data/ImageDenoisingPhantom.mat', 'r')
# print(list(file.keys()))
imageNoiseless = file['imageNoiseless']
imageNoiseless = np.array(imageNoiseless)
imageNoisy = file['imageNoisy']


def mycomplex(a):
    return a[0] + 1j * a[1]


imageNoisy = np.vectorize(mycomplex)(np.array(imageNoisy))


# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.imshow(imageNoisyR)
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.imshow(imageNoiseless)

def RRMSE(A, B):
    # Always use noiseless image as A
    Aa = np.abs(A)
    return np.sqrt(np.sum(np.sum(np.square(Aa - np.abs(B))))) / np.sqrt(np.sum(np.sum(np.square(Aa))))


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
            #print("tau:", tau)
            break
        if newval < oldval:
            tau = tau*1.1
        else:
            tau = tau/2
        oldval = newval
        objective_function_values.append(oldval)
        i += 1
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(objective_function_values)
    #print(len(objective_function_values))
    return objective_function_values


I1 = np.copy(imageNoisy)
I2 = np.copy(imageNoisy)
I3 = np.copy(imageNoisy)

fig = plt.figure()

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(np.abs(imageNoisy))
ax1.set_title("Noisy, RRMSE " + '%.3f' % RRMSE(imageNoiseless, imageNoisy))
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

# minRRMSE = RRMSE(imageNoiseless, imageNoisy)
# minAlpha = 0.72
# finalModel = np.copy(imageNoisy)
# # finding training parameters for quadratic
#print("Quadratic")
# for alpha_quadratic in [0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84]:
#     objective_function = complex_model(I1, imageNoisy, alpha_quadratic, g1, h1)
#     print(objective_function[-1])
#     temp = RRMSE(imageNoiseless, I1)
#     minRRMSE, minAlpha, finalModel = (temp, alpha_quadratic, I1) if temp < minRRMSE else (
#     minRRMSE, minAlpha, finalModel)
#     I1 = np.copy(imageNoisy)
#
# print(minAlpha, minRRMSE)
# from the above search

alphaQuad = 0.78
modelQuad = np.copy(imageNoisy)
objQuad = complex_model(modelQuad, imageNoisy, alphaQuad, g1, h1)

#print(RRMSE(imageNoiseless, modelQuad))
ax2 = fig.add_subplot(2, 3, 3)
ax2.imshow(np.abs(modelQuad))
ax2.set_title("Quadratic, RRMSE " + '%.3f' % RRMSE(imageNoiseless, modelQuad))
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
#print("Huber")

# minAlpha = 1
# minRRMSE = RRMSE(imageNoiseless, imageNoisy)
# finalModel = np.copy(imageNoisy)
# minGamma = 1

# for alpha_huber in np.arange(0.005, 0.008, 0.001):
#     for gamma_huber in np.arange(0.0005, 0.001, 0.0001):
#         objective_function = complex_model(I2,
#                                            imageNoisy,
#                                            alpha_huber,
#                                            lambda X: g2(X, gamma_huber),
#                                            lambda X: h2(X, gamma_huber))
#         print(objective_function[-1])
#         temp = RRMSE(imageNoiseless, I2)
#         minRRMSE, minAlpha, finalModel, minGamma = (temp, alpha_huber, I2, gamma_huber) if temp < minRRMSE \
#             else (minRRMSE, minAlpha, finalModel, minGamma)
#         I2 = np.copy(imageNoisy)
#
# print(minAlpha, minRRMSE, minGamma)


# from the above commented grid search
alphaHuber = 0.007
gammaHuber = 0.0009
modelHuber = np.copy(imageNoisy)
objHuber = complex_model(modelHuber, imageNoisy, alphaHuber, lambda X: g2(X, gammaHuber), lambda X: h2(X, gammaHuber))

#print(RRMSE(imageNoiseless, modelHuber))

ax3 = fig.add_subplot(2, 3, 4)
ax3.imshow(np.abs(modelHuber))
ax3.set_title("Huber, RRMSE " + '%.3f' % RRMSE(imageNoiseless, modelHuber))
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
#print("Adaptive")

# minAlpha = 1
# minRRMSE = RRMSE(imageNoiseless, imageNoisy)
# finalModel = np.copy(imageNoisy)
# minGamma = 100
# for alpha_adaptive in np.arange(0.025, 0.03, 0.001): # 0.025
#     for gamma_adaptive in np.arange(0.001, 0.005, 0.001): # 0.01, # 0.12
#         objective_function = complex_model(I3,
#                                            imageNoisy,
#                                            alpha_adaptive,
#                                            lambda X: g3(X, gamma_adaptive),
#                                            lambda X: h3(X, gamma_adaptive))
#         print(objective_function[-1])
#         temp = RRMSE(imageNoiseless, I3)
#         print(temp)
#         minRRMSE, minAlpha, finalModel, minGamma = (temp, alpha_adaptive, I3, gamma_adaptive) if temp < minRRMSE \
#             else (minRRMSE, minAlpha, finalModel, minGamma)
#         I3 = np.copy(imageNoisy)
#
# print(minAlpha, minRRMSE, minGamma)

alphaAdaptive = 0.029
gammaAdaptive = 0.004
modelAdaptive = np.copy(imageNoisy)
objAdaptive = complex_model(modelAdaptive, imageNoisy, alphaAdaptive,
                            lambda X: g3(X, gammaAdaptive),
                            lambda X: h3(X, gammaAdaptive))

#print(RRMSE(imageNoiseless, modelAdaptive))

ax4 = fig.add_subplot(2, 3, 6)
ax4.imshow(np.abs(modelAdaptive))
ax4.set_title("Adaptive, RRMSE " + '%.3f' % RRMSE(imageNoiseless, modelAdaptive))
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
# ax3 = fig.add_subplot(3, 2, 5)
# ax3.imshow(np.abs(finalModel))
# ax3.set_title("Adaptive, RRMSE " + '%.3f' % RRMSE(imageNoiseless, finalModel))


ax5 = fig.add_subplot(2, 3, 2)
ax5.imshow(np.abs(imageNoiseless))
ax5.set_title("Noiseless")
ax5.get_xaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)
plt.savefig('../results/phantom.png')
fig.clear()
plt.close()


def saveObjPlots(obj, name):
    plt.plot(obj)
    plt.savefig('../results/'+name+'ObjectivePlot.png')
    fig.clear()
    plt.close()


def optimal(alpha, cost, grad, name, gamma=-1):
    if gamma==-1:
        print(name)
        print("Optimal alpha: ", alpha)
        temp = np.copy(imageNoisy)
        complex_model(temp, imageNoisy, alpha, cost, grad)
        print("RRMSE: ", RRMSE(imageNoiseless, temp))
        print("0.8alpha: ", 0.8*alpha)
        temp = np.copy(imageNoisy)
        complex_model(temp, imageNoisy, 0.8*alpha, cost, grad)
        print("RRMSE: ", RRMSE(imageNoiseless, temp))
        if 1.2*alpha<=1:
            print("1.2alpha: ", 1.2*alpha)
            temp = np.copy(imageNoisy)
            complex_model(temp, imageNoisy, 1.2*alpha, cost, grad)
            print("RRMSE: ", RRMSE(imageNoiseless, temp))
    else:
        print(name)
        print("Optimal alpha: ", alpha)
        print("Optimal gamma: ", gamma)
        temp = np.copy(imageNoisy)
        complex_model(temp, imageNoisy, alpha, lambda X: cost(X, gamma), lambda X:grad(X, gamma))
        print("RRMSE: ", RRMSE(imageNoiseless, temp))
        print("0.8alpha, gamma: ", 0.8*alpha, gamma)
        temp = np.copy(imageNoisy)
        complex_model(temp, imageNoisy, 0.8*alpha, lambda X: cost(X, gamma), lambda X:grad(X, gamma))
        print("RRMSE: ", RRMSE(imageNoiseless, temp))
        if 1.2*alpha<=1:
            print("1.2alpha, gamma: ", 1.2*alpha, gamma)
            temp = np.copy(imageNoisy)
            complex_model(temp, imageNoisy, 1.2*alpha, lambda X: cost(X, gamma), lambda X:grad(X, gamma))
            print("RRMSE: ", RRMSE(imageNoiseless, temp))
        print("alpha, 0.8gamma: ", alpha, 0.8*gamma)
        temp = np.copy(imageNoisy)
        complex_model(temp, imageNoisy, alpha, lambda X: cost(X, 0.8*gamma), lambda X:grad(X, 0.8*gamma))
        print("RRMSE: ", RRMSE(imageNoiseless, temp))
        print("alpha, 1.2gamma: ", alpha, gamma)
        temp = np.copy(imageNoisy)
        complex_model(temp, imageNoisy, alpha, lambda X: cost(X, 1.2*gamma), lambda X:grad(X, 1.2*gamma))
        print("RRMSE: ", RRMSE(imageNoiseless, temp))
        

for i in [(objQuad, 'phatomQuadratic'), (objHuber, 'phatomHuber'), (objAdaptive, 'phantomAdaptive')]:
    saveObjPlots(i[0], i[1])    
#for i in [[0.78, g1, h1, "Quadratic"], [0.007, g2, h2, "Huber", 0.0009], [0.029, g3, h3, "Adpative", 0.004]]:
#    optimal(*i)
from scipy.io import savemat
savemat('../results/phantomQuadratic', {'image': modelQuad})
savemat('../results/phantomHuber', {'image': modelHuber})
savemat('../results/phantomAdaptive', {'image':modelAdaptive})
