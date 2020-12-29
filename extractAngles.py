import cv2 as cv
import pywt
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as st
import time
# from scipy.misc import imresize
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel/kernel.max()

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def getAnglesFromKeypoints(image,keypoints,patchSize=6,angleWindowInDegrees=60):
    angleWindow = angleWindowInDegrees*math.pi/180
    pi2 = math.pi*2
    #image = cv.imread('/home/jbrogan4/Documents/Projects/Medifor/tensorflow/models/research/delf/delf/python/examples/snowwhite.jpg')
    if len(image.shape) == 3:
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    #surf = cv.xfeatures2d.SURF_create()
    #keypoints= surf.detect(image)
    coeffs2 = pywt.dwt2(image, 'bior1.3')
    horiz = coeffs2[1][0]
    vert = coeffs2[1][1]
    wavescale = image.shape[0]/horiz.shape[0]
    kernelDict = {}
    j = 0
    for kp in keypoints:
        scale = kp.size
        pt = kp.pt
        x1 = int((pt[0]-patchSize*scale/2)/wavescale)
        y1 = int((pt[1]-patchSize*scale/2)/wavescale)
        x2 = int((pt[0] + patchSize * scale / 2)/wavescale)
        y2 = int((pt[1] + patchSize * scale / 2)/wavescale)
        if not x2-x1 == y2-y1:
            #go with minimum size
            d = min(x2-x1,y2-y1)
            x2 = x1+d
            y2 = y1+d
        else:
            d = x2-x1
        cropLeft = x1 < 0
        cropRight = x2 >= horiz.shape[1]
        cropTop = y1 < 0
        cropBottom = y2 >= horiz.shape[0]
        hvals = horiz[max(0,y1):y2,max(0,x1):x2]
        vvals = vert[max(0,y1):y2,max(0,x1):x2]
        if d not in kernelDict:
            kernel = gkern(d,2.5)
            kernelDict[d] = kernel
        else:
            kernel = kernelDict[d].copy()
        if cropLeft: kernel = kernel[:,-hvals.shape[1]:]
        if cropRight: kernel = kernel[:,:hvals.shape[1]]
        if cropTop: kernel = kernel[-hvals.shape[0]:,:]
        if cropBottom: kernel = kernel[:hvals.shape[0],:]
        #if hvals.shape[0] == 53 and hvals.shape[1] == 37:
        #    print('bad')
        hvals = (hvals*kernel).flatten()
        vvals = (vvals * kernel).flatten()
        mags,angles = cv.cartToPolar(hvals,vvals)
        mags = mags.flatten()
        angles = angles.flatten()
        sortedAngleIndexes = np.argsort(angles)
        sortedAngles = angles[sortedAngleIndexes]
        sortedHvals = hvals[sortedAngleIndexes]
        sortedVvals = vvals[sortedAngleIndexes]
        bestAngle = -1
        maxMag = 0
        j+=1
        sortedUniqueAngles,sortedUniqueAngleIndexes = np.unique(sortedAngles,return_index=True)
        endIndex = 0
        maxAngle = 0
        for i in range(0,6):
            startIndex = endIndex
            minAngle = maxAngle
            maxAngle = ((i+1)*60*pi2/360)%pi2
            around = minAngle > maxAngle #We've gone fully around
            endIndex = np.searchsorted(sortedUniqueAngles, maxAngle)
            if not around:
                indexList = sortedUniqueAngleIndexes[startIndex:endIndex]
            else:
                indexList = np.hstack((sortedUniqueAngleIndexes[startIndex:],sortedUniqueAngleIndexes[:endIndex]))
            windowHsum = np.sum(sortedHvals[indexList])
            windowVsum = np.sum(sortedVvals[indexList])
            totalMag,totalAngle = cart2pol(windowHsum,windowVsum)
            if totalAngle < 0: totalAngle+=pi2
            # plt.clf()
            # plt.scatter(sortedHvals,sortedVvals)
            # plt.scatter(sortedHvals[indexList],sortedVvals[indexList],color='r')
            if totalMag > maxMag:
                maxMag = totalMag
                bestAngle = totalAngle
        # plt.show()
        kp.angle = bestAngle*360/pi2
    return keypoints
        # print(bestAngle)




#getAnglesFromKeypoints(None,None)
