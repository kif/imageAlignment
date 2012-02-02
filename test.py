#!/usr/bin/python

import feature
import scipy
import numpy
import matplotlib
from pylab import *
import time


def Visual_SURF(im1, im2):
    s0, s1 = im1.shape
    t0 = time.time()
    out = feature.surf2(im1, im2, 1)
    t1 = time.time()
    print("Image alignement (shape: %sx%s) took %.3fs. SURF found %i control points" % (s0, s1, t1 - t0, out.shape[0]))

    bigimg = numpy.zeros((s0 , s1 * 2))
    bigimg[:, :s1] = im1
    bigimg[:, s1:] = im2
    imshow(bigimg)
    arrow(s1, 0, 0, s0, width=0)
    for i in range(out.shape[0]):
        color = (cos(i) ** 2, cos(i + 1) ** 2, cos(i + 2) ** 2)
        arrow(out[i, 1], out[i, 0], out[i, 3] - out[i, 1] + s1 , out[i, 2] - out[i, 0] , width=0, color=color)
    show()
    return out

def Visual_SIFT(im1, im2):
    s0, s1 = im1.shape
    t0 = time.time()
    out = feature.sift2(im1, im2, 1)

    t1 = time.time()
    print("Image alignement (shape: %sx%s) took %.3fs. SIFT found %i control points" % (s0, s1, t1 - t0, out.shape[0]))
    bigimg = numpy.zeros((s0 , s1 * 2))
    bigimg[:, :s1] = im1
    bigimg[:, s1:] = im2
    imshow(bigimg)
    arrow(s1, 0, 0, s0, width=0)
    for i in range(out.shape[0]):
        color = (cos(i) ** 2, cos(i + 1) ** 2, cos(i + 2) ** 2)
        arrow(out[i, 1], out[i, 0], out[i, 3] - out[i, 1] + s1 , out[i, 2] - out[i, 0] , width=0, color=color)
    show()
    return out

def Visual_ASIFT(im1, im2):
    s0, s1 = im1.shape
    t0 = time.time()
    out = feature.asift2(im1, im2, 1)

    t1 = time.time()
    print("Image alignement (shape: %sx%s) took %.3fs. ASIFT found %i control points" % (s0, s1, t1 - t0, out.shape[0]))
    bigimg = numpy.zeros((s0 , s1 * 2))
    bigimg[:, :s1] = im1
    bigimg[:, s1:] = im2
    imshow(bigimg)
    arrow(s1, 0, 0, s0, width=0)
    for i in range(out.shape[0]):
        color = (cos(i) ** 2, cos(i + 1) ** 2, cos(i + 2) ** 2)
        arrow(out[i, 1], out[i, 0], out[i, 3] - out[i, 1] + s1 , out[i, 2] - out[i, 0] , width=0, color=color)
    show()
    return out

def calcShift(npa, mask=None):
    """
    Calculates the shift based on the mean of the median half 
    @param npa: numpy array of size (n,4)
    @param mask: mask with valid pixel>0
    """
    n, m = npa.shape
    assert m == 4
    v0 = out[:, 0] - out[:, 2]
    v1 = out[:, 1] - out[:, 3]
#    d = v0 ** 2 + v1 ** 2
#    s = numpy.argsort(d)
#    float(v0[s][n / 4:3 * n / 4].mean())
#    return float(v0[s][n / 4:-n / 4].mean()), float(v1[s][n / 4:-n / 4].mean())
    return scipy.median(v0), scipy.median(v1)
if __name__ == "__main__":
    #lena1 = numpy.zeros((512, 512))
    #scipy.lena()
    #lena1[100:150, 160:200] = 1
    ao1, ao2 = 5, 3
    print("Absolute offset is %s,%s" % (ao1, ao2))
    lena1 = scipy.lena()
    lena2 = numpy.zeros_like(lena1)
    lena2[5:, 3:] = lena1[:-ao1, :-ao2]
#    Visual_SURF(lena1, lena2)
    out = feature.surf2(lena1, lena2, verbose=1)
    import scipy
    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
    print "Median", scipy.median(out[:, 0] - out[:, 2]), scipy.median(out[:, 1] - out[:, 3])
    print "clacShift", calcShift(out)

    print "*" * 80
    out = feature.sift2(lena1, lena2, verbose=1)
#    out = Visual_SIFT(lena1, lena2)
    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
    print "Median", scipy.median(out[:, 0] - out[:, 2]), scipy.median(out[:, 1] - out[:, 3])
    print "clacShift", calcShift(out)

    print "*" * 80
    out = feature.asift2(lena1, lena2, verbose=0)
    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
    print "Median", scipy.median(out[:, 0] - out[:, 2]), scipy.median(out[:, 1] - out[:, 3])
    print "clacShift", calcShift(out)
