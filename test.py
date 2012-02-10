#!/usr/bin/python

import feature
import scipy
import numpy
import matplotlib
from pylab import *
import time

def Visual(im1, im2, ctrlPt):
    s00, s01 = im1.shape
    s10, s11 = im2.shape

    bigimg = numpy.zeros((max(s00, s10) , s01 + s11))
    bigimg[:s00, :s01] = im1
    bigimg[:s10, s01:] = im2
    imshow(bigimg)
    arrow(s01, 0, 0, max(s00, s10), width=0)
    for i in range(ctrlPt.shape[0]):
        color = (cos(i) ** 2, cos(i + 1) ** 2, cos(i + 2) ** 2)
        arrow(ctrlPt[i, 1], ctrlPt[i, 0], ctrlPt[i, 3] - ctrlPt[i, 1] + s01 , ctrlPt[i, 2] - ctrlPt[i, 0] , width=0, color=color)
    show()


def Visual_SURF(im1, im2):
    t0 = time.time()
    out1 = feature.surf2(im1, im2, 1)
    if out1.shape[0] < 15:
        out = out1
        print("Image alignment (shapes: %s,%s) took %.3fs. SURF found %i control points" %
              (im1.shape, im2.shape, time.time() - t0, out1.shape[0],))
    else:
        out = feature.reduce_orsa(out1)
        print("Image alignment (shapes: %s,%s) took %.3fs. SURF found %i control points; Reduced to %i with ORSA" %
               (im1.shape, im2.shape, time.time() - t0, out1.shape[0], out.shape[0]))
    Visual(im1, im2, out)
    return out

def Visual_SIFT(im1, im2):
    t0 = time.time()
    out1 = feature.sift2(im1, im2, 1)
    if out1.shape[0] < 15:
        out = out1
        print("Image alignment (shapes: %s,%s) took %.3fs. SIFT found %i control points" %
              (im1.shape, im2.shape, time.time() - t0, out1.shape[0],))
    else:
        out = feature.reduce_orsa(out1)
        print("Image alignment (shapes: %s,%s) took %.3fs. SIFT found %i control points; Reduced to %i with ORSA" %
               (im1.shape, im2.shape, time.time() - t0, out1.shape[0], out.shape[0]))
    Visual(im1, im2, out)
    return out

def Visual_ASIFT(im1, im2):
    t0 = time.time()
    out1 = feature.asift2(im1, im2, 1)
    if out1.shape[0] < 15:
        out = out1
        print("Image alignment (shapes: %s,%s) took %.3fs. ASIFT found %i control points" %
              (im1.shape, im2.shape, time.time() - t0, out1.shape[0],))
    else:
        out = feature.reduce_orsa(out1)
        print("Image alignment (shapes: %s,%s) took %.3fs. ASIFT found %i control points; Reduced to %i with ORSA" %
               (im1.shape, im2.shape, time.time() - t0, out1.shape[0], out.shape[0]))
    Visual(im1, im2, out)
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
    out = Visual_SURF(lena1, lena2)
#    out = feature.surf2(lena1, lena2, verbose=1)
    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
    print "Median", scipy.median(out[:, 0] - out[:, 2]), scipy.median(out[:, 1] - out[:, 3])
    raw_input("Enter to continue")
#    out2 = feature.reduce_orsa(out)
#    print "SURF: %s keypoint; ORSA -> %s" % (out.shape[0], out2.shape[0])
#    out = out2
#    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
#    print "Median", scipy.median(out[:, 0] - out[:, 2]), scipy.median(out[:, 1] - out[:, 3])

    print "*" * 80
    #out = feature.sift2(lena1, lena2, verbose=1)
    out = Visual_SIFT(lena1, lena2)
    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
    print "Median", scipy.median(out[:, 0] - out[:, 2]), scipy.median(out[:, 1] - out[:, 3])
    raw_input("Enter to continue")
#    print "clacShift", calcShift(out)
#    out2 = feature.reduce_orsa(out)
#    print "SIFT: %s keypoint; ORSA -> %s" % (out.shape[0], out2.shape[0])
#    out = out2
#    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
#    print "Median", scipy.median(out[:, 0] - out[:, 2]), scipy.median(out[:, 1] - out[:, 3])
#    print "clacShift", calcShift(out)

    sift(lena1, lena2, verbose=1)
#    print "*" * 80
##    out = feature.asift2(lena1, lena2, verbose=0)
#    out = Visual_ASIFT(lena1, lena2)
#    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
#    print "Median", scipy.median(out[:, 0] - out[:, 2]), scipy.median(out[:, 1] - out[:, 3])
#    raw_input("Enter to continue")
#    print "clacShift", calcShift(out)
