#!/usr/bin/python
from math import sin, cos, pi
import numpy
import matplotlib
from matplotlib import pylab
import scipy
import feature

l = scipy.misc.lena().astype(numpy.float32)
fig = pylab.figure()
sp = fig.add_subplot(1, 1, 1)
sp.imshow(l, interpolation="nearest", cmap="gray")
siftalignement = feature.SiftAlignment()
kp = siftalignement.sift(l)
print kp[:, :4]
print kp[:, :4].max(axis=0)
print kp.shape[0]

for i in range(kp.shape[0]):
    x = kp[i, 0]
    y = kp[i, 1]
    scale = kp[i, 2]
    angle = kp[i, 3]
    sp.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="blue",
                     arrowprops=dict(facecolor='blue', edgecolor='blue', width=1),)
fig.show()
raw_input("enter to quit")
