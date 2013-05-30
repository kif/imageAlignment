#!/usr/bin/python
from math import sin, cos, pi
import numpy
import matplotlib
from matplotlib import pylab
import scipy
import feature

l = scipy.misc.lena().astype(numpy.float32)
#l1 = l[5:, 6:]
#l2 = l[:-5, :-6]
l1=l2=l
fig = pylab.figure()
sp1 = fig.add_subplot(1, 2, 1)
sp2 = fig.add_subplot(1, 2, 2)
sp1.imshow(l1, interpolation="nearest", cmap="gray")
sp2.imshow(l2, interpolation="nearest", cmap="gray")
siftalignement = feature.SiftAlignment()
kp1 = siftalignement.sift(l1)
print("")
kp2 = siftalignement.sift(l2)
print kp1[:, :4], kp2[:, :4]
print kp1.shape[0], kp2.shape[0]
match = siftalignement.match(kp1, kp2)

for i in range(kp1.shape[0]):
    x = kp1[i, 0]
    y = kp1[i, 1]
    scale = kp1[i, 2]
    angle = kp1[i, 3]
    if ((match[:, 1] == x) * (match[:, 0] == y)).sum() > 0:
        color = "blue"
    else:
        color = "red"
    sp1.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color=color,
                     arrowprops=dict(facecolor=color, edgecolor=color, width=1),)
for i in range(kp2.shape[0]):
    x = kp2[i, 0]
    y = kp2[i, 1]
    scale = kp2[i, 2]
    angle = kp2[i, 3]
    if ((match[:, 3] == x) * (match[:, 2] == y)).sum() > 0:
        color = "blue"
    else:
        color = "red"
    sp2.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color=color,
                     arrowprops=dict(facecolor=color, edgecolor=color, width=1),)

print match.shape[0]

fig.show()
raw_input("enter to quit")
