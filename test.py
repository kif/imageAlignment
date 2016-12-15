#!/usr/bin/python

import sys
import feature
import scipy
import scipy.misc
import numpy
import matplotlib
from pylab import *
import time
from itertools import combinations, permutations
import Image
import multiprocessing


def img2array(fn):
    i=Image.open(fn)
    i.load()
    j = numpy.fromstring(i.convert("F").tostring(), dtype="float32")
    j.shape = -1, i.size[0]
    return j

def siftManyImg(fn, nbcpu=None,cut=2000):
    result={}
    if not nbcpu:
        nbcpu = multiprocessing.cpu_count()
    for i in range((len(fn)-1)//nbcpu+1):
        lf=[fn[0]]+[fn[j+1] for j in range(i*nbcpu,(i+1)*nbcpu) if j<len(fn)-1]
        print lf
        ld=[img2array(j)[cut:] for j in lf]
        dr=feature.siftn(*ld,verbose=False,vs_first=True)
        for k,v in dr.items():
            result[(lf[k[0]],lf[k[1]])]=v
            print(lf[k[0]],lf[k[1]],calcShift(v))
    return result

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
    if npa is None:
        return None
    n, m = npa.shape
    assert m == 4
    v0 = npa[:, 0] - npa[:, 2]
    v1 = npa[:, 1] - npa[:, 3]
    return numpy.median(v0), numpy.median(v1)


def stitch(*img):
    if len(img) <= 1:
        return img

    d = feature.siftn(*img, verbose=True)
    t0 = time.time()
    for i in d:
        print i, "calcShift", calcShift(d[i])
    so = setOfOffsets(d)
    bigShape0 = max([i.shape[0] for i in img])
    bigShape1 = max([i.shape[1] for i in img])
    minPos = [0, 0]
    maxPos = list(img[0].shape)
    shifts = [(0, 0)]
    for i in range(1, len(img)):
        bp = so.bestPath(0, i)[-1]
        print ((0, i), bp)
        s = bp["shift"]
        shifts.append(s)
        if s[0] < minPos[0]:
            minPos[0] = s[0]
        if s[1] < minPos[1]:
            minPos[1] = s[1]
        if s[0] + img[i].shape[0] > maxPos[0]:
            maxPos[0] = s[0] + img[i].shape[0]
        if s[1] + img[i].shape[1] > maxPos[1]:
            maxPos[1] = s[1] + img[i].shape[1]
    print minPos, maxPos, shifts
    print "Alignement:", time.time() - t0
    big = numpy.zeros((maxPos[0] - minPos[0] + 1, maxPos[1] - minPos[1] + 1))
    print big.shape
    for i in range(len(img)):
        s0 = int(round(shifts[i][0] - minPos[0]))
        s1 = int(round(shifts[i][1] - minPos[1]))
        print img[i].shape
        print s0, s0 + img[i].shape[0], s1, s1 + img[i].shape[1]
        big[s0:s0 + img[i].shape[0],
            s1:s1 + img[i].shape[1]] = img[i]
    print big.shape
    print big
    return big



class setOfOffsets(object):
    def __init__(self, dico):
        """
        @param dico: key is 2-tuple of images index, value is array of offsets 
        """
        self.offsets = {}
        self.shifts = {}
        self.counts = {}
        self.idxMin = sys.maxint
        self.idxMax = 0

        for key in dico:
            val = dico[key]
            if val is None:
                continue
            rkey = (key[1], key[0])
            self.counts[key] = self.counts[rkey] = val.shape[0]
            if key[0] < self.idxMin:
                self.idxMin = key[0]
            if key[1] < self.idxMin:
                self.idxMin = key[1]
            if key[0] > self.idxMax:
                self.idxMax = key[0]
            if key[1] > self.idxMax:
                self.idxMax = key[1]
            if  val.shape[1] == 2:
                self.offsets[key] = val
                m0 = numpy.median(val[0])
                m1 = numpy.median(val[1])
                self.shifts[key] = numpy.array([m0, m1])
            elif val.shape[1] == 4:
                self.offsets[key] = val[:, :2] - val[:, 2:]
                self.shifts[key] = numpy.array(calcShift(val))
            self.shifts[rkey] = -self.shifts[key]
            self.offsets[rkey] = -self.offsets[key]
        self.pairs = [i for i in self.shifts if self.shifts is not None ]
        self.pairs.sort()


    def calcPath(self, start, stop):
        paths = {} #key tuple of all vertices, value: dict containing: tuple of edge shift, tuple of shifts, tuple of counts  
        if start == stop:
            return
        if start not in xrange(self.idxMin, self.idxMax + 1):
            print("start (%i) out of range(%i,%i)" % (start, self.idxMin, self.idxMax + 1))
            return
        if stop not in xrange(self.idxMin, self.idxMax + 1):
            print("Stop (%s) out of range(%i,%i)" % (stop, self.idxMin, self.idxMax + 1))
            return
        #direct edge is always possible:
#        if (start, stop) in self.shifts:
#            paths[(start, stop)] = {"edge":(self.shifts[(start, stop)],),
#                                    "count":(self.counts[(start, stop)],) }
        comb = []
        if self.idxMax - self.idxMin > 10:
            match = False
            maxJumps = 0
            intermediates1 = [start]
            intermediates2 = [stop]
            while True:
                maxJumps += 1

                for i in intermediates1[:]:
                    for j in range(self.idxMin, self.idxMax + 1):
                        if j in intermediates1:
                            continue
                        elif ((i, j) in self.pairs):
                            if j in intermediates2:
                                match = True
                            else:
                                intermediates1.append(j)
                for i in intermediates2[:]:
                    for j in range(self.idxMin, self.idxMax + 1):
                        if j in intermediates2:
                            continue
                        elif ((i, j) in self.pairs):
                            if j in intermediates1:
                                match = True
                            else:
                                intermediates2.append(j)
                if match:
                    break
            intermediates = intermediates1 + intermediates2
            intermediates.sort()
        else:
            maxJumps = self.idxMax - self.idxMin
            intermediates = range(self.idxMin, self.idxMax + 1)
        if len(intermediates) > 10:
            gen = combinations
        else:
            gen = permutations
        intermediates.remove(start)
        intermediates.remove(stop)


        for jumps in range(len(intermediates)):
            comb += gen(intermediates, jumps)
        for path in comb:
            ok = True
            v = (start,) + path + (stop,)
            e = [(i, j) for i, j in zip(v[:-1], v[1:])]
            s = []
            c = []
            for p in e:
                if p in self.shifts:
                    s.append(self.shifts[p])
                    c.append(self.counts[p])
                else:
                    ok = False
            if not ok: continue
            paths[v] = {"shift":s, "count":c}
        return paths

    def bestPath(self, start, stop):
        def mysort(a, b):
            if a["count"] > b["count"]:
                return 1
            elif a["count"] < b["count"]:
                return -1
            elif len(a["path"]) < len(b["path"]):
                return 1
            elif len(a["path"]) > len(b["path"]):
                return -1
            else:
                return 0
        d = []
        paths = self.calcPath(start, stop)
        if paths is not None:
            for key in paths:
                shift = numpy.zeros(2)
                count = min(paths[key]["count"])
                for npa in paths[key]["shift"]:
                    shift += npa
                d.append({"path":key, "shift": shift, "count":count})
            d.sort(mysort)
        return d

if __name__ == "__main__":
    #face1 = numpy.zeros((512, 512))
    #scipy.misc.face(gray=True)
    #face1[100:150, 160:200] = 1
    ao1, ao2 = 5, 3
    print("Absolute offset is %s,%s" % (ao1, ao2))
    face1 = scipy.misc.face(gray=True)
    face2 = numpy.zeros_like(face1)
    face2[ao1:, ao2:] = face1[:-ao1, :-ao2]
#    out = Visual_SURF(face1, face2)
    """
    out = feature.surf2(face1, face2, verbose=1)
    print "clacShift", calcShift(out)

#    raw_input("Enter to continue")
    out2 = feature.reduce_orsa(out)
#    print "SURF: %s keypoint; ORSA -> %s" % (out.shape[0], out2.shape[0])
#    out = out2
    print "*" * 80
#    out = feature.sift2(face1, face2, verbose=1)
    out = Visual_SIFT(face1, face2)
    print "clacShift", calcShift(out)
    out2 = feature.reduce_orsa(out)
    print "SIFT: %s keypoint; ORSA -> %s" % (out.shape[0], out2.shape[0])
    out = out2
    print "clacShift", calcShift(out)
    raw_input("Enter to continue")

    print "*" * 80
#    out = feature.asift2(face1, face2, verbose=0)
#    out = Visual_ASIFT(face1, face2)
#    print "Mean", (out[:, 0] - out[:, 2]).mean(), (out[:, 1] - out[:, 3]).mean()
#    print "Median", numpy.median(out[:, 0] - out[:, 2]), numpy.median(out[:, 1] - out[:, 3])
#    raw_input("Enter to continue")
#    print "clacShift", calcShift(out)
    """
    '''
    l1 = face1[:300, :300]
    l2 = face1[:300, 200:] + 10
    l3 = face1[200:, 200:] + 20
    l4 = face1[200:, :300] + 30
    d = feature.sift(l1, l2, l3, l4, verbose=True)
    for i in d:
        print i, "clacShift", calcShift(d[i])
    so = setOfOffsets(d)
    for i in range(4):
        print "#"*50
        for j in range(i + 1, 4):
            r = so.bestPath(i, j)
            print (i, j)
            for k in r:
                print k
    '''
    if len(sys.argv) < 2:
        l00 = face1[:150, :150] + 0
        l01 = face1[:150, 100:250] + 4
        l02 = face1[:150, 200:350] + 8
        l03 = face1[:150, 300:] + 12

        l10 = face1[100:250, :150] + 16
        l11 = face1[100:250, 100:250] + 20
        l12 = face1[100:250, 200:350] + 24
        l13 = face1[100:250, 300:] + 28

        l20 = face1[200:350, :150] + 32
        l21 = face1[200:350, 100:250] + 36
        l22 = face1[200:350, 200:350] + 40
        l23 = face1[200:350, 300:] + 44

        l30 = face1[300:, :150] + 48
        l31 = face1[300:, 100:250] + 52
        l32 = face1[300:, 200:350] + 56
        l33 = face1[300:, 300:] + 60

        l = (l00, l01, l02, l03, l10, l11, l12, l13, l20, l21, l22, l23, l30, l31, l32, l33)
    else:
        import Image
        I = [Image.open(i) for i in sys.argv[1:] if i.lower().endswith(".jpg") or i.lower().endswith(".tif")]
        l = []
        for i in I:
            print i
            i.load()
            j = numpy.fromstring(i.convert("F").tostring(), dtype="float32")
            j.shape = -1, i.size[0]
            l.append(j)
    imshow(stitch(*l), cmap="gray")
    show()
#    d = feature.sift(*l, verbose=True)
#    k = d.keys()
#    k.sort()
#    for i in k:
#        print i, "clacShift", calcShift(d[i])
#    so = setOfOffsets(d)
#    print "#"*50
##    for i in so.bestPath(0, 1):
##        print i
#    for i in range(1, len(l)):
#        print 0, "-->", i
#        print 0, i, so.bestPath(0, i)[-1]

    raw_input("Enter to continue")
