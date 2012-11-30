# -*- coding: utf8 -*-
#
#    Project: Image Alignment
#
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Jerome Kieffer"
__license__ = "GPLv3"
__date__ = "30/11/2012"
__copyright__ = "2011-2012, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"
__doc__ = "this is a cython wrapper for feature extraction algorithm"

import cython, time, threading, multiprocessing
from cython.operator cimport dereference as deref
from cython.parallel cimport prange
from cpython.object cimport PyObject
import numpy
cimport numpy
from libcpp cimport bool
from libcpp.pair  cimport pair
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.list cimport list
from libc.stdint cimport uint64_t, uint32_t
from threading import Semaphore
from surf cimport  image, keyPoint, descriptor, listDescriptor, getKeyPoints, listKeyPoints, listMatch, octave, interval, matchDescriptor, get_points
from sift cimport  keypoint, keypointslist, default_sift_parameters, compute_sift_keypoints, siftPar, matchingslist, compute_sift_matches, compute_sift_keypoints_flimage, flimage
from asift cimport compute_asift_matches, compute_asift_keypoints
from orsa cimport Match, MatchList, orsa
from crc32 cimport crc32
from multiprocessing import cpu_count

def mycrc(float[:] data):
    return crc32(< char *> & data[0], data.size * sizeof(float))


def normalize_image(numpy.ndarray img not None):
    maxi = numpy.float32(img.max())
    mini = numpy.float32(img.min())
    return numpy.ascontiguousarray(numpy.float32(255) * (img - mini) / (maxi - mini), dtype=numpy.float32)

cdef class SiftAlignment:
    cdef siftPar sift_parameters
    cdef map[uint32_t, keypointslist] dictKeyPointsList
    cdef FastRLock lock
    cdef object sem
#    cdef list[uint32_t] processing

    def __cinit__(self):
        default_sift_parameters(self.sift_parameters)
        self.dictKeyPointsList = map[uint32_t, keypointslist]()
        self.lock = FastRLock()
        self.sem = Semaphore(<int>cpu_count())
#        self.processing = list()
    
    def __dealloc__(self):
        self.clear()
        
    def clear(self):
        """
        Empty the vector of keypoints.
        """
        with self.lock:
            self.dictKeyPointsList.empty()
#            self.processing.empty()

    @cython.boundscheck(False)
    def sift(self, numpy.ndarray img not None):
        """
        Calculate the SIFT descriptor for an image and stores it.

        @param img: 2D numpy array representing the image
        @return: index of keypoints in the list
        """
        cdef float[:, :] data = normalize_image(img)
        cdef keypointslist kp
        cdef uint32_t idx = crc32(< char *> & data[0, 0], img.size * sizeof(float))
        cdef bool found=False
        with self.sem:
            with nogil:
                compute_sift_keypoints(< float *> & data[0, 0], kp, data.shape[1], data.shape[0], self.sift_parameters)
        with self.lock:
            self.dictKeyPointsList[idx] = kp
        return idx

    @cython.boundscheck(False)
    def match(self, data1, data2):
        """
        calculate the matching between two images already analyzed.

        @param idx1, idx2: indexes of the images in the stored
        @return:   n x 4 numpy ndarray with [y1,x1,y2,x2] control points.
        """
        cdef uint32_t idx1, idx2 
        if type(data1) == numpy.ndarray:
            idx1 = self.sift(data1)
        else:
            idx1 = data1
            
        if type(data2) == numpy.ndarray:
            idx2 = self.sift(data2)
        else:
            idx2 = data2       
        
        cdef size_t i, max_size = self.dictKeyPointsList.size()
#        if idx1 > max_size or idx2 > max_size:
#            raise IndexError("Currently %i images have been processed and you requested image %i and %i" % (max_size, idx1, idx2))
        cdef keypointslist kp1 , kp2
        with self.lock:
            kp2 = self.dictKeyPointsList[idx2]
            kp1 = self.dictKeyPointsList[idx1]
        cdef matchingslist matchings
        with nogil:
            compute_sift_matches(kp1, kp2, matchings, self.sift_parameters);
        cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((matchings.size(), 4), dtype=numpy.float32)
        for i in range(matchings.size()):
            out[i, 0] = matchings[i].first.y
            out[i, 1] = matchings[i].first.x
            out[i, 2] = matchings[i].second.y
            out[i, 3] = matchings[i].second.x
        return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def asift2(numpy.ndarray in1 not None, numpy.ndarray in2 not None, bool verbose=False):
    """
    Call ASIFT on a pair of images
    @param in1: first image
    @type in1: numpy ndarray
    @param in2: second image
    @type in2: numpy ndarray
    @param verbose: indicate the default verbosity
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    cdef int i
    cdef int num_of_tilts1 = 7
    cdef int num_of_tilts2 = 7
    cdef int verb = < int > verbose
    cdef siftPar siftparameters
    default_sift_parameters(siftparameters)
    cdef vector[ vector[ keypointslist ]] keys1
    cdef vector[ vector[ keypointslist ]] keys2
    cdef int num_keys1 = 0, num_keys2 = 0
    cdef int num_matchings
    cdef matchingslist matchings

#    cdef vector [ float ] ipixels1_zoom, ipixels2_zoom
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data1 = numpy.ascontiguousarray(255. * (in1.astype("float32") - in1.min()) / (in1.max() - in1.min()))
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data2 = numpy.ascontiguousarray(255. * (in2.astype("float32") - in2.min()) / (in2.max() - in2.min()))
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] fdata1 = data1.flatten()
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] fdata2 = data2.flatten()
    cdef vector [ float ] ipixels1_zoom = vector [ float ](< size_t > data1.size)
    cdef vector [ float ] ipixels2_zoom = vector [ float ](< size_t > data2.size)
    for i in range(data1.size):
        ipixels1_zoom[i] = < float > fdata1[i]
    for i in range(data2.size):
        ipixels2_zoom[i] = < float > fdata2[i]

    if verbose:
        import time
        print("Computing keypoints on the two images...")
        tstart = time.time()
        num_keys1 = compute_asift_keypoints(ipixels1_zoom, data1.shape[1] , data1.shape[0] , num_of_tilts1, verb, keys1, siftparameters)
        tint = time.time()
        print "ASIFT took %.3fs image1: %i ctrl points" % (tint - tstart, num_keys1)
        num_keys2 = compute_asift_keypoints(ipixels2_zoom, data2.shape[1], data2.shape[0], num_of_tilts2, verb, keys2, siftparameters)
        tend = time.time()
        print "ASIFT took %.3fs image2: %i ctrl points" % (tend - tint, num_keys2)
        tend = time.time()
        num_matchings = compute_asift_matches(num_of_tilts1, num_of_tilts2,
                                              data1.shape[1] , data1.shape[0],
                                               data2.shape[1], data2.shape[0],
                                               verb, keys1, keys2, matchings, siftparameters)
        tmatch = time.time()
        print("Matching: %s point, took %.3fs " % (num_matchings, tmatch - tend))
    else:
        num_keys1 = compute_asift_keypoints(ipixels1_zoom, data1.shape[1] , data1.shape[0] , num_of_tilts1, verb, keys1, siftparameters)
        num_keys2 = compute_asift_keypoints(ipixels2_zoom, data2.shape[1], data2.shape[0], num_of_tilts2, verb, keys2, siftparameters)
        num_matchings = compute_asift_matches(num_of_tilts1, num_of_tilts2,
                                              data1.shape[1] , data1.shape[0],
                                               data2.shape[1], data2.shape[0],
                                               verb, keys1, keys2, matchings, siftparameters)

    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((num_matchings, 4), dtype="float32")
    matchings.begin()
    for i in range(matchings.size()):
        out[i, 0] = matchings[i].first.y
        out[i, 1] = matchings[i].first.x
        out[i, 2] = matchings[i].second.y
        out[i, 3] = matchings[i].second.x
    return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_orsa(numpy.ndarray inp not None, shape=None, bool verbose=False):
    """
    Call ORSA (keypoint checking)
    @param inp: n*4 ot n*2*2 array representing keypoints.
    @type in1: numpy ndarray
    @param shape: shape of the input images (unless guessed)
    @type shape: 2-tuple of integers
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """

    cdef int i, num_matchings, insize, p
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data = numpy.ascontiguousarray(inp.astype("float32").reshape(-1, 4))
    insize = data.shape[0]
    if insize < 10:
        return data
    cdef vector [ Match ]  match_coor = vector [ Match ](< size_t > insize)
    cdef int t_value_orsa = 10000
    cdef int verb_value_orsa = verbose
    cdef int n_flag_value_orsa = 0
    cdef int mode_value_orsa = 2
    cdef int stop_value_orsa = 0
    cdef float nfa
    cdef int width, heigh
    if shape is None:
        width = int(1 + max(data[:, 1].max(), data[:, 3].max()))
        heigh = int(1 + max(data[:, 0].max(), data[:, 2].max()))
    elif hasattr(shape, "__len__") and len(shape) >= 2:
        width = int(shape[1])
        heigh = int(shape[0])
    else:
        width = heigh = int(shape)
    cdef vector [ float ] index = vector [ float ](< size_t > data.shape[0])
    tmatch = time.time()
    with nogil:
        for i in range(data.shape[0]):
            match_coor[i].y1 = < float > data[i, 0]
            match_coor[i].x1 = < float > data[i, 1]
            match_coor[i].y2 = < float > data[i, 2]
            match_coor[i].x2 = < float > data[i, 3]
    # epipolar filtering with the Moisan - Stival ORSA algorithm.
        nfa = orsa(width, heigh, match_coor, index, t_value_orsa, verb_value_orsa, n_flag_value_orsa, mode_value_orsa, stop_value_orsa)
    tend = time.time()
    num_matchings = index.size()
    if verbose:
        print("Matching with ORSA: %s => %s, took %.3fs, nfs=%s" % (insize, num_matchings, tend - tmatch, nfa))
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((num_matchings, 4), dtype="float32")
    for i in range(index.size()):
        p = < int > index[i]
        out[i, 0] = data[p, 0]
        out[i, 1] = data[p, 1]
        out[i, 2] = data[p, 2]
        out[i, 3] = data[p, 3]
    return out

cdef void printCtrlPointSift(keypointslist kpt, int maxLines=10):
    """
    Print the control points
    """
    cdef int i
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((kpt.size(), 4), dtype="float32")
    for i in range(kpt.size()):
        out[i, 0] = kpt[i].x
        out[i, 1] = kpt[i].y
        out[i, 2] = kpt[i].scale
        out[i, 3] = kpt[i].angle
    out.sort(axis=0)
    for i in range(min(maxLines, kpt.size())):
        print out[i]

cdef void printMatching(matchingslist match, int maxLines=10):
    """
    Print the matching control points
    """
    cdef int i
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((match.size(), 4), dtype="float32")
    for i in range(match.size()):
        out[i, 0] = match[i].first.x
        out[i, 1] = match[i].first.y
        out[i, 2] = match[i].second.x
        out[i, 3] = match[i].second.y
    out.sort(axis=0)
    for i in range(min(maxLines, match.size())):
        print out[i]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def surf2(numpy.ndarray in1 not None, numpy.ndarray in2 not None, bool verbose=False):
    """
    Call surf on a pair of images
    @param in1: first image
    @type in1: numpy ndarray
    @param in2: second image
    @type in2: numpy ndarray
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    cdef listKeyPoints * l1 = new listKeyPoints()
    cdef listKeyPoints * l2 = new listKeyPoints()
    cdef listDescriptor * listeDesc1
    cdef listDescriptor * listeDesc2
    cdef listMatch * matching

    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data1 = numpy.ascontiguousarray(255. * (in1.astype("float32") - in1.min()) / (in1.max() - in1.min()))
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data2 = numpy.ascontiguousarray(255. * (in2.astype("float32") - in2.min()) / (in2.max() - in2.min()))
    cdef image * img1 = new image(data1.shape[1], data1.shape[0])
    img1.img = < float *> data1.data
    cdef image * img2 = new image(data2.shape[1], data2.shape[0])
    img2.img = < float *> data2.data

    if verbose:
        import time
        time_init = time.time()
        listeDesc1 = getKeyPoints(img1, octave, interval, l1, verbose)
        time_int = time.time()
        print "SURF took %.3fs image1: %i ctrl points" % (time_int - time_init, listeDesc1.size())
        time_int = time.time()
        listeDesc2 = getKeyPoints(img2, octave, interval, l2, verbose)
        time_finish = time.time()
        print "SURF took %.3fs image2: %i ctrl points" % (time_finish - time_int, listeDesc2.size())
        time_finish = time.time()
        matching = matchDescriptor(listeDesc1, listeDesc2)
        time_matching = time.time()
        print("Matching %s point, took %.3fs " % (matching.size(), time_matching - time_finish))
    else:
        with nogil:
            listeDesc1 = getKeyPoints(img1, octave, interval, l1, verbose)
            listeDesc2 = getKeyPoints(img2, octave, interval, l2, verbose)
            matching = matchDescriptor(listeDesc1, listeDesc2)

    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((matching.size(), 4), dtype="float32")
    get_points(matching, < float *> (out.data))
    del matching, l1, l2, listeDesc1, listeDesc2
    return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sift2(img1, img2, bool verbose=False):
    """
    Call SIFT on a pair of images
    @param in1: first image
    @type in1: numpy ndarray
    @param in2: second image
    @type in2: numpy ndarray
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    cdef size_t i
    cdef float[:, :] data1 = normalize_image(img1)
    cdef float[:, :] data2 = normalize_image(img2)
    cdef keypointslist k1, k2
    cdef siftPar para
    cdef matchingslist matchings
    default_sift_parameters(para)
    if verbose:
        import time
        t0 = time.time()
        compute_sift_keypoints(< float *> & data1[0, 0], k1, data1.shape[1], data1.shape[0], para);
        t1 = time.time()
        print "SIFT took %.3fs image1: %i ctrl points" % (t1 - t0, k1.size())
        t1 = time.time()
        compute_sift_keypoints(< float *> & data2[0, 0], k2, data2.shape[1], data2.shape[0], para);
        t2 = time.time()
        print "SIFT took %.3fs image2: %i ctrl points" % (t2 - t1, k2.size())
        t2 = time.time()
        compute_sift_matches(k1, k2, matchings, para);
        print("Matching: %s point, took %.3fs " % (matchings.size(), time.time() - t2))
    else:
        with nogil:
            compute_sift_keypoints(< float *> & data1[0, 0], k1, data1.shape[1], data1.shape[0], para);
            compute_sift_keypoints(< float *> & data2[0, 0], k2, data2.shape[1], data2.shape[0], para);
            compute_sift_matches(k1, k2, matchings, para);

    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((matchings.size(), 4), dtype="float32")
    for i in range(matchings.size()):
        out[i, 0] = matchings[i].first.y
        out[i, 1] = matchings[i].first.x
        out[i, 2] = matchings[i].second.y
        out[i, 3] = matchings[i].second.x
    return out

def pos(int n, int k, bool vs_first=False):
    """get postion i,j from index k in an upper-filled square array
    [ 0 0 1 2 3 ]
    [ 0 0 4 5 6 ]
    [ 0 0 0 7 8 ]
    [ 0 0 0 0 9 ]
    [ 0 0 0 0 0 ]

    pos(5,9): (3, 4)
    pos(5,8): (2, 4)
    pos(5,7): (2, 3)
    pos(5,6): (1, 4)
    pos(5,5): (1, 3)
    pos(5,4): (1, 2)
    pos(5,3): (0, 4)
    pos(5,2): (0, 3)
    pos(5,1): (0, 2)
    pos(5,0): (0, 1)

    """
    if vs_first:
        return 0, k + 1
    cdef int i, j
    for i in range(n):
        if k < (n - i - 1):
            j = i + 1 + k
            break
        else:
            k = k - (n - i - 1)
    return i, j

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def siftn(*listArg, bool verbose=False, bool vs_first=False):
    """
    Call SIFT on a pair of images
    @param *listArg: images
    @type *listArg: numpy ndarray
    @param verbose: print informations when finished
    @type verbose: boolean
    @param vs_first: calculate sift always vs first img
    @type vs_first: boolean
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    t0 = time.time()
    cdef int i, j, k, n, m, p, t
    cdef vector[flimage] lstInput
    cdef vector [keypointslist] lstKeypointslist
    cdef vector[matchingslist] lstMatchinglist
    cdef vector[MatchList] lstMatchlist
    cdef vector[float] tmpIdx
    cdef vector [vector[float]] lstIndex
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] tmpNPA
    cdef siftPar para
    cdef int t_value_orsa = 10000
    cdef int verb_value_orsa = 0
    cdef int n_flag_value_orsa = 0
    cdef int mode_value_orsa = 2
    cdef int stop_value_orsa = 0
    cdef float nfa
    cdef MatchList tmpMatchlist
    cdef Match tmpMatch

    default_sift_parameters(para)
    for obj in listArg:
        if isinstance(obj, numpy.ndarray):
            tmpNPA = numpy.ascontiguousarray((255. * (obj.astype("float32") - obj.min()) / < float > (obj.max() - obj.min())).flatten())
            lstInput.push_back(flimage(< int > obj.shape[1], < int > obj.shape[0], < float *> tmpNPA.data))
            lstKeypointslist.push_back(keypointslist())
    n = lstInput.size()
    if vs_first:
        m = n - 1
    else:
        m = n * (n - 1) / 2
    for k in range(m):
        lstMatchinglist.push_back(matchingslist())
        lstMatchlist.push_back(tmpMatchlist)
        lstIndex.push_back(tmpIdx)

    t1 = time.time()
    with nogil:
        for i in prange(n):
            compute_sift_keypoints_flimage(lstInput[i], lstKeypointslist[i], para)
    t2 = time.time()
    with nogil:
        for k in prange(m):
            #Calculate indexes
            if vs_first:
                i = 0
                j = 1 + k
            else:
                t = k
                for i in range(n):
                    if t < (n - i - 1):
                        j = i + 1 + t
                        break
                    else:
                        t = t - (n - i - 1)
            #i,j = pos(n,k)
            compute_sift_matches(lstKeypointslist[i], lstKeypointslist[j], lstMatchinglist[k], para)
    t3 = time.time()
    #with nogil:
    for k in range(m):
            for p in range(< int > lstMatchinglist[k].size()):
                tmpMatch = Match(x1=lstMatchinglist[k][p].first.x,
                                 y1=lstMatchinglist[k][p].first.y,
                                 x2=lstMatchinglist[k][p].second.x,
                                 y2=lstMatchinglist[k][p].second.y)
                lstMatchlist[k].push_back(tmpMatch)
    t4 = time.time()
    with nogil:
        for k in prange(m):
            if (< int > lstMatchinglist[k].size()) > (< int > 20):
                nfa = orsa((lstInput[i].nwidth() + lstInput[j].nwidth()) / 2, (lstInput[i].nheight() + lstInput[j].nheight()) / 2,
                                lstMatchlist[k], lstIndex[k],
                                t_value_orsa, verb_value_orsa, n_flag_value_orsa, mode_value_orsa, stop_value_orsa)
    t5 = time.time()
    out = {}
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] outArray
    for k in range(m):
        tmpMatchlist = lstMatchlist[k]
        if tmpMatchlist.size() == 0:
            out[pos(n, k, vs_first)] = None
        elif tmpMatchlist.size() <= 20:
            outArray = numpy.zeros((tmpMatchlist.size(), 4), dtype="float32")
            for p in range(tmpMatchlist.size()):
                outArray[p, 0] = tmpMatchlist[p].y1
                outArray[p, 1] = tmpMatchlist[p].x1
                outArray[p, 2] = tmpMatchlist[p].y2
                outArray[p, 3] = tmpMatchlist[p].x2
            out[pos(n, k, vs_first)] = outArray
        else:
            outArray = numpy.zeros((lstIndex[k].size(), 4), dtype="float32")
            for p in range(lstIndex[k].size()):
                i = < int > lstIndex[k][p]
                outArray[p, 0] = tmpMatchlist[i].y1
                outArray[p, 1] = tmpMatchlist[i].x1
                outArray[p, 2] = tmpMatchlist[i].y2
                outArray[p, 3] = tmpMatchlist[i].x2
            out[pos(n, k, vs_first)] = outArray
    t6 = time.time()
    print verbose
    if verbose:
        print("Serial setup for SIFT took %.3fs for %i images" % ((t1 - t0), n))
        print("Parallel SIFT took %.3fs for %i images" % ((t2 - t1), n))
        print("Parallel Matching took %.3fs for %i pairs of images" % ((t3 - t2), m))
        print("Serial copy took %.3fs for %i pairs of images" % ((t4 - t3), m))
        print("Parallel ORSA took %.3fs for %i pairs of images" % ((t5 - t4), m))
        print("Serial build of numpy arrays took %.3fs for %i pairs of images" % ((t6 - t5), m))
        for k in range(m):
            print("point %i images pair: %s found %i ctrl pt -> %i" % (k, pos(n, k), lstMatchinglist[k].size(), lstIndex[k].size()))

    return out
#Cython implementaation of fast locking
# author: Stephan Behnel additions from Jerome Kieffer
# http://code.activestate.com/recipes/577336/
# C++ version 
from cpython cimport pythread
from cpython.exc cimport PyErr_NoMemory
from libcpp.list cimport list
from time import time as _time
from time import sleep as _sleep

cdef class FastRLock:
    """Fast, re-entrant locking.

    Under uncongested conditions, the lock is never acquired but only
    counted.  Only when a second thread comes in and notices that the
    lock is needed, it acquires the lock and notifies the first thread
    to release it when it's done.  This is all made possible by the
    wonderful GIL.
    """
    cdef pythread.PyThread_type_lock _real_lock
    cdef long _owner            # ID of thread owning the lock
    cdef int _count             # re-entry count
    cdef int _pending_requests  # number of pending requests for real lock
    cdef bint _is_locked        # whether the real lock is acquired

    def __cinit__(self):
        self._owner = -1
        self._count = 0
        self._is_locked = False
        self._pending_requests = 0
        self._real_lock = pythread.PyThread_allocate_lock()
        if self._real_lock is NULL:
            PyErr_NoMemory()

    def __dealloc__(self):
        if self._real_lock is not NULL:
            pythread.PyThread_free_lock(self._real_lock)
            self._real_lock = NULL

    def acquire(self, bint blocking=True):
        return lock_lock(self, pythread.PyThread_get_thread_ident(), blocking)

    def release(self):
        if self._owner != pythread.PyThread_get_thread_ident():
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    # compatibility with threading.RLock

    def __enter__(self):
        # self.acquire()
        return lock_lock(self, pythread.PyThread_get_thread_ident(), True)

    def __exit__(self, t, v, tb):
        # self.release()
        if self._owner != pythread.PyThread_get_thread_ident():
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    def _is_owned(self):
        return self._owner == pythread.PyThread_get_thread_ident()


cdef inline bint lock_lock(FastRLock lock, long current_thread, bint blocking) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if lock._count:
        # locked! - by myself?
        if current_thread == lock._owner:
            lock._count += 1
            return 1
    elif not lock._pending_requests:
        # not locked, not requested - go!
        lock._owner = current_thread
        lock._count = 1
        return 1
    # need to get the real lock
    return _acquire_lock(
        lock, current_thread,
        pythread.WAIT_LOCK if blocking else pythread.NOWAIT_LOCK)

cdef bint _acquire_lock(FastRLock lock, long current_thread, int wait) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if not lock._is_locked and not lock._pending_requests:
        # someone owns it but didn't acquire the real lock - do that
        # now and tell the owner to release it when done. Note that we
        # do not release the GIL here as we must absolutely be the one
        # who acquires the lock now.
        if not pythread.PyThread_acquire_lock(lock._real_lock, wait):
            return 0
        #assert not lock._is_locked
        lock._is_locked = True
    lock._pending_requests += 1
    with nogil:
        # wait for the lock owning thread to release it
        locked = pythread.PyThread_acquire_lock(lock._real_lock, wait)
    lock._pending_requests -= 1
    #assert not lock._is_locked
    #assert lock._count == 0
    if not locked:
        return 0
    lock._is_locked = True
    lock._owner = current_thread
    lock._count = 1
    return 1

cdef inline void unlock_lock(FastRLock lock) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    #assert lock._owner == pythread.PyThread_get_thread_ident()
    #assert lock._count > 0
    lock._count -= 1
    if lock._count == 0:
        lock._owner = -1
        if lock._is_locked:
            pythread.PyThread_release_lock(lock._real_lock)
            lock._is_locked = False

#cdef class Condition:
#    cdef FastRLock _lock
#    cdef list[FastRLock] _waiters
#    def __cinit__(self, lock=None):
#        if lock is None:
#            lock = FastRLock()
#        self._lock = lock
#        self._waiters = list[FastRLock]()
#    
#    def __dealloc__(self):
#        self._waiters.empty()
#        
#    def acquire(self, bint blocking=True):
#        self._lock.acquire(blocking)
#        
#    def release(self):
#        self._lock.release()
#
#    def __enter__(self):
#        return self._lock.__enter__()
#
#    def __exit__(self, *args):
#        return self._lock.__exit__(*args)
#
#    def __repr__(self):
#        return "<Condition(%s, %d)>" % (self._lock, self._waiters.size())
#
#    def _release_save(self):
#        return self._lock.release()           # No state to save
#
#    def _acquire_restore(self, x):
#        return self._lock.acquire()           # Ignore saved state
#
#    def _is_owned(self):
#        return self._lock._is_owned()
#
#    def wait(self, timeout=None):
#        
#        cdef double delay,endtime,remaining,ctimeout
#        if timeout is None:
#            ctimeout = 0
#        else:
#            ctimeout = <double>timeout
#        if not self._is_owned():
#            raise RuntimeError("cannot wait on un-acquired lock")
#        cdef FastRLock waiter = FastRLock()
#        waiter.acquire()
#        self._waiters.push_back(waiter)
#        saved_state = self._release_save()
#        try:    # restore state no matter what (e.g., KeyboardInterrupt)
#            if ctimeout == 0:
#                waiter.acquire()
#            else:
#                # Balancing act:  We can't afford a pure busy loop, so we
#                # have to sleep; but if we sleep the whole timeout time,
#                # we'll be unresponsive.  The scheme here sleeps very
#                # little at first, longer as time goes on, but never longer
#                # than 20 times per second (or the timeout time remaining).
#                endtime = _time() + ctimeout
#                delay = 0.0005 # 500 us -> initial delay of 1 ms
#                while True:
#                    gotit = waiter.acquire(0)
#                    if gotit:
#                        break
#                    remaining = endtime - _time()
#                    if remaining <= 0:
#                        break
#                    delay = min(delay * 2., remaining, .05)
#                    _sleep(delay)
#                if not gotit:
#                    print("%s.wait(%s): timed out", self, ctimeout)
#                    try:
#                        self._waiters.remove(waiter)
#                    except ValueError:
#                        pass
#        finally:
#            self._acquire_restore(saved_state)
#            
#    def notify(self, n=1):
#        if not self._is_owned():
#            raise RuntimeError("cannot notify on un-acquired lock")
#        self._waiters
#        if self.waiters.size()==0:
#            return
#        for i in range(self.waiters.size()):
#            if i>=n:
#                break
#            waiter=self.waiters[i]
#            waiter.release()
#            try:
#                self.waiters.remove(waiter)
#            except ValueError:
#                pass
#
#    def notify_all(self):
#        self.notify(self._waiters.size())
#
#    notifyAll = notify_all
#
#          
#cdef class Semaphore:
#    cdef int _value
#    cdef Condition _cond
#    def __cinit__(self, int value=1):
#        if value < 0:
#            raise ValueError("semaphore initial value must be >= 0")
#        self._cond = Condition(FastRLock())
#        self._value = value
#
#    def acquire(self, blocking=True):
#        rc = False
#        self._cond.acquire()
#        while self._value == 0:
#            if not blocking:
#                break
#            self._cond.wait()
#        else:
#            self._value = self._value - 1
#            rc = True
#        self._cond.release()
#        return rc
#
#    __enter__ = acquire
#
#    def release(self):
#        self._cond.acquire()
#        self._value = self._value + 1
#        self._cond.notify()
#        self._cond.release()
#
#    def __exit__(self, t, v, tb):
#        self.release()
#

