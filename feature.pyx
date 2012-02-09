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
__date__ = "01/02/2012"
__copyright__ = "2011-2012, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"
__doc__ = "this is a cython wrapper for feature extraction algorithm"

import cython, time
cimport numpy
import numpy
from libcpp cimport bool
from libcpp.pair  cimport pair
from libcpp.vector cimport vector

cdef extern from "surf/image.h":
    cdef cppclass image:
        image(int , int)
        int idImage
        int width, height
        float * img

cdef extern from "surf/keypoint.h":
    cdef cppclass keyPoint:
#        keyPoint() #
        keypoint(float, float, float, float, bool)
        float    x, y, scale, orientation
        bool     signLaplacian
#    typedef  std::vector < keyPoint *> listKeyPoints
    ctypedef vector[ keyPoint *] listKeyPoints  

cdef extern from "surf/descriptor.h":
    cdef cppclass descriptor:
        descriptor()
        descriptor(descriptor *)
        keyPoint * kP
    #std::vector<descriptor*> listDescriptor;
    ctypedef  vector[ descriptor * ] listDescriptor
    listDescriptor * getKeyPoints(image * , int , int , listKeyPoints * , bool)nogil

cdef extern from "surf/match.h":
    #typedef std::pair<descriptor*,descriptor*> pairMatch;
    ctypedef  pair[ descriptor  , descriptor  ] pairMatch
    #typedef std::vector<pairMatch > listMatch;
    ctypedef  vector[ pairMatch ] listMatch
    listMatch * matchDescriptor(listDescriptor * , listDescriptor *)nogil

cdef extern from "surf/lib.h":
    int octave
    int interval

cdef extern from "surf/surf_match.h":
    void get_points(listMatch * , float *)nogil

cdef extern from "sift/demo_lib_sift.h":
    struct keypoint:
        float   x
        float   y
        float   scale
        float   angle
        float * vec
    #typedef std::vector<keypoint> keypointslist;    
    ctypedef  vector[ keypoint ] keypointslist
    struct siftPar:
        int OctaveMax
        int DoubleImSize
        int order
        float  InitSigma
        int BorderDist
        int Scales
        float PeakThresh
        float  EdgeThresh
        float  EdgeThresh1
        int OriBins
        float OriSigma
        float OriHistThresh
        float  MaxIndexVal
        int  MagFactor
        float   IndexSigma
        int  IgnoreGradSign
        float MatchRatio
        float MatchXradius
        float MatchYradius
        int noncorrectlylocalized
    void default_sift_parameters(siftPar par)
    void compute_sift_keypoints(float * input, keypointslist  keypoints, int width, int height, siftPar par) nogil
    #typedef std::pair<keypoint,keypoint> matching;
    ctypedef  pair[ keypoint  , keypoint  ] matching
    #typedef std::vector<matching> matchingslist;
    ctypedef  vector[ matching ] matchingslist
    void compute_sift_matches(keypointslist keys1, keypointslist keys2, matchingslist matchings, siftPar par) nogil


cdef extern from "asift/compute_asift_keypoints.h":
    int compute_asift_keypoints(vector[float] image, int width, int height, int num_of_tilts, int verb, vector[ vector[ keypointslist ]] & keys_all, siftPar siftparameters) nogil
cdef extern from "asift/compute_asift_matches.h":
    int compute_asift_matches(int num_of_tilts1, int num_of_tilts2, int w1, int h1, int w2, int h2, int verb, vector[vector[keypointslist]] keys1, vector[vector[keypointslist]] keys2, matchingslist matchings, siftPar siftparameters) nogil

cdef extern from "libMatch/match.h":
    struct Match:
        float x1, y1, x2, y2

cdef extern from "orsa/orsa.h":
    float orsa(int width, int height, vector[Match] match, vector[float] index, int t_value, int verb_value, int n_flag_value, int mode_value, int stop_value)nogil



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


def sift2(numpy.ndarray in1 not None, numpy.ndarray in2 not None, bool verbose=False):
    """
    Call SIFT on a pair of images
    @param in1: first image 
    @type in1: numpy ndarray
    @param in2: second image 
    @type in2: numpy ndarray
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    cdef int i
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data1 = numpy.ascontiguousarray(255. * (in1.astype("float32") - in1.min()) / (in1.max() - in1.min()))
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data2 = numpy.ascontiguousarray(255. * (in2.astype("float32") - in2.min()) / (in2.max() - in2.min()))
    cdef keypointslist k1, k2
    cdef siftPar para
    cdef matchingslist matchings
    default_sift_parameters(para)
    if verbose:
        import time
        t0 = time.time()
        compute_sift_keypoints(< float *> data1.data, k1, data1.shape[1], data1.shape[0], para);
        t1 = time.time()
        print "SIFT took %.3fs image1: %i ctrl points" % (t1 - t0, k1.size())
        t1 = time.time()
        compute_sift_keypoints(< float *> data2.data, k2, data2.shape[1], data2.shape[0], para);
        t2 = time.time()
        print "SIFT took %.3fs image2: %i ctrl points" % (t2 - t1, k2.size())
        t2 = time.time()
        compute_sift_matches(k1, k2, matchings, para);
        print("Matching: %s point, took %.3fs " % (matchings.size(), time.time() - t2))
    else:
        with nogil:
            compute_sift_keypoints(< float *> data1.data, k1, data1.shape[1], data1.shape[0], para);
            compute_sift_keypoints(< float *> data2.data, k2, data2.shape[1], data2.shape[0], para);
            compute_sift_matches(k1, k2, matchings, para);
              
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((matchings.size(), 4), dtype="float32")
    for i in range(matchings.size()):
        out[i, 0] = matchings[i].first.y
        out[i, 1] = matchings[i].first.x
        out[i, 2] = matchings[i].second.y
        out[i, 3] = matchings[i].second.x
#    del matchings, k1, k2, para
    return out
    

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


def reduce_orsa(numpy.ndarray inp not None, bool verbose=False):
    """
    Call ORSA (keypoint checking) 
    @param inp: n*4 ot n*2*2 array representing keypoints. 
    @type in1: numpy ndarray
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    cdef int i, num_matchings, insize,p
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data = numpy.ascontiguousarray(inp.reshape(-1,4))
    insize = data.shape[0]
    cdef vector [ Match ]  match_coor = vector [ Match ](< size_t > insize)
    cdef int t_value_orsa = 10000
    cdef int verb_value_orsa = verbose
    cdef int n_flag_value_orsa = 0
    cdef int mode_value_orsa = 2
    cdef int stop_value_orsa = 0
    cdef float nfa
    cdef int width = int(1+max(data[:,1].max(),data[:,3].max()))
    cdef int heigh = int(1+max(data[:,0].max(),data[:,2].max()))
    cdef vector [ float ] index = vector [ float ](< size_t > data.shape[0])
    tmatch=time.time()
    with nogil: 
        for i in range(data.shape[0]):
            match_coor[i].y1 = < float > data[i,0]
            match_coor[i].x1 = < float > data[i,1]
            match_coor[i].y2 = < float > data[i,2]
            match_coor[i].x2 = < float > data[i,3]
    # epipolar filtering with the Moisan - Stival ORSA algorithm.
        nfa = orsa(width, heigh, match_coor, index, t_value_orsa, verb_value_orsa, n_flag_value_orsa, mode_value_orsa, stop_value_orsa)
    tend = time.time()
    num_matchings = index.size()
    if verbose:
        print("Matching with ORSA: %s => %s, took %.3fs, nfs=%s" % (insize, num_matchings, tend-tmatch,nfa))
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((num_matchings, 4), dtype="float32")
    for i in range(index.size()):
        p=<int>index[i]
        out[i,0] = data[p,0]
        out[i,1] = data[p,1]
        out[i,2] = data[p,2]
        out[i,3] = data[p,3]
    return out

