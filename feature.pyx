# -*- coding: utf8 -*-
#
#    Project: Azimuthal integration 
#             https://forge.epn-campus.eu/projects/azimuthal
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
    ctypedef vector[ keyPoint * ] listKeyPoints  #iter is declared as being of type vector<int>::iterator

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
    void compute_sift_keypoints(float * input, keypointslist  keypoints, int width, int height, siftPar par)
    #typedef std::pair<keypoint,keypoint> matching;
    ctypedef  pair[ keypoint  , keypoint  ] matching
    #typedef std::vector<matching> matchingslist;
    ctypedef  vector[ matching ] matchingslist
    void compute_sift_matches(keypointslist keys1, keypointslist keys2, matchingslist matchings, siftPar par)

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
        listeDesc2 = getKeyPoints(img2, octave, interval, l2, verbose)
        time_finish = time.time()
        matching = matchDescriptor(listeDesc1, listeDesc2)
        time_matching = time.time()
        print "SURF took %.3fs image1: %i ctrl points" % (time_int - time_init, listeDesc1.size())
        print "SURF took %.3fs image2: %i ctrl points" % (time_finish - time_int, listeDesc2.size())
        print("Matching %s point, took %.3fs " % (matching.size(), time_matching - time_finish))
    else:
        with nogil:
            listeDesc1 = getKeyPoints(img1, octave, interval, l1, verbose)
            listeDesc2 = getKeyPoints(img2, octave, interval, l2, verbose)
            matching = matchDescriptor(listeDesc1, listeDesc2)


    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((matching.size(), 4), dtype="float32")
    get_points(matching, < float *> (out.data))
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
    cdef image * img1 = new image(data1.shape[1], data1.shape[0])
    img1.img = < float *> data1.data
    cdef image * img2 = new image(data2.shape[1], data2.shape[0])
    img2.img = < float *> data2.data
    cdef keypointslist k1, k2
    cdef siftPar para
    cdef matchingslist n
    default_sift_parameters(para)
    if verbose:
        import time
        t0 = time.time()
        compute_sift_keypoints(< float *> data1.data, k1, data1.shape[1], data1.shape[0], para);
        t1 = time.time()
        compute_sift_keypoints(< float *> data2.data, k2, data2.shape[1], data2.shape[0], para);
        t2 = time.time()
        print "SIFT took %.3fs image1: %i ctrl points" % (t1 - t0, k1.size())
        print "SIFT took %.3fs image2: %i ctrl points" % (t2 - t1, k2.size())
        compute_sift_matches(k1, k2, n, para);
        print("Matching: %s point, took %.3fs " % (n.size(), time.time() - t2))
    else:
        compute_sift_keypoints(< float *> data1.data, k1, data1.shape[1], data1.shape[0], para);
        compute_sift_keypoints(< float *> data2.data, k2, data2.shape[1], data2.shape[0], para);
        compute_sift_matches(k1, k2, n, para);

#    if n.size > 10:
#        
#    else:
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((n.size(), 4), dtype="float32")
    for i in range(n.size()):
        out[i, 0] = n[i].first.y
        out[i, 1] = n[i].first.x
        out[i, 2] = n[i].second.y
        out[i, 3] = n[i].second.x
    return out
#TODO:
#        if(n.size() > 10)
#        {
#            std::vector < float > index;
#            // Guoshen Yu, 2010.09.23
#            // index.clear();
#
#            int t_value_orsa = 10000;
#            int verb_value_orsa = 0;
#            int n_flag_value_orsa = 0;
#            int mode_value_orsa = 2;
#            int stop_value_orsa = 0;
#
#            // epipolar filtering with the Moisan - Stival ORSA algorithm.
#            // float nfa = orsa(w1, h1, match_coor, index, t_value_orsa, verb_value_orsa, n_flag_value_orsa, mode_value_orsa, stop_value_orsa);
#
#            float nfa = orsa((img3 -> w() + img4 -> w()) / 2, (img3 -> h() + img4 -> h()) / 2, match2_coor, index, t_value_orsa, verb_value_orsa, n_flag_value_orsa, mode_value_orsa, stop_value_orsa);
#            cout << "ORSA(SIFT) said that : " << index.size() << " good matchs. nfa = " << nfa << endl;
