#!/usr/bin/cython

__author__ = "Jerome Kieffer"
__date__ = "13/11/2013"
__doc__ = """
Cython binding for ORSA: Optimized Ransac
"""
import time
import numpy
cimport numpy
import cython
from libcpp.vector cimport vector
from orsa_cpp cimport Match, MatchList, orsa

dtype_kp = numpy.dtype([('x', numpy.float32),
                        ('y', numpy.float32),
                        ('scale', numpy.float32),
                        ('angle', numpy.float32),
                        ('desc', (numpy.uint8, 128))
                        ])

cdef packed struct dtype_kp_t:
    numpy.float32_t   x, y, scale, angle
    unsigned char desc[128]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sift_orsa(inp not None, shape=None, bint verbose=0):
    """
    Call ORSA (keypoint checking) on sift matched keypoints
    
    @param inp: n*2 array representing of keypoints.
    @param shape: shape of the input images (unless guessed)
    @type shape: 2-tuple of integers
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """

    cdef int i, num_matchings
    cdef float[:,:] data_x = inp.x
    cdef float[:,:] data_y = inp.y
    cdef int insize = inp.shape[0]
    if insize < 10:
        return inp
    cdef vector [ Match ]  match_coor = vector [ Match ](< size_t > insize)
    cdef int t_value_orsa = 10000
    cdef int verb_value_orsa = verbose
    cdef int n_flag_value_orsa = 0
    cdef int mode_value_orsa = 2
    cdef int stop_value_orsa = 0
    cdef float nfa
    cdef int width, heigh
    if shape is None:
        # keypoints are at least a 5 pixels of the border
        width = int(5 + max(inp[:, 0].x.max(), inp[:, 1].x.max())) 
        heigh = int(5 + max(inp[:, 0].y.max(), inp[:, 1].y.max())) 
    elif hasattr(shape, "__len__") and len(shape) >= 2:
        width = int(shape[1])
        heigh = int(shape[0])
    else:
        width = heigh = int(shape)
    cdef vector [ float ] index = vector [ float ](< size_t > insize)
    tmatch = time.time()
    with nogil:
        for i in range(insize):
            match_coor[i].x1 = data_x[i,0]
            match_coor[i].y1 = data_y[i,0]
            match_coor[i].x2 = data_x[i,1]
            match_coor[i].y2 = data_y[i,1]

    # epipolar filtering with the Moisan - Stival ORSA algorithm.
        nfa = orsa(width, heigh, match_coor, index, t_value_orsa, verb_value_orsa, n_flag_value_orsa, mode_value_orsa, stop_value_orsa)
    tend = time.time()
    num_matchings = index.size()
    if verbose:
        print("Matching with ORSA: %s => %s, took %.3fs, nfs=%s" % (insize, num_matchings, tend - tmatch, nfa))
#    cdef numpy.ndarray[numpy.int64_t, ndim=1] out_index  = numpy.zeros(num_matchings, dtype=numpy.int64)
#    with nogil:
#        for i in range(num_matchings):
#            out_index[i] = < int > index[i]
    out_index = numpy.array(index, dtype=numpy.int32) 
    out = inp[out_index]
    return out
