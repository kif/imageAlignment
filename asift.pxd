from libcpp cimport bool
from libcpp.pair  cimport pair
from libcpp.vector cimport vector
from sift cimport keypointslist, siftPar
cdef extern from "asift/compute_asift_keypoints.h":
    int compute_asift_keypoints(vector[float] image, int width, int height, int num_of_tilts, int verb, vector[ vector[ keypointslist ]] & keys_all, siftPar siftparameters) nogil
cdef extern from "asift/compute_asift_matches.h":
    int compute_asift_matches(int num_of_tilts1, int num_of_tilts2, int w1, int h1, int w2, int h2, int verb, vector[vector[keypointslist]] keys1, vector[vector[keypointslist]] keys2, matchingslist matchings, siftPar siftparameters) nogil
