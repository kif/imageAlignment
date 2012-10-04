from libcpp cimport bool
from libcpp.pair  cimport pair
from libcpp.vector cimport vector

from image cimport flimage

cdef extern from "sift/sift.h":
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
    void compute_sift_keypoints_flimage(flimage img, keypointslist  keypoints, siftPar par) nogil

    #typedef std::pair<keypoint,keypoint> matching;
    ctypedef  pair[ keypoint  , keypoint  ] matching
    #typedef std::vector<matching> matchingslist;
    ctypedef  vector[ matching ] matchingslist
    void compute_sift_matches(keypointslist keys1, keypointslist keys2, matchingslist matchings, siftPar par) nogil

