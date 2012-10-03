# Cython header for keypoint comparison using Orsa algorithm

from libcpp.vector cimport vector

cdef extern from "libMatch/match.h":
    struct Match:
        float x1, y1, x2, y2

cdef extern from "orsa/orsa.h":
    float orsa(int width, int height, vector[Match] match, vector[float] index, int t_value, int verb_value, int n_flag_value, int mode_value, int stop_value)nogil

ctypedef  vector[ Match ] MatchList
