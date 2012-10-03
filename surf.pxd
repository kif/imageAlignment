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
    ctypedef vector[ keyPoint * ] listKeyPoints

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
