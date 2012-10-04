
cdef extern from "image/flimage.h":
    cdef cppclass flimage:
        flimage()
        flimage(int w, int h)
        flimage(int w, int h, float v)
        flimage(int w, int h, float * v)
        int nwidth() nogil
        int nheight() nogil
