from libc.stdint cimport uint64_t
cdef extern from "crc64.h":
    uint64_t crc64(char * seq, unsigned int lg_seq)
