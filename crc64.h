#include <stdint.h>
#define POLY64REV     0x95AC9329AC4BC9B5
#define INITIALCRC    0xFFFFFFFFFFFFFFFF

uint64_t crc64(char *seq, unsigned int lg_seq);
