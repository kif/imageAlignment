#include <stdint.h>

#define POLY64REV     0x95AC9329AC4BC9B5
#define INITIALCRC    0xFFFFFFFFFFFFFFFF
uint64_t crc64(char *seq, unsigned int lg_seq)
{
    unsigned short i;
	unsigned char j;
    uint64_t crc = INITIALCRC;
	uint64_t part;
    static bool init = false;
    static uint64_t CRCTable[256];

    if (!init)
    {
		init = true;
		for (i = 0; i < 256; i++)
		{
		    part = i;
			for (j = 0; j < 8; j++)
		    {
				if (part & 1)
					part = (part >> 1) ^ POLY64REV;
				else
				    part >>= 1;
		    }
			CRCTable[i] = part;
		}
    }

    while (lg_seq-- > 0)
		crc = CRCTable[(crc ^ *seq++) & 0xff] ^ (crc >> 8);

	return crc;
}
