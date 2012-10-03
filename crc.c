#define POLY64REV     0x95AC9329AC4BC9B5
#define INITIALCRC    0xFFFFFFFFFFFFFFFF

void crc64(char *seq, unsigned int lg_seq, unsigned __int64 *pFinalCrc)
{
    unsigned short i;
	unsigned char j;
    unsigned __int64 crc = INITIALCRC;
	unsigned __int64 part;
    static bool init = false;
    static unsigned __int64 CRCTable[256];

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

	*pFinalCrc = crc;

    return;
}
