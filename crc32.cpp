#include <stdint.h>
#include <smmintrin.h>
#include <cpuid.h>

bool initialized(false);
uint32_t slowcrc_table[1<<8];

void slowcrc_init() {
	uint32_t i, j, a;

	for (i=0;i<(1<<8);i++) {
		a=((uint32_t)i)<<24;
		for (j=0;j<8;j++) {
			if (a&0x80000000)
				a=(a<<1)^0x11EDC6F41;
			else
				a=(a<<1);
		}
		slowcrc_table[i]=a;
	}
	initialized=true;
}

uint32_t slowcrc(char *str, uint32_t len) {
	uint32_t lcrc=~0;
	char *p, *e;

	e=str+len;
	for (p=str;p < e;++p)
		lcrc=(lcrc>>8)^slowcrc_table[(lcrc^(*p))&0xff];
	return ~lcrc;
}

uint32_t crc32(char *str, uint32_t len) {
  		 if (!initialized){
  			slowcrc_init();
  		 }
  		 return slowcrc(str,len);
}
