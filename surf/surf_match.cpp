/*
 *  surf_matching.cpp
 *  Extract a serie of matching point from surf comparison
 *
 *  Created by Jerome Kieffer (kieffer@esrf.fr).
 *  Copyright 2012 ESRF.
 *  Licenced under the GPL
 */
#define ABS(x)    (((x) > 0) ? (x) : (-(x)))

#include "surf_match.h"


using namespace std;

//Extract matching points
void get_points(listMatch* m,float* out){
	for(int i=0;i<m->size();i++)
	{
		out[4*i] = ((((*m)[i]).first)->kP)->y;
		out[4*i+1] = ((((*m)[i]).first)->kP)->x;
		out[4*i+2] = ((((*m)[i]).second)->kP)->y;
		out[4*i+3]  = ((((*m)[i]).second)->kP)->x;
	}
}









