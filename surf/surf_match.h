/**
 * \file surf.h
 * \brief Header for SURF tools
 *
 * The only one you have to include.
 *
 */


#ifndef SURF
#define SURF
#include "integral.h"
#include "descriptor.h"
#include "keypoint.h"
#include "match.h"

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void get_points(listMatch* m,float* out);

#endif
