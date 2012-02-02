/*
 *  lib.h
 *  SURF
 *
 *  Created by Edouard Oyallon on 19/06/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef LIB
#define LIB
#include <math.h>
inline int fround(float flt);
inline float gaussian(float x, float y, float sig);
inline float absval(float x);
inline int absval(int x);

/** \def pi
 * \brief What else ?
 */
#define pi 3.14159265358979323846 /// Constante pi...
/** \def threshold
 * \brief Threshold for which a point of the Hessian is compute as a maxima.
 */
#define threshold_maximum 4000

/** \def nombreSecteur
 * \brief Nombre de secteur qu'on considère pour la fonction d'orientation
 */
#define number_sector 16 
/** \def rate
 * \brief Rate between the last and previous last minimum in euclidean norm for matching
 */
#define rate 0.8
/** \def octave
 * \brief Clear
 */
#define octave 4
/** \def intervall
 * \brief Clear
 */
#define interval 4

/** \def iteration_interpolation
 * \brief Number of interpolation possible.(recursrive)
 */
#define iteration_interpolation 5


/// Compute a gaussian
/** Empty..
 */
inline float gaussian(float x, float y, float sig)
{
	return 1/(2*pi*sig*sig)*expf( -(x*x+y*y)/(2*sig*sig));
}

/// Round function...
/** Set inline... */
inline int fround(float flt)
{
	return (int) (flt+0.5f);
}
/// Absolute value int
inline int absval(int x)
{
	return ((x>0)?x:-x);
}

/// Absolute value float
inline float absval(float x)
{
	return ((x>0)?x:-x);
}
#endif