/**
 * \file integrale.h
 * \brief {File concerning calculus of integral image, convolution, haarwavelet, etc} 
 * 
 */

/** \def m
 * \brief get min between a and b
 */
#ifndef INTEGRAL
#define INTEGRAL
#include "lib.h"
#include "image.h"


imageIntegral* computeIntegralImage(image* img,bool verbose);
//float squareConvolutionXY(image* imgInt,int a,int c,int b,int d,int x,int y);



inline float squareConvolutionXY(imageIntegral* imgInt,int a,int b,int c,int d,int x,int y)
{
	int a1=x-a;
	int a2=y-b;
	int b1=a1-c;
	int b2=a2-d;
	
		return (*imgInt)(b1,b2)+(*imgInt)(a1,a2)-(*imgInt)(b1,a2)-(*imgInt)(a1,b2);// ((A+D)-(B+C));
	
}	
/*----------------------------------------------------*/
/// Compute convolution by a square
/** Compute image in the point (x,y)
 * \param (a,b) inferior side of the square
 * \param (c,d) weight and height
 */
inline float squareConvolutionXY2(float* img,int w,int a,int b,int c,int d,int x,int y)
{
	int a1=x-a;
	int a2=y-b;
	int b1=a1-c;
	int b2=a2-d;
	return img[ b2*w + b1 ]+img[ a2*w + a1 ]-img[ a2*w + b1 ]-img[ b2*w + a1 ];
	
}






/// Haar belong X
/** Compute in (x,y)
 */
inline float haarX(imageIntegral* img,int x,int y,int tailleFiltre)
{
	
	return -(squareConvolutionXY(img,1,-tailleFiltre-1,-tailleFiltre-1,tailleFiltre*2+1, x, y)+
			squareConvolutionXY(img, 0,-tailleFiltre-1, tailleFiltre+1,tailleFiltre*2+1, x, y));
	
	
}



/// Haar belong Y
/** Compute in (x,y)
 */

inline float haarY(imageIntegral* img,int x,int y,int tailleFiltre)
{return -(squareConvolutionXY(img, -tailleFiltre-1,1, 2*tailleFiltre+1,-tailleFiltre-1, x, y)+
		 squareConvolutionXY(img, -tailleFiltre-1,0, 2*tailleFiltre+1,tailleFiltre+1, x, y));
}

#endif

