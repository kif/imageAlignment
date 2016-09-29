/**
 * \file keypoint.h
 * \brief Get SURF Key Points
 *
 * Involved functions
 *
 */

#ifndef KEYPOINT

#define KEYPOINT
#include <vector>
#include <math.h>
#include "lib.h"
#include "image.h"
/*#include <omp.h>
*/

/** 
 \class keyPoint For the keypoint.
 \brief Has the key point with natural notations.
 */
class keyPoint {
public:
	float	x,y,scale,orientation;
	bool signLaplacian;
	keyPoint(float a, float b, float c,  float e, bool f);
	
	keyPoint(keyPoint* a);
	keyPoint();
	
};

/// The vector of keypoint
typedef  std::vector<keyPoint*> listKeyPoints;



#include "descriptor.h"
#include "integral.h"


//inline bool isMaximum(image** imageStamp,int x,int y,int sigma);
void addKeyPoint(imageIntegral* img,float i,float j,bool signeLapl,float scale,listKeyPoints* listePointsClefs,bool verbose);
float getOrientation(imageIntegral* imgInt,int x,int y,int nombreSecteurs,float scale,bool verbose);
bool interpolationSpaceScale(image** img,int x, int y,int sig,float &scale,float &x2,float &y2, int iteration);
/// True if maxima STRICT
/** Compute 26 pixels & check it is more than the fixed threshold
 */
inline bool isMaximum(image** imageStamp,int x,int y,int scale)
{
	float tmp=(*(imageStamp[scale]))(x,y);
	if(absval(tmp)>threshold_maximum )
	{
			for(int j=-1+y;j<2+y;j++)
				for(int i=-1+x;i<2+x;i++)
					if((*(imageStamp[scale-1]))(i,j)>tmp or (*(imageStamp[scale]))(i,j)>tmp or (*(imageStamp[scale+1]))(i,j)>tmp )
						return false;
		
		
		return true;
	}						
	else
		return false;
}






#endif

