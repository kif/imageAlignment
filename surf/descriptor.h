/**
 * \file descriptor.h
 * \brief Header for SURF descriptors
 *
 * All functions required are there.
 *
 */
/// \class descriptor
/** \brief The descriptor itself
 */
#ifndef DESCRIPTOR

#define DESCRIPTOR
#include <math.h>
#include "image.h"
#include "keypoint.h"

#include "integral.h"
#include "descriptor.h"

#include "lib.h"



/// \class vectorDescriptor
/** \brief A descriptor */
class vectorDescriptor{
	public :
	float sumDx;
	float sumDy;
	float sumAbsDx;
	float sumAbsDy;
	vectorDescriptor(float a,float b,float c, float d);
	~vectorDescriptor();
	vectorDescriptor();
};




class descriptor{
public:vectorDescriptor* list;///Will be the 16 elements of the descriptor
	keyPoint* kP;/// Keypoint linked to the descriptor.
	descriptor();
	descriptor(descriptor* a);
	
	~descriptor();
    descriptor(const descriptor & des);
	
};

typedef  std::vector<descriptor*> listDescriptor;



descriptor* makeDescriptor(imageIntegral* imgInt,keyPoint* pC);
listDescriptor* getDescriptor(imageIntegral* imgInt,listKeyPoints* lPC);
listDescriptor* getKeyPoints(image *img,int numberOctave,int numberInterval,listKeyPoints* lKP,bool verbose);

#endif