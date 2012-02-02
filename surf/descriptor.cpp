/**
 * \file descripteur.h
 * \brief Header for SURF descriptors
 *
 * All functions required are there.
 *
 */
#include "descriptor.h"


vectorDescriptor::vectorDescriptor(float a,float b,float c, float d)
{
	this->sumDx=a;
	this->sumDy=b;
	this->sumAbsDx=c;
	this->sumAbsDx=d;
}
vectorDescriptor::~vectorDescriptor()
{
}
vectorDescriptor::vectorDescriptor()
{}
descriptor::descriptor()
{
	this->list=new vectorDescriptor[16];
}

descriptor::descriptor(descriptor* a)
{
	this->list=new vectorDescriptor[16];
	for(int i=0;i<16;i++)
		this->list[i]=a->list[i];
	this->kP=new keyPoint(a->kP);
}


descriptor::~descriptor()
{
	
	
	delete kP;
	delete[] list;
}
descriptor::descriptor(const descriptor & des)
{
	this->list=new vectorDescriptor[16];
	this->kP=new keyPoint();
	for(int i=0;i<16;i++)
		(this->list)[i]=(des.list)[i];
	this->kP=(des.kP);
}


/// Create a list of descriptor
/** Everything is stocked in a vector
 * \param *imgInt : l'image intégrale
 * \param *lPC : la liste de points clefs à passer en adresse
 */
listDescriptor* getDescriptor(imageIntegral* imgInt,listKeyPoints* lPC)
{
	listDescriptor* lD=new listDescriptor();
	for(int i=0;i<lPC->size();i++)
		lD->push_back(makeDescriptor(imgInt, (*lPC)[i]));
	delete imgInt;
	return lD;
}





/// Compute the descriptor in 20*scale domain
/** Use kP given
 */
descriptor* makeDescriptor(imageIntegral* imgInt,keyPoint* pC)
{
	float scale=pC->scale;
	descriptor* desc=new descriptor();
	// Let's divide in a 4x4 zone the space around the interest point
	float cosA=cosf(pC->orientation);
	float sinA=sinf(pC->orientation);
	float norm=0;
	float u,v;
	float gauss,responseU,responseV,responseX,responseY;
	//We divide in 16 sectors
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<4;j++)
		{
			(desc->list[4*i+j]).sumDx=0;
			(desc->list[4*i+j]).sumAbsDx=0;
			(desc->list[4*i+j]).sumDy=0;
			(desc->list[4*i+j]).sumAbsDy=0;
			// Then each 4x4 becomes a 5x5 zone
			for(int k=0;k<5;k++)
			{
				for(int l=0;l<5;l++)
				{
					u=(pC->x+scale*(cosA*((i-2)*5+k+0.5)-sinA*((j-2)*5+l+0.5)));//-0.5 is here to center pixel
					v=(pC->y+scale*(sinA*((i-2)*5+k+0.5)+cosA*((j-2)*5+l+0.5)));
					responseX=haarX(imgInt,u,v,fround(scale) );
					responseY=haarY(imgInt,u,v,fround(scale));
					
					//We use a gaussian to weight
					gauss=gaussian(((i-2)*5+k+0.5),((j-2)*5+l+0.5),3.3f);
					responseU = gauss*( -responseX*sinA + responseY*cosA);
					responseV = gauss*(responseX*cosA + responseY*sinA);
					(desc->list[4*i+j]).sumDx+=responseU;
					(desc->list[4*i+j]).sumAbsDx+=absval(responseU);
					(desc->list[4*i+j]).sumDy+=responseV;
					(desc->list[4*i+j]).sumAbsDy+=absval(responseV);
					
				}
			}
			//We compute the norm of the vector
			norm+=(desc->list[4*i+j]).sumAbsDx*(desc->list[4*i+j]).sumAbsDx+(desc->list[4*i+j]).sumAbsDy*(desc->list[4*i+j]).sumAbsDy+((desc->list[4*i+j]).sumDx*(desc->list[4*i+j]).sumDx+(desc->list[4*i+j]).sumDy*(desc->list[4*i+j]).sumDy);			

		}
	}
//Then we normalized it.
	norm=sqrtf(norm);
	if(norm!=0)
	for(int i=0;i<16;i++)
	{
		(desc->list[i]).sumDx/=norm;
		(desc->list[i]).sumAbsDx/=norm;
		(desc->list[i]).sumDy/=norm;
		(desc->list[i]).sumAbsDy/=norm;	
	}
	desc->kP=new keyPoint(pC);
	return desc;
}


