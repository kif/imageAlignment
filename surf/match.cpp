/**
 * \file correspondances.h
 * \brief Header for matching
 *
 * Empty.
 *
 */

#include "match.h"

/// Function for matching, return the list of match
/** It uses the ration power.
 */
listMatch* matchDescriptor(listDescriptor * l1, listDescriptor * l2)
{
	listMatch* match=new listMatch();
	float thrm=rate*rate;
	
	for(int i=0;i<l1->size();i++)
	{
		int position=-1;
		float d1=3;
		float d2=3;
		
		for(int j=0;j<l2->size();j++)
		{
			float d=euclideanDistance((*l1)[i],(*l2)[j]);
			//We seek the secund minimum
			if((((*l1)[i])->kP)->signLaplacian==(((*l2)[j])->kP)->signLaplacian)
			{
				
				d2=((d2>d)?d:d2);
				if( d1>d)
				{
					position=j;
					d2=d1;
					d1=d;
				}
				
			}
			
			
		}
		
		// We want its position positive(found a min) and that there are at least another min behind it at 0.65 ratio
		if(position>=0  && thrm*d2>d1)
		{

			pairMatch p;
			p.first=new descriptor((*l1)[i]);
			p.second=new descriptor((*l2)[position]);

			match->push_back(p);	
		}
		
		
	}

	return match;
	
}
/// Make a lign btween a,b point with intensity given. Easy to read.

void lign(image *img,float xa,float ya,float xb,float yb,float intensite)
{
	xa=(int) xa;ya=(int)ya;xb=(int)xb;yb=(int)yb;
	if(xa>xb)
	{
		
lign(img,xb,yb,xa,ya,intensite);
	}
	
	else{
		if(xb==xa)
		{
			if(ya<yb)
				for(int i=0;i<=yb-ya;i++)
					(*img)(xa,ya+i)=intensite;
			else 
				for(int i=0;i<=ya-yb;i++)
					(*img)(xa,yb+i)=intensite;
			
			
		}
		else{
			float a,b;
			a=(float)(yb-ya)/(xb-xa);
			b=ya;
			for(int i=0;i<=xb-xa+1;i++)
			{
					if(a>0)
					{
					for(int j=(int)((float)(a*i)+b);j<(int)((float)(a*(i+1))+b);j++)
						if ( (j>ya && j<yb) || (j>yb && j<ya))

						(*img)(xa+i,j)=intensite;
					}

				else 
					for(int j=(int)((float)(a*(i+1))+b);j<(int)((float)(a*i)+b);j++)
						if ( (j>ya && j<yb) || (j>yb && j<ya))

						(*img)(xa+i,j)=intensite;
				
		
				
				
				
			}
			
			
		}
		
	}
}





/// Show descriptors
/** Put them with orientation on image.
 */
image*	 showDescriptors(image* img1,listDescriptor* listeDesc,bool afficher)
{
	image* img=new image(img1);
	
	for(int i=0;i<img->w();i++)
		for(int j=0;j<img->h();j++)
			(*img)(i,j)=0;
	
	for(int i=0;i<listeDesc->size();i++)
	{
		if(afficher)
		{ 
			std::cout<<"Descripteur : "<<i<<" : x > "<<((*listeDesc)[i]->kP)->x<<", y > "<<((*listeDesc)[i]->kP)->y<<", scale >"<<((*listeDesc)[i]->kP)->scale<<", orientation >"<<((*listeDesc)[i]->kP)->orientation<<", signe :"<<((*listeDesc)[i]->kP)->signLaplacian<<std::endl;
			std::cout<<"Elements associés :"<<std::endl;
		}
		float angle=((*listeDesc)[i]->kP)->orientation;
		
		float ech=((*listeDesc)[i]->kP)->scale;
		float cos1=cos(angle);
		float sin1=sin(angle);
		int x0=fround(((*listeDesc)[i]->kP)->x);
		int y0=fround(((*listeDesc)[i]->kP)->y);
		
		//We draw a circle...
		for(int i=0;i<1000;i++)
		(*img)(x0+cosf(i*2*pi/1000)*fround(1*ech),y0+sinf(i*2*pi/1000)*fround(1*ech))=100;
	
		
		
		
	
		
		for(int a=0;a<fround(1*ech)+1;a++)
		{
			(*img)(x0+a*cos1,y0+a*sin1)=50;
			
		}
		
	}
	return img;		
	
}



/// SQuare of euclidean distance...!
float euclideanDistance(descriptor* a,descriptor* b)
{
	
	float sum=0;
	for(int i=0;i<16;i++)
	{
		sum+=((a->list)[i].sumDx-(b->list)[i].sumDx)*((a->list)[i].sumDx-(b->list)[i].sumDx)
		+((a->list)[i].sumDy-(b->list)[i].sumDy)*((a->list)[i].sumDy-(b->list)[i].sumDy)
		+((a->list)[i].sumAbsDy-(b->list)[i].sumAbsDy)*((a->list)[i].sumAbsDy-(b->list)[i].sumAbsDy)
		+((a->list)[i].sumAbsDx-(b->list)[i].sumAbsDx)*((a->list)[i].sumAbsDx-(b->list)[i].sumAbsDx);
		
		
		
		
	}
	return sum;
}
