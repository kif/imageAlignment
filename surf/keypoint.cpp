/**
 * \file pointsclefs.h
 * \brief Get SURF Key Points
 *
 * Involved functions
 *
 */

#include <ctime>

#include "keypoint.h"
#include <sstream>
#include <iostream>
#include <fstream>
/*#include <omp.h>
*/

keyPoint::keyPoint(float a, float b, float c,  float e, bool f)
{
	this->x=a;
	this->y=b;
	this->scale=c;
	this->orientation=e;
	this->signLaplacian=f;
}
keyPoint::keyPoint(keyPoint* a){
	this->x=a->x;
	this->y=a->y;
	this->scale=a->scale;
	this->orientation=a->orientation;
	this->signLaplacian=a->signLaplacian;
}
keyPoint::keyPoint()
{
}


/// Function which computes Hessian at each scales, and then get key points with the extrema detections.
/** Each argument is taken in parameters.
 * \param *img image to compute
 * \param numberOctave number of octave to compute - should be 4
 * \param numberInterval number of interval - should be 4
 * \param *listKeyPoints list of keypoints
 * \return listDescriptors clear
 */
listDescriptor* getKeyPoints(image *img,int numberOctave,int numberInterval,listKeyPoints* lKP,bool verbose)
{	

	//Integral image.
	imageIntegral* imgInt=computeIntegralImage(img,verbose);
	image*  hessian[numberInterval];
	image* signLaplacian[numberInterval];

	
	//Initialize
		for(int i=0;i<numberInterval;i++)
		{
			hessian[i]=new image(img);//Ca c'est pas top niveau resources regarder du côté de <vector> plus tard
			signLaplacian[i]=new image(img);

		
		}

	// Let's compute the Hessian for each step

	// To compute the size l of the filter.
	int puiss2=1;
	
	int l;

	float Dxx,Dxy,Dyy;
	
	float aux4;
	float aux1;
	float aux2;
	float aux5;
	float aux3;
	float l4;
	float ml;
	int intervalCounter;
	int octaveCounter;
	int x,y,i;
	int h=imgInt->h();
	int w=imgInt->w();

	for( octaveCounter=0;octaveCounter<numberOctave;octaveCounter++)
	{
		puiss2*=2;	
		

		for( intervalCounter=0;intervalCounter<numberInterval;intervalCounter++)
		{
			l=(puiss2*(intervalCounter+1)+1); //the famous l
		// aux are there in order to reduce the number of computation in the further for			
			aux4=-l+1;
			aux1=aux4/2;
			aux2=3*l;
			aux5=aux1-l;
			aux3=2*l-1;
			l4=1/(float)(l*l*l*l);
			ml=-l;
			float *a=(*hessian[intervalCounter]).img;
			float *b=(*signLaplacian[intervalCounter]).img;
			float* imgInt2=(*imgInt).img;
			int w2=imgInt->trueWidth;
			// Let's compute it at the given scale.
			 i=0;
#pragma intel for parallel private(x,Dxx,Dyy,Dxy) firstprivate(w2,aux1,aux2,aux3,aux4,l,aux5,intervalCounter,l4,w)// shared(hessian,signLaplacian,imgInt)

				for( y=0;y<h;y++)
				{
					{

						for( x=0;x<w;x++)

					{

				// We build filters. If you want to check them, just compute with a Dirac image.
					
					Dxx=squareConvolutionXY2(imgInt2,w2,aux4,aux5,aux3,aux2,x,y)-3.*squareConvolutionXY2(imgInt2,w2,aux4,aux1,aux3,l,x,y);
					Dyy=squareConvolutionXY2(imgInt2,w2,aux5,aux4,aux2,aux3,x,y)-3.*squareConvolutionXY2(imgInt2,w2,aux1,aux4,l,aux3,x,y);
						Dxy=squareConvolutionXY2(imgInt2,w2,1,1,l,l,x,y)+squareConvolutionXY2(imgInt2,w2,0,0,ml,ml,x,y)+squareConvolutionXY2(imgInt2,w2,1,0,l,ml,x,y)+squareConvolutionXY2(imgInt2,w2,0,1,ml,l,x,y);
					
							// We weight it by the size.
					
					a[ y*w + x]=(Dxx*Dyy-Dxy*Dxy*0.81)*l4;b[ y*w+ x]=Dxx+Dyy>0;
					//(hessian[intervalCounter])->img[i]=(Dxx*Dyy-Dxy*Dxy*0.81f)*l4;(signLaplacian[intervalCounter])->img[i]=Dxx+Dyy>0;i++;
					}}
				}
			
		}
		float dscale=0;
		float dx=0;
		float dy=0;
		// Now we seek extrema
		// We can't seek maxima at the interval border so it starts to 1, end to numberInterval-1

		for(int intervalCounter=1;intervalCounter<numberInterval-1;intervalCounter++)
			{
				l=(puiss2*(intervalCounter+1)+1);
				//We don't seek extrema on side

#pragma intel for parallel private(x,dx,dy,dscale) firstprivate(l,puis2,verbose,intervalCounter) shared(lKP)// shared(hessian,signLaplacian,imgInt)
{
				for( int y=1;y<h-1;y++)

				for( int x=1;x<w-1;x++)
						
						
						// Maxima+Interpolation (infinite norm <0.5)
						if(isMaximum(hessian,x,y,intervalCounter ))
							if( interpolationSpaceScale(hessian,x,y,intervalCounter,dscale,dx,dy,iteration_interpolation))
											
							addKeyPoint(imgInt,(float) x+dx,(float)y+dy, (*(signLaplacian[intervalCounter]))(x,y),0.4f*((float) l+puiss2*dscale),lKP,verbose);//on peut prendre sigma sans prendre la valeur associé à l'échelle par continuité de tous les motifs..
						
}					
					
			
				if(verbose)
				{
					std::string a="tmp/i";
					
					std::ostringstream oss;
					oss<<img->returnIdImage();
					
					oss<<octaveCounter;
					
					a+="/ioi";
					
					
					oss<<intervalCounter;
					a+=oss.str();
					
					
					
					a+=".png";
					(hessian[intervalCounter])->printImage((char *)a.c_str());
					}
					
					
					}
				
			
		
		}
		
	
	

	//Let's free memory
for(int j=0;j<numberInterval;j++)
{
delete hessian[j];
delete signLaplacian[j];
}
	//Let's get descriptor now 
	return getDescriptor(imgInt,lKP);
}





/// Add a key point in a vector
/** It call the function getOrientation
 */

void addKeyPoint(imageIntegral* img,float i,float j,bool signL,float scale,listKeyPoints* lKP,bool verbose)
{
	// We prefer not to have side effetcs
	if(i>fround(scale*10)+2 && j>fround(10*scale)+2 && img->w()-fround(10*scale)-2>i && img->h()-fround(10*scale)-2>j)
	{	
		keyPoint* pt=new keyPoint(i,j,scale,getOrientation(img,  i,j,number_sector,scale,verbose),signL);
		lKP->push_back(pt);
	}
	
}






/// Find orientation of a keypoint
/** We build sectors and we get the wavelet response.
 * In a Pi/3 angular area, the extremum is taken.
 */
float getOrientation(imageIntegral* imgInt,int x,int y,int sectors,float scale,bool verbose)
{
	float haarResponseX[sectors];
	float haarResponseY[sectors];
	float haarResponseSectorX[sectors];
	float haarResponseSectorY[sectors];
	float answerX;
	float answerY,gauss;
	int theta;
	//We put it to 0
	for(int i=0;i<sectors;i++)
	{
		haarResponseSectorX[i]=0;
		haarResponseSectorY[i]=0;
		haarResponseX[i]=0;
		haarResponseY[i]=0;
	}
	
	// We compute answer in each sector
	for(int i = -6; i <= 6; i++) 
	{
		for(int j = -6; j <= 6; j++) 
		{
			if(i*i + j*j <= 36) 
			{
				//We get the answer
				 answerX=haarX(imgInt, x+scale*i,y+scale*j,fround(2*scale));
				 answerY=haarY(imgInt, x+scale*i,y+scale*j,fround(2*scale));
				
				//We compute the angle
				theta=(int)( atan2f(answerY,answerX)* sectors/(2*pi)); 
				theta=((theta>=0)?(theta):(theta+sectors));
				// We weight by a gaussian
				 gauss=gaussian(i,j,2);
				
				//We add the answer
				haarResponseSectorX[theta]+=answerX*gauss;
				haarResponseSectorY[theta]+=answerY*gauss;
			}
		}
	}

	// Now we compute the solution in a pi/3 windows.	
	for(int i=0;i<sectors;i++)
	{
		for(int j=-fround(sectors/12);j<=fround(sectors/12);j++)
		{
			
			if(0<=i+j && i+j<sectors)
			{
				haarResponseX[i]+=haarResponseSectorX[i+j];
				haarResponseY[i]+=haarResponseSectorY[i+j];
			}
			// We work %sectors
			else if(i+j<0)
			{
				haarResponseX[i]+=haarResponseSectorX[sectors+i+j];
				haarResponseY[i]+=haarResponseSectorY[i+j+sectors];
			}
			
			else 
			{
				haarResponseX[i]+=haarResponseSectorX[i+j-sectors];
				haarResponseY[i]+=haarResponseSectorY[i+j-sectors];				
			}
			

			
		}
	}

// Now we seek the maximum
	float max=haarResponseX[0]*haarResponseX[0]+haarResponseY[0]*haarResponseY[0];
	int t=0;
	for(int i=1;i<sectors;i++)
	{
		float norme=haarResponseX[i]*haarResponseX[i]+haarResponseY[i]*haarResponseY[i];
		t=((max<norme)?i:t);
		max=((max<norme)?norme:max);
	}

//We show circular graphics	
	if(verbose )
	{
		image* onche=new image(2*(fround(6*scale)),2*fround(6*scale));
		
		for(int i = -fround(6*scale); i < fround(6*scale); i++) 
		{
			for(int j = -fround(6*scale); j < fround(6*scale); j++) 
			{
				(*onche)(i+fround(6*scale),j+fround(6*scale))=0;
				if(i*i + j*j <= fround(36*scale*scale)) 
				{
					// The angular sector : we divide in 2pi/sector and then compute in the appropriate one the response
					int theta=(int)((atan2f(j,i))*((float)sectors)/( 2.0*(float) pi));
					
					// Problem is atan2 has value in [-pi,pi]
					theta=((theta>=0)?theta:theta+sectors);
					(*onche)(i+fround(6*scale),j+fround(6*scale))=haarResponseSectorX[theta]*haarResponseSectorX[theta]+haarResponseY[theta]*haarResponseY[theta];
				}
			}
		}
		
		for(int i=0;i<fround(6*scale);i++)
			(*onche)(fround(6*scale)+i*cos(atan2f(haarResponseY[t],haarResponseX[t])),fround(6*scale)+i*sin(atan2f(haarResponseY[t],haarResponseX[t])))+=max;
		
		std::string a="tmp/i";
		
		std::ostringstream oss;
		oss<<x;
		oss<<y;
		
		
		a+="/xy";
		
		
		a+=oss.str();
		a+=".png";
		onche->printImage((char *)a.c_str());
		delete onche;
	}
	
	
	
//Now we can send the maximum
	
	return atan2f(haarResponseY[t],haarResponseX[t]);
}







/// We interpolate maximum with finite difference
/** Tolerance put to 0.5(a pixel).
 */
bool interpolationSpaceScale(image** img,int x, int y,int sig,float &scale,float &x2,float &y2,int iteration)
{
	//If outside image...
	if(x<=0 || y<=0 || x>=img[sig]->w()-2 || y>=img[sig]->h()-2)
		return false;
	float dx,dy,dsig,dxx,dyy,dsigsig,dxy,dxsigma,dysigma;
	//Nabla X
	dx=((*(img[sig]))(x+1,y)-(*(img[sig]))(x-1,y))/2;
	dy=((*(img[sig]))(x,y+1)-(*(img[sig]))(x,y-1))/2;
	dsig=((*(img[sig+1]))(x,y)-(*(img[sig-1]))(x,y))/2;
	//Hessian X
	float a=(*(img[sig]))(x,y);
	dxx=(*(img[sig]))(x+1,y)+(*(img[sig]))(x-1,y)-2*a;
	dyy=(*(img[sig]))(x,y+1)+(*(img[sig]))(x,y+1)-2*a;
	dsigsig=((*(img[sig-1]))(x,y)+(*(img[sig+1]))(x,y)-2*a);
	
	dxy=((*(img[sig]))(x+1,y+1)-(*(img[sig]))(x+1,y-1)-(*(img[sig]))(x-1,y+1)+(*(img[sig]))(x-1,y-1))/4;
	dxsigma=((*(img[sig+1]))(x+1,y)-(*(img[sig+1]))(x-1,y)-(*(img[sig-1]))(x+1,y)+(*(img[sig-1]))(x-1,y))/4;
	dysigma=((*(img[sig+1]))(x,y+1)-(*(img[sig+1]))(x,y-1)-(*(img[sig-1]))(x,y+1)+(*(img[sig-1]))(x,y-1))/4;
	
	float det=(dxx*dyy*dsigsig-dxx*dysigma*dysigma-dyy*dxsigma*dxsigma+2*dxsigma*dysigma*dxy-dsigsig*dxy*dxy);
	if(det!=0) //Matrix must be inversible
	{
		x2=(-1/det*(dx*(dyy*dsigsig-dysigma*dysigma)+dy*(dxsigma*dysigma-dsigsig*dxy)+dsig*(dxy*dysigma-dyy*dxsigma)));
		y2=(-1/det*(dx*(dxsigma*dysigma-dsigsig*dxy)+dy*(dxx*dsigsig-dxsigma*dxsigma)+dsig*(dxy*dxsigma-dxx*dysigma)));
		scale=-1/det*((dxy*dysigma-dyy*dxsigma)*dx+dy*(dxy*dxsigma-dxx*dysigma)+dsig*(dxx*dyy-dxy*dxy));
		if(absval(x2)<0.5 && absval(y2)<0.5 && absval(scale)<0.5)
		{
			
			
			return true;
		}
		else {
			if( iteration==0)
			return false;
			else 
			{
				iteration--;// We call it recursively
				return interpolationSpaceScale(img,(int)(x+x2+0.5),(int)(y+y2+0.5),sig,scale,x2,y2,iteration);
			}
			
		}
		
	}
	else {
		return false;
	}
	
}
