/**
 * \file integrale.h
 * \brief {File concerning calculus of integral image, convolution, haarwavelet, etc} 
 * 
 */


#include "integral.h"



/// Compute integrale image
/** Take in argument the image, symetrize it and then calculate.
 */
imageIntegral* computeIntegralImage(image* img,bool verbose)
{
	int starter=3*(3+pow(2.0f,octave)*(interval+2));
	image* stamp=new image(img->w()+2*starter,img->h()+2*starter);
	int i2,j2;
		for(int j=-starter;j<img->h()+starter;j++)
			for(int i=-starter;i<img->w()+starter;i++)

		{
			i2=i;
			j2=j;
			if(i<0)
				i2=-i;
			else if(i>img->w()-1)
				i2=2*(img->w()-1)-i;
			if(j<0)
				j2=-j;
			
			else if(j>img->h()-1)
			
				j2=2*(img->h()-1)-j;
				
			
			
			
			(*stamp)(i+starter,j+starter)=(*img)(i2,j2);
		}
//Now we use the stamp to compute its integral image.
	imageIntegral* imgInt=new imageIntegral(stamp);
	(*imgInt)(0,0)=(*stamp)(0,0);
	for(int i=1;i<stamp->w();i++)
		(*imgInt)(i,0)=(*imgInt)(i-1,0)+(*stamp)(i,0);
	
	
	for(int j=1;j<stamp->h();j++)
	{	
		float h=0.f;
	for(int i=0;i<stamp->w();i++)
	{
		h+=(*stamp)(i,j);
			(*imgInt)(i,j)=(*imgInt)(i,j-1)+h;
		}
	}
	delete stamp;
	if(verbose) 
		imgInt->printImage((char*)"imInt.png");
	imgInt->center(starter,img);
	
	return imgInt;
	
}



