
#include "image.h"


/** \class image image.h
 * Here we stock an image
 */


	

image::image(const image &img)
{
	std::cout<<"lol"<<std::endl;
}
	/// Print image
void image::printImage(  char nomFichier[])
	{

		std::string  nomFichier2="";
		
		nomFichier2+=nomFichier;
		unsigned char* data=new unsigned char[width*height];

		float max,mini,max_abs;
		max=img[0];
		mini=img[0];
		for(int i=0;i<width*height;i++)
		{if(mini>img[i]) mini=img[i]; if (max<img[i]) max=img[i];}
		
		max_abs=((absval(mini)>absval(max))?absval(mini):absval(max));
	
		
		
		for(int i=0;i<width*height;i++)
		{
			//float lambda=img[i]/max_abs;
			float lambda=(img[i]-max)/(mini-max);

			//data[i]=fround(lambda*127+127);
			data[i]=fround((1-lambda)*255);
		}
		

		
//		write_png_u8(nomFichier2.c_str(), data, width, height, 1);
		delete[] data;

		

	}
	
	/// Print image with a parallel one, which will be in color(you superpose them)
void image::printImagePara(  char nomFichier[],image* para,image* orsa,image* line)
{
	
	std::string  nomFichier2="";
	
	nomFichier2+=nomFichier;
	float* data=new float[3*width*height];
	
	float max,mini;
	max=img[0];
	mini=img[0];
	for(int i=0;i<width*height;i++)
	{
		if(mini>img[i]) mini=img[i]; if (max<img[i]) max=img[i];
	}
	
	
	
	
	
	for(int i=0;i<width*height;i++)
			{
				float lambda=(img[i]-max)/(mini-max);
				data[i]=fround(0*lambda+(1-lambda)*255);
				data[width*height+i]=fround(0*lambda+(1-lambda)*255);//fround(0*lambda+(1-lambda)*255);
				data[2*width*height+i]=fround(0*lambda+(1-lambda)*255);
				
				
				
								
				if((*line)(i%para->w(),i/para->w())!=0)
				{
					data[2*width*height+i]=fround(0*lambda+(1-lambda)*130)+(*para)(i%para->w(),i/para->w());
					
					data[0*width*height+i]=0;
					data[1*width*height+i]=0;
				}
				
				
				if((*orsa)(i%para->w(),i/para->w())!=0)
				{
					data[2*width*height+i]=0;
					
					data[0*width*height+i]=0;
					data[1*width*height+i]=fround(0*lambda+(1-lambda)*130)+(*para)(i%para->w(),i/para->w());
				}
				
				
				
				if((*para)(i%para->w(),i/para->w())!=0)
				{
					data[2*width*height+i]=0;
					
					data[0*width*height+i]=fround(0*lambda+(1-lambda)*130)+(*para)(i%para->w(),i/para->w());
					data[1*width*height+i]=0;
				}
				
				}	
	
	
	
//	write_png_f32(nomFichier2.c_str(), data, width,height,3);
	delete[] data;
	
	
	
}


	/// First constructor
	/** black image, 0 ID
	 */	image::image(int x,int y)
	{
		this->width=x;
		this->height=y;
		this->idImage=0;
		this->img=new float[width*height];
		
	}
	/// Second constructor
	/** Black image, n ID
	 */
image::image(int x,int y,int n)
	{
		this->width=x;
		this->height=y;
		this->idImage=n;

		this->img=new float[width*height];
		for(int i=0;i<width*height;i++)
			this->img[i]=0;
	
	}
	
	/// Last constructor
	/** Black image, similar to im(same size)
	 */
image::image(image* im)
	{
		this->width=im->width;
		this->height=im->height;
		this->idImage=im->idImage;
		this->img=new float[width*height];
		
	}
/// Constructor of integral image
imageIntegral::imageIntegral(image* im):image(im)
{
	
		
	this->trueWidth=im->width;
}
image::~image()
	{
		
		
			delete[] this->img;
		
	}

imageIntegral::~imageIntegral()
{
	

		for(int i=0;i<((trueWidth+1))*(trueWidth-width)/2;i++)
			img--;

	
}


int image::returnIdImage()
{
	return idImage;
}


/// Center for the integrale image(put the pointer at the good position because of symetrization)
void imageIntegral::center(int begin,image* im)
{
	for(int i=0;i<width*begin+begin;i++)
		img++;
	width=im->w();
	height=im->h();
	
}



