

#ifndef IMAGE

#define IMAGE

#include <iostream>
#include <sstream>


//#include "io_png.h"
#include <string>
#include "lib.h"

/** \class image image.h
 * Here we stock an image
 */

/** \class image image.h
 * Here we stock an image
 */


class image {
public:		
	 /// pointeur vers l'image	
	int idImage;/// This is put only in order to name images.
	
	


int	width,height;	/// image size ; trueWidth contains the real width for image integral
	float*	img;
	/// Give the array.
	int returnIdImage();
	inline float* array(){return img;}	
	/// Weight
	inline int w() {return width;}
	/// Height
	inline int h(){return height;}
	
	
	/// Affiche l'image tout en préfixant son nom par l'id.
	void printImage(  char nomFichier[]);
	
	void printImagePara(  char nomFichier[],image* para,image* orsa,image* line);

	/// Operator (x,y) give the matrix in x,y point.
	inline float operator()(int x, int y) const {
	
			
			   return img[ y*width + x ];
			
		
	}
	
	/// cf supra
	inline float& operator()(int x, int y) {
		
			 
			
			return img[ y*width+ x ];
		}
	
	///Affiche l'image avec des seuils.	
	void afficher_minimaliste(  char nomFichier[],float minSeuil,float maxSeuil);
	
	/// First constructor
	/** black image, 0 ID
	 */	image(int x,int y);
	
	/// Second constructor
	/** Black image, n ID
	 */
	image(int x,int y,int n);	
	/// Last constructor
	/** Black image, similar to im(same size)
	 */
	image(image* im);
	
	~image();
	image(const image &img);
};

class imageIntegral : public image {
	
public:	  int trueWidth;

	imageIntegral(image* im);
	~imageIntegral();
	float& operator()(int x, int y) {

		return img[ y*trueWidth+ x ];}
	float operator()(int x, int y) const {
				return img[ y*trueWidth+ x ];}
	/// Center for the integrale image(put the pointer at the good position because of symetrization)
	void center(int begin,image* im);	


};


#endif
