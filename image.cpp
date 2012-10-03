#include <iostream>
#include <vector>

class Image{
	public:
		Image(){
			std::cout<<"Image::Image() default constructor"<<std::endl;
		}

		Image(const Image&){
			std::cout<<"IMage::Image() copy constructor"<<std::endl;
		}
		Image(Image&&) noexcept{
			std::cout<<"Image::Image() move constructor"<<std::endl;
			}
		~Image(){
			std::cout<<"Image::Image default destructor "<<std::endl;
			}
		Image& operator= (const Image&){
			std::cout<<"Assignment with copy"<<std::endl;
			return *this;
		}
		Image& operator= (Image&&) noexcept {
			std::cout<<"Assignment with move"<<std::endl;
			return *this;
		}

};

Image foo(){
	Image i;
	return i;
}

int main(){
	std::vector<Image> v;
	//v.reserve(100);
	Image i;
	std::cout <<"pass 1"<<std::endl;
	v.push_back(Image());
	std::cout <<"pass 2"<<std::endl;
	i=Image();
	std::cout <<"pass 3"<<std::endl;
	v.push_back(i);
	std::cout <<"pass 4"<<std::endl;
	Image i2=foo();
	std::cout <<"pass 5"<<std::endl;

}
