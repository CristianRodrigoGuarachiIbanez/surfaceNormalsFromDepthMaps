#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
namespace NORM{
class NormalizeImage{
    public:
    NormalizeImage(std::string filepath, std::string MODE);
    NormalizeImage(std::string filepath);
    NormalizeImage();
    ~NormalizeImage();

    void loadImage(std::string filepath);
    void setImage(cv::Mat&image){
        this->depth = image.clone();
        if (depth.empty()){
            throw;
        }
        depth.convertTo(depth, CV_32FC1);
    }
    void selectMethod(std::string MODE);
    void convMethod();
    void crossMethod();
    void crossfastMethod();
    void jetmap();
    cv::Mat getNormalizedImage(){
        if(!this->normals.empty()){
            return this->normals;
        }else{
            std::cout<< "no normalized image was generated! " << std::endl;
        }
    }
    cv::Mat getEmbeddedImage(){
        if(!this->embedding.empty()){
            return this->embedding;
        }else{
            std::cout<< "no embedded image was generated! " << std::endl;
        }
    }
    void pprintImage(std::string PATH, std::string MODE);
    private:
    cv::Mat depth;
    cv::Mat normals;
    cv::Mat embedding;
    std::string output;


};
}