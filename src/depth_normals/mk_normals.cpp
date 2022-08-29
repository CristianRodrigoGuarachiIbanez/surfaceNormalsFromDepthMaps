#include "mk_normals.h"
namespace NORM{
NormalizeImage::NormalizeImage(std::string filepath, std::string MODE){

    loadImage(filepath);
      selectMethod(MODE);
}

NormalizeImage::NormalizeImage(std::string filepath){
    loadImage(filepath);
}

NormalizeImage::~NormalizeImage(){
    this->normals.deallocate();
}

NormalizeImage::NormalizeImage(){

}

void NormalizeImage::loadImage(std::string filepath){
    this->depth = cv::imread(filepath);
    if (depth.empty()){
        throw;
    }
    cv::cvtColor(depth, depth, CV_RGB2GRAY);
    depth.convertTo(depth, CV_32FC1);
}

void NormalizeImage::selectMethod(std::string MODE){

    if(MODE == "conv"){
        convMethod();
    }else if(MODE == "cross"){
        crossMethod();
    }else if(MODE == "jetmap"){
        jetmap();
    }else if(MODE == "crossfast"){
        crossfastMethod();
    }else{
        std::cout<< " this option is not available! "<<std::endl;
        throw;
    }
}
void NormalizeImage::convMethod(){


 // Shape, Illumination, and Reflectance from Shading
        // Jonathan T. Barron (https://jonbarron.info/), Jitendra Malik
        // IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2015
        // https://drive.google.com/file/d/1RvyCiDMg--jyO8lLBvopp0o271LvREoa/view

        // filters
        cv::Mat_<float> f1 = (cv::Mat_<float>(3, 3) << 1,  2,  1,
                                                0,  0,  0,
                                                -1, -2, -1) / 8;

        cv::Mat_<float> f2 = (cv::Mat_<float>(3, 3) << 1, 0, -1,
                                                2, 0, -2,
                                                1, 0, -1) / 8;

        /* Other filters that could be used:
        % f1 = [0, 0, 0;
        %       0, 1, 1;
        %       0,-1,-1]/4;
        %
        % f2 = [0, 0, 0;
        %       0, 1,-1;
        %       0, 1,-1]/4;

        or

        % f1 = [0, 1, 0;
        %       0, 0, 0;
        %       0,-1, 0]/2;
        %
        % f2 = [0, 0, 0;
        %       1, 0, -1;
        %       0, 0, 0]/2;
        */

        cv::Mat f1m, f2m;
        cv::flip(f1, f1m, 0);
        cv::flip(f2, f2m, 1);

        cv::Mat n1, n2;
        cv::filter2D(depth, n1, -1, f1m, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(depth, n2, -1, f2m, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

        n1 *= -1;
        n2 *= -1;

        cv::Mat temp = n1.mul(n1) + n2.mul(n2) + 1;
        cv::sqrt(temp, temp);

        cv::Mat N3 = 1 / temp;
        cv::Mat N1 = n1.mul(N3);
        cv::Mat N2 = n2.mul(N3);

        std::vector<cv::Mat> N;
        N.push_back(N1);
        N.push_back(N2);
        N.push_back(N3);

        // cv::Mat normals;
        cv::merge(N, this->normals);

        //cv::imshow("convolution_based_normals", normals);

        normals *= 255;
        normals.convertTo(normals, CV_8UC3);
        // std::cout<< " hier " << normals.channels()<<std::endl;
        // cv::imwrite(PATH + "conv_normals.png", normals);
}

void NormalizeImage::crossMethod(){

    this->normals = cv::Mat(depth.size(), CV_32FC3);
    int yy, xx;
        for(int x = 0; x < depth.cols; ++x)
        {
            for(int y = 0; y < depth.rows; ++y)
            {
                    // 3d pixels, think (x,y, depth)
                    /* * * * *
                    * * t * *
                    * l c * *
                    * * * * */
                if(y == 0){
                    yy = 0;
                } else{
                 yy = y-1;
                }
                cv::Vec3f t(x,y-1,depth.at<float>(yy, x));
                if(x ==0){
                    xx = 0;
                }else{
                    xx = x-1;
                }
                cv::Vec3f l(x-1,y,depth.at<float>(y, xx));
                cv::Vec3f c(x,y,depth.at<float>(y, x));

                cv::Vec3f d = (l-c).cross(t-c);

                cv::Vec3f n = normalize(d);
                normals.at<cv::Vec3f>(y,x) = n;

            }
        }

        // cv::imshow("explicitly cross_product normals", normals);
        normals *= 255;
        normals.convertTo(normals, CV_8UC3);
        // cv::imwrite(PATH + "cross_normals.png", normals);

}
void NormalizeImage::crossfastMethod(){
    this->normals = cv::Mat(depth.size(), CV_32FC3);
    int xx, yy, yyy, xxx;
        for(int x = 0; x < depth.rows; ++x)
        {
            for(int y = 0; y < depth.cols; ++y)
            {
                // use float instead of double otherwise you will not get the correct result
                // check my updates in the original post. I have not figure out yet why this
                // is happening.
                // std::cout<< "dzdx -> "<< depth.at<float>(x+1, y) << " "  << depth.at<float>(x-1, y) <<std::endl;
                if(x < depth.rows-1){
                    xx = x+1;
                }else{
                    xx = 0;
                }
                if(x==0){
                    xxx = 0;
                }else{
                    xxx = x-1;
                }
                float dzdx = (depth.at<float>(xx, y) - depth.at<float>(xxx, y)) / 2.0;
                // std::cout<< " xxx -> " << xxx << "xx -> " << xx <<" x -> "<< x  <<std::endl;

                if(y==0){
                    yy = 0;
                }else{
                    yy = y-1;
                }
                if(y<depth.cols-1){
                    yyy = y+1;
                }else{
                    yyy = 0;
                }
                float dzdy = (depth.at<float>(x, yyy) - depth.at<float>(x, yy)) / 2.0;
                // std::cout<< " yyy -> " << yyy << "yy -> " << yy <<" y -> "<< y  <<std::endl;

                cv::Vec3f d(-dzdx, -dzdy, 1.0f);
                cv::Vec3f n = normalize(d);
                normals.at<cv::Vec3f>(x, y) = n;
            }
        }

        // cv::imshow("fast cross_product normals", normals);
        normals *= 255;
        normals.convertTo(normals, CV_8UC3);
        // cv::imwrite(PATH + "crossfast_normals.png", normals);
}

void NormalizeImage::jetmap(){
    // Multimodal Deep Learning for Robust RGB-D Object Recognition
        // Andreas Eitel, Jost Tobias Springenberg, Luciano Spinello, Martin Riedmiller, Wolfram Burgard
        // https://ieeexplore.ieee.org/abstract/document/7353446

        depth.convertTo(depth, CV_8UC1);
        this->embedding = cv::Mat(depth.size(), CV_8UC3);

        cv::applyColorMap(depth, embedding, cv::COLORMAP_JET);

        // cv::imshow("depth information to jet colormap", embedding);
        // cv::imwrite(PATH + "jetmapembedding.png", embedding);

}

void NormalizeImage::pprintImage(std::string PATH, std::string MODE){

     if(MODE == "conv"){
        cv::imwrite(PATH + "conv_normals.png", this->normals);
    }else if(MODE == "cross"){
        cv::imwrite(PATH + "cross_normals.png", this->normals);
    }else if(MODE == "jetmap"){
        cv::imwrite(PATH + "jetmapembedding.png", this->embedding);
    }else if(MODE == "crossfast"){
        cv::imwrite(PATH + "crossfast_normals.png", this->normals);
    }else{
        std::cout<< " this option is not available!"<<std::endl;
        throw;
    }
}
}

int main(int argc, char* argv[])
{
    // check input
    if (argc != 3)
    {
        std::cout << std::endl;
        std::cout << "DEPTH2NORMALS" << std::endl;
        std::cout << "Compute the normals of a grayscale/depth image." << std::endl;
        std::cout << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout <<"\t./depth2normals FILE MODE" << std::endl;
        std::cout << std::endl;
        std::cout << "FILE: grayscale/depth image path with extension" << std::endl;
        std::cout << "MODE: method to compute the normals; must be in [conv, cross, crossfast, jetmap]" << std::endl;
        std::cout << std::endl;

        return -1;
    }

    std::string FILE(argv[1]);
    std::string MODE(argv[2]);
    NORM::NormalizeImage img(FILE, MODE);
    //img.crossfastMethod();
    img.pprintImage("./", MODE);

}