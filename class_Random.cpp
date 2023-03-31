#include "class_Random_header.h"


uniform_rv::uniform_rv():m_ceil(0.0),m_floor(1.0){}
uniform_rv::uniform_rv(const double& floor,const double& ceil):m_ceil(ceil),m_floor(floor){}

double uniform_rv::get_ceil(){
    // Method to get the ceil value used for defining the uniform random variable
    return m_ceil;
}

double uniform_rv::get_floor(){
    // Method to get the floor value used for defining the uniform random variable
    return m_floor;
}

double uniform_rv::draw(){
    // Method to draw the uniform random variable
    std::random_device rd;  // It will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution <> dis(0.0,1.0);
    return(m_floor+(m_ceil-m_floor)*dis(gen));
}

gaussian_rv::gaussian_rv():m_mean(0.0),m_std(1.0){}
gaussian_rv::gaussian_rv(const double& mean,const double& std):m_mean(mean),m_std(std){}

double gaussian_rv::get_mean(){
    // Method to get the mean value used for defining the gaussian random variable
    return m_mean;
}
double gaussian_rv::get_std(){
    // Method to get the standard value used for defining the gaussian random variable
    return m_std;
}

double gaussian_rv::draw(){
    // Method to draw the gaussian random variable
    uniform_rv seed_uniform = uniform_rv();
    double u1=seed_uniform.draw();
    double u2=seed_uniform.draw();
    double rv=sqrt(-2*log(u1))*cos(2*M_PI*u2);
    return(m_mean+m_std*rv);
}