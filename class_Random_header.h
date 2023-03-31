#ifndef CLASSRANDOM_H
#define CLASSRANDOM_H
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>


class uniform_rv{
    public:
    uniform_rv();
    uniform_rv(const double& floor,const double& ceil);
    uniform_rv(const uniform_rv& rv);

    double get_floor();
    double get_ceil();

    double draw();

    private:
    double m_floor;
    double m_ceil;

};


class gaussian_rv{
    public:
    gaussian_rv();
    gaussian_rv(const double& mean,const double& std);
    gaussian_rv(const gaussian_rv& rv);

    double get_mean();
    double get_std();

    double draw();

    private:
    double m_mean;
    double m_std;

};

#endif