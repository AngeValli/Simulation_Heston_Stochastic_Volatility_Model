#ifndef CLASSTENORCUBICSPLINE_H
#define CLASSTENORCUBICSPLINE_H
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include "class_Matrix_header.h"



class tenor_cubicspline
{
    public:
    tenor_cubicspline();
    tenor_cubicspline(const double& tenor,const Matrix& strike_vector,const Matrix& spline_param);
    tenor_cubicspline(const tenor_cubicspline& cubicspline);


    double get_tenor();
    int get_card_strike();
    Matrix get_strike_vector();
    Matrix get_spline_param();
    void describe_cubicspline();
    static tenor_cubicspline build_tenor_cubicspline_from_chain(const Matrix& tenor_vol_chain);
    int which_strike_index(const double &strike);
    double interpolate_vol(const double &strike);
    double compute_implied_vol(const double &strike);
    Matrix build_implied_vol_curve(const double& strike_mesh);
    Matrix build_implied_vol_curve(const double& strike_start,const double& strike_end,const int& card);









    private:

    double m_tenor;
    Matrix m_strike_vector;
    Matrix m_spline_param;




};
#endif