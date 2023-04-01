#ifndef CLASSSURFACEVOL_H
#define CLASSSURFACEVOL_H
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include "class_tenor_cubicspline_header.h"
using namespace std;
class surface_vol {

    public:
    surface_vol();
    surface_vol(const Matrix& tenor_vector,const std::vector<tenor_cubicspline> &vol_param);

    static surface_vol build_surface_vol_from_chain(const Matrix& option_chain);

    std::vector<tenor_cubicspline> get_vol_param();
    Matrix get_tenor_vector();
    void describe_surface_vol();
    int which_tenor_index(const double& tenor);
    tenor_cubicspline get_tenor_cubicspline(const int& tenor_index);
    double interpolate_implied_vol(const double& tenor,const double& strike,const double& rate);
    double compute_implied_vol(const double& tenor,const double& strike,const double& rate);
    Matrix build_surface_vol(const double& tenor_start,const double& tenor_end,const int& tenor_card,const double& strike_start,const double& strike_end,const int& strike_card,const double& rate);




    private:
    Matrix m_tenor_vector;
    std::vector<tenor_cubicspline> m_vol_param;




};

#endif