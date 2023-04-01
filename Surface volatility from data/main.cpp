
#include "class_surface_vol_header.h"
#include "../class_Matrix_header.h"
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include <typeinfo>
#include <sstream>
using Pair=std::pair<double,double>;

int main(){

    Matrix read_matrix=Matrix::read_matrix_csv("./data/option_implied_vol.csv");
    surface_vol surface_test = surface_vol::build_surface_vol_from_chain(read_matrix);
    surface_test.describe_surface_vol();
    surface_test.build_surface_vol(0.01,10,1000,1,300,300,0.005).print_Matrix();

    return 0;
}