
#include "class_HestonModel_header.h"
#include "class_PathSimulatorEuler2F_header.h"
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

    double risk_free_rate=0.01;
    Pair init_spot_variance=Pair(10,1);
    double correlation = 0.1;
    double reversion_level=1;
    double reversion_speed=0.1;
    double vol_of_vol=0.1;

    HestonModel heston_model = HestonModel(risk_free_rate,init_spot_variance,correlation,reversion_level,reversion_speed,vol_of_vol);

    Matrix time_point=Matrix::matrix_linspace(0,10,1000,true);

    time_point.print_Matrix();

    return 0;
}