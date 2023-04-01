
#ifndef CLASSPATHSIMULATOREULER2F_H
#define CLASSPATHSIMULATOREULER2F_H
#include "../class_Matrix_header.h"
#include "class_Model2F_header.h"
#include "class_Random_header.h"
#include <vector>
#include <cmath>
#include <math.h>

using namespace std;

class PathSimulatorEuler2F{

public:
PathSimulatorEuler2F(const Matrix& time_point,const Model2F& model);
PathSimulatorEuler2F(const PathSimulatorEuler2F& path_simulator);
~PathSimulatorEuler2F();

std::vector<Matrix> path();

private:
Matrix* m_time_points;
Model2F* m_model;

};
#endif