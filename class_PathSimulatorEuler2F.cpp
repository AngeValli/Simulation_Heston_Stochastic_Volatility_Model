#include "class_PathSimulatorEuler2F_header.h"


PathSimulatorEuler2F::PathSimulatorEuler2F(const Matrix& time_point,const Model2F& model):m_model(model.clone()),m_time_points(time_point.clone()){
}

PathSimulatorEuler2F::PathSimulatorEuler2F(const PathSimulatorEuler2F& path_simulator):m_model(path_simulator.m_model),m_time_points(path_simulator.m_time_points){
}

PathSimulatorEuler2F::~PathSimulatorEuler2F(){
    delete m_time_points;
    delete m_model;
}



vector<Matrix> PathSimulatorEuler2F::path(){
    // Method to compute the simulation of the 2 Factor model based on Euler schema and output the result
    int n_time_points=(*m_time_points).get_row();
    double model_correlation=(*m_model).get_correlation();
    double model_rate=(*m_model).get_risk_free_rate();
    Matrix spot_vector = Matrix(n_time_points,1);
    Matrix variance_vector = Matrix(n_time_points,1);
    gaussian_rv g=gaussian_rv();

    spot_vector.set_element(1,1,(*m_model).get_init_spot_variance().first);
    variance_vector.set_element(1,1,(*m_model).get_init_spot_variance().second);

    double working_spot;
    double working_variance;

    double backward_spot;
    double backward_variance;
    double backward_time_point;
    double time_delta;
    double g1;
    double g2;

    for(int j=2;j<=n_time_points;j++){
        backward_time_point=(*m_time_points)(j-1,1);
        backward_spot=spot_vector(j-1,1);
        backward_variance=variance_vector(j-1,1);
        time_delta=(*m_time_points)(j,1)-backward_time_point;
        g1=g.draw();
        g2=model_correlation*g1+sqrt(1-pow(model_correlation,2.0))*g.draw();

        working_spot=backward_spot*(1+model_rate*time_delta+(*m_model).sigma_function(backward_time_point,backward_spot)*(*m_model).psi_function(backward_variance)*sqrt(time_delta)*g1);
        working_variance=backward_variance+(*m_model).variance_drift(backward_time_point,backward_variance)*time_delta+(*m_model).variance_diffusion(backward_time_point,backward_variance)*sqrt(time_delta)*g2;

        spot_vector.set_element(j,1,working_spot);
        variance_vector.set_element(j,1,working_variance);

    }

    vector<Matrix> res_vector;

    res_vector.push_back(spot_vector);
    res_vector.push_back(variance_vector);

    return res_vector;



}

