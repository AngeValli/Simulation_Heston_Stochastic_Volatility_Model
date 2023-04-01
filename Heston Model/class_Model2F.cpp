#include "class_Model2F_header.h"


Model2F::Model2F(const double& risk_free_rate,const double& correlation,const Pair init_spot_variance):m_risk_free_rate(risk_free_rate),m_correlation(correlation),m_init_spot_variance(init_spot_variance)
{

}

Model2F::Model2F(const Model2F& model):m_correlation(model.m_correlation),m_risk_free_rate(model.m_risk_free_rate),m_init_spot_variance(model.m_init_spot_variance)
{

}


Model2F& Model2F::operator=(const Model2F& model){
    // Definition of the equality operator between two models (so having same correlation, same initial spot variance and same risk free rate)
    if (this!=&model) {
        m_correlation=model.m_correlation;
        m_init_spot_variance=model.m_init_spot_variance;
        m_risk_free_rate=model.m_risk_free_rate;
    }
    return(*this);
}

double Model2F::get_risk_free_rate() const{
    // Method to get the risk free rate associated with the model
    return(this->m_risk_free_rate);
}
double Model2F::get_correlation() const{
    // Method to get the correlation associated with the model
    return(this->m_correlation);
}
Pair Model2F::get_init_spot_variance() const{
    // Method to get the initial spot variance associated with the model
    return this->m_init_spot_variance;
}



Pair Model2F::drift_pair(const double& time,const Pair& spot_variance) const{
    // Method to compute the drift pair
    Pair drift_pair=Pair();
    drift_pair.first=m_risk_free_rate*spot_variance.first;
    drift_pair.second=this->variance_drift(time,spot_variance.second);
    return drift_pair;
}

Pair Model2F::diffusion_pair(const double& time,const Pair& spot_variance) const{
    // Method to compute the diffusion pair
    Pair diffusion_pair=Pair();
    diffusion_pair.first=spot_variance.first*this->sigma_function(time,spot_variance.first)*this->psi_function(spot_variance.second);
    diffusion_pair.second=this->variance_diffusion(time,spot_variance.second);
    return diffusion_pair;
}