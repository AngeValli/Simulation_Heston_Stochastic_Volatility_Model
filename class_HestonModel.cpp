#include "class_HestonModel_header.h"


HestonModel::HestonModel(const double& risk_free_rate,const Pair& init_spot_variance, const double& correlation,const double& reversion_level,const double& reversion_speed,const double& vol_of_vol):Model2F(risk_free_rate,correlation,init_spot_variance),m_reversion_level(reversion_level),m_reversion_speed(reversion_speed),m_vol_of_vol(vol_of_vol){

}

HestonModel::HestonModel(const HestonModel& model):Model2F::Model2F(model),m_reversion_level(model.m_reversion_level),m_reversion_speed(model.m_reversion_speed),m_vol_of_vol(model.m_vol_of_vol){}

HestonModel& HestonModel::operator=(const HestonModel &model){
    // Definition of the equality operator for the Heston model (same reversion level, reversion speed and volatility of volatility)
    if(this!=&model){
        Model2F::operator=(model);
        m_reversion_level=model.m_reversion_level;
        m_reversion_speed=model.m_reversion_speed;
        m_vol_of_vol=model.m_vol_of_vol;

    }

    return(*this);

}

HestonModel* HestonModel::clone() const {
    // Method to clone a Heston model, ouput of the pointer
    HestonModel* ptr = new HestonModel(*this);
    return(ptr);
}

double HestonModel::get_reversion_level() const{
    // Method to get the reversion level of the model
    return(m_reversion_level);
}

double HestonModel::get_reversion_speed() const{
    // Method to get the reversion speed of the model
    return(m_reversion_speed);
}
double HestonModel::get_vol_of_vol() const{
    // Method to get the volatility of volatility of the model
    return(m_vol_of_vol);
}


double HestonModel::psi_function(const double &variance) const {
    // Method to define the psi function for Heston model
    return(std::sqrt(variance));
}

double HestonModel::sigma_function(const double &time, const double &spot) const{
    // Method to define the sigma function for Heston model
    return(1);

}

double HestonModel::variance_drift(const double &time, const double &variance) const{
    // Method to define the variance drift term for Heston model
    return(m_reversion_speed*(m_reversion_level-variance));
}

double HestonModel::variance_diffusion(const double &time, const double &variance) const{
    // Method to define the variance diffusion term for Heston model
    return(m_vol_of_vol*std::sqrt(variance));
}
