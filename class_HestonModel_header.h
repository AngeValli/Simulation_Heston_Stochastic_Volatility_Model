#ifndef CLASSHESTONMODEL_H
#define CLASSHESTONMODEL_H
#include "class_Model2F_header.h"
#include <cmath>
class HestonModel : public Model2F{

    public:
    HestonModel(const double& risk_free_rate,const Pair& init_spot_variance, const double& correlation,const double& reversion_level,const double& reversion_speed,const double& vol_of_vol);
    HestonModel(const HestonModel& model);

    HestonModel& operator=(const HestonModel& model);

    HestonModel* clone() const override;

    double get_reversion_level() const;
    double get_reversion_speed() const;
    double get_vol_of_vol() const;

    double sigma_function(const double& time,const double& spot) const override;
    double psi_function(const double& variance) const override;
    double variance_drift(const double& time,const double& variance) const override;
    double variance_diffusion(const double& time,const double& variance)const override;



    private:
    double m_reversion_level;
    double m_reversion_speed;
    double m_vol_of_vol;



};
#endif