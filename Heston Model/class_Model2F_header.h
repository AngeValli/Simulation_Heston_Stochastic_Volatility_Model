#ifndef CLASSMODEL2F_H
#define CLASSMODEL2F_H
#include <utility>
using Pair=std::pair<double,double>;

class Model2F {

    public:
    Model2F(const double& risk_free_rate,const double& correlation,const Pair init_spot_variance);
    Model2F(const Model2F& model);
    virtual ~Model2F() {};

    Model2F& operator=(const Model2F& model);


    double get_risk_free_rate() const;
    double get_correlation() const;
    Pair get_init_spot_variance() const;

    virtual Model2F* clone() const =0;

    virtual double sigma_function(const double& time,const double& spot) const =0;
    virtual double psi_function(const double& variance) const =0;
    virtual double variance_drift(const double& time,const double& variance) const =0;
    virtual double variance_diffusion(const double& time,const double& variance)const =0;

    Pair drift_pair(const double& time,const Pair& spot_variance) const;
    Pair diffusion_pair(const double& time,const Pair& spot_variance) const;









    protected:
    double m_risk_free_rate;
    double m_correlation;
    Pair m_init_spot_variance;



};

#endif
