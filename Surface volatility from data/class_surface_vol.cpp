#include "class_surface_vol_header.h"

using namespace std;

surface_vol::surface_vol(const Matrix& tenor_vector,const vector<tenor_cubicspline> &vol_param):m_tenor_vector(tenor_vector){
    // Method to instantiate the surface volatility from tenor vector and volatility parameters vector
    for (int i=0;i<tenor_vector.get_row();i++){
        m_vol_param.push_back(vol_param[i]);
    }
}


surface_vol surface_vol::build_surface_vol_from_chain(const Matrix& option_chain){
    // Method to build surface volatility from chain
    std::vector<tenor_cubicspline> vol_param;
    Matrix tenor_vector=option_chain.get_sub_matrix(2,1,option_chain.get_row()-1,1);
    Matrix strike_layer=option_chain.get_row_vector(1);
    for(int i=2;i<=option_chain.get_row();i++){
        vol_param.push_back(tenor_cubicspline::build_tenor_cubicspline_from_chain(strike_layer.v_concat(option_chain.get_row_vector(i))));
    }
    return(surface_vol(tenor_vector,vol_param));
}

void surface_vol::describe_surface_vol(){
    // Method to describe surface volatility from strike and tenor
    cout<<"Strike card : "<<this->get_tenor_cubicspline(1).get_strike_vector().get_row()<<endl;
    cout<<"Strike vector :"<<endl;
    this->get_tenor_cubicspline(1).get_strike_vector().transpose().print_Matrix();
    cout<<"Tenor card : "<< m_tenor_vector.get_row()<<endl;
    cout<<"Tenor vector :"<<endl;
    m_tenor_vector.transpose().print_Matrix();

}

vector<tenor_cubicspline> surface_vol::get_vol_param(){
    // Method to get volatility parameter
    return(m_vol_param);
}

tenor_cubicspline surface_vol::get_tenor_cubicspline(const int& tenor_index){
    // Method to get tenor cubicspline parameter
    return(m_vol_param[tenor_index-1]);
}



Matrix surface_vol::get_tenor_vector(){
    // Method to get tenor vector
    return(m_tenor_vector);
}

int surface_vol::which_tenor_index(const double& tenor){
    // Method to derive tenor index
    int card_tenor=m_tenor_vector.get_row();
    if((tenor>m_tenor_vector.get_element(card_tenor,1)) || (tenor<m_tenor_vector.get_element(1,1))){
        throw std::invalid_argument("Error in method which_tenor_index, given tenor is out of range");
    }
    if (tenor==m_tenor_vector.get_element(card_tenor,1)) {
        return card_tenor;
    }
    int tenor_index=1;
    while(tenor>=m_tenor_vector.get_element(tenor_index,1)){
        tenor_index+=1;
    }
    return tenor_index-1;
}

double surface_vol::interpolate_implied_vol(const double &tenor, const double &strike,const double& rate){
    // Method to interpolate implied volatility from tenor, strike and rate
    if((tenor>m_tenor_vector.get_element(-1,1))||(tenor<m_tenor_vector.get_element(1,1))){
        throw std::invalid_argument("Error in method interpolate_implied_vol tenor is out of range");
    }

    int tenor_index=this->which_tenor_index(tenor);

    if(tenor_index==m_tenor_vector.get_row()){
        return(this->get_tenor_cubicspline(tenor_index).compute_implied_vol(strike));
    }

    double floor_tenor=m_tenor_vector.get_element(tenor_index,1);
    double ceil_tenor=m_tenor_vector.get_element(tenor_index+1,1);

    double floor_strike=strike*exp(rate*(floor_tenor-tenor));
    double ceil_strike=strike*exp(rate*(ceil_tenor-tenor));


    double floor_implied_vol=this->get_tenor_cubicspline(tenor_index).compute_implied_vol(floor_strike);
    double ceil_implied_vol=this->get_tenor_cubicspline(tenor_index+1).compute_implied_vol(ceil_strike);

    // cout<<floor_tenor<<endl;
    // cout<<ceil_tenor<<endl;
    // cout<<tenor<<endl;
    // cout<<floor_strike<<endl;
    // cout<<ceil_strike<<endl;
    // cout<<floor_implied_vol<<endl;
    // cout<<ceil_implied_vol<<endl;

    double implied_vol = sqrt((1/tenor)*((pow(floor_implied_vol,2.0)*floor_tenor)+(tenor-floor_tenor)*((pow(ceil_implied_vol,2.0)*ceil_tenor)-(pow(floor_implied_vol,2.0)*floor_tenor))/(ceil_tenor-floor_tenor)));

    return(implied_vol);

}

double surface_vol::compute_implied_vol(const double& tenor,const double& strike,const double& rate){
    // Method to compute implied volatility from tenor, strike and rate
    if(tenor<0 || strike<0){
        throw std::invalid_argument("Error in method surface_vol::compute_implied_vol strike or tenor is negative");

    }
    if (tenor<m_tenor_vector.get_element(1,1)){
        return(this->interpolate_implied_vol(m_tenor_vector.get_element(1,1),strike*exp(rate*(m_tenor_vector.get_element(1,1)-tenor)),rate));
    }
    if (tenor>m_tenor_vector.get_element(-1,1)){
        return(this->interpolate_implied_vol(m_tenor_vector.get_element(-1,1),strike*exp(rate*(m_tenor_vector.get_element(-1,1)-tenor)),rate));
    }
    return this->interpolate_implied_vol(tenor,strike,rate);
}

Matrix surface_vol::build_surface_vol(const double& tenor_start,const double& tenor_end,const int& tenor_card,const double& strike_start,const double& strike_end,const int& strike_card,const double& rate){
    // Method to build surface volatility from tenor start, end and cardinal, strike start, end and cardinal, and rate
    Matrix surface_vol=Matrix(tenor_card+1,strike_card+1);
    Matrix strike_range=Matrix::matrix_linspace(strike_start,strike_end,strike_card,true);
    Matrix tenor_range=Matrix::matrix_linspace(tenor_start,tenor_end,tenor_card,true);
    surface_vol.set_sub_matrix(2,1,tenor_range);
    surface_vol.set_sub_matrix(1,2,strike_range.transpose());
    for (int i=2;i<=surface_vol.get_row();i++) {
        for (int j=2;j<=surface_vol.get_col();j++) {
            surface_vol.set_element(i,j,this->compute_implied_vol(surface_vol.get_element(i,1),surface_vol.get_element(1,j),rate));
        }
    }
    return surface_vol;
}