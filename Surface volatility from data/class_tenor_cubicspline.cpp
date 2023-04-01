#include "class_tenor_cubicspline_header.h"

using namespace std;



tenor_cubicspline::tenor_cubicspline(const double& tenor,const Matrix& strike_vector,const Matrix& spline_param):m_tenor(tenor),m_strike_vector(strike_vector),m_spline_param(spline_param){

}

tenor_cubicspline::tenor_cubicspline(const tenor_cubicspline& cubicspline):m_tenor(cubicspline.m_tenor),m_strike_vector(cubicspline.m_strike_vector),m_spline_param(cubicspline.m_spline_param){

}


double tenor_cubicspline::get_tenor(){
    // Method to get the tenor of the cubicspline method
    return m_tenor;
}

int tenor_cubicspline::get_card_strike(){
    // Method to get the cardinal of the strike vector
    return m_strike_vector.get_row();
}

Matrix tenor_cubicspline::get_strike_vector(){
    // Method to get the strike vector
    return m_strike_vector;
}

Matrix tenor_cubicspline::get_spline_param(){
    // Method to get the the spline parameter
    return m_spline_param;
}

void tenor_cubicspline::describe_cubicspline(){
    // Method to print the description of the cubicspline  (characterized by its tenor, its strike and its cubicspline parameter)
    cout<<"tenor :"<<m_tenor<<endl;
    cout<<"strike cardinal :"<<m_strike_vector.get_row()<<endl;
    cout<<"strike vector :"<<endl;
    m_strike_vector.transpose().print_Matrix();
    cout<<"cubicspline param :"<<endl;
    m_spline_param.print_Matrix();

}

tenor_cubicspline tenor_cubicspline::build_tenor_cubicspline_from_chain(const Matrix& tenor_vol_chain){
    // Method to build the tenor cubicspline from chain
    double tenor=tenor_vol_chain.get_element(2,1);
    int card_strike=tenor_vol_chain.get_col()-1;

    Matrix strike_vector=tenor_vol_chain.get_sub_matrix(1,2,1,card_strike).transpose();

    Matrix vol_vector=tenor_vol_chain.get_sub_matrix(2,2,1,card_strike).transpose();

    Matrix a_vector = Matrix::nul_matrix(card_strike-3,1);
    Matrix b_vector = Matrix::nul_matrix(card_strike-2,1);
    Matrix c_vector = Matrix::nul_matrix(card_strike-3,1);
    Matrix r_vector = Matrix::nul_matrix(card_strike-2,1);


    for(int i=1;i<=a_vector.get_row();i++){
        a_vector.set_element(i, 1, strike_vector.get_element(i+2,1)-strike_vector.get_element(i+1,1));
    }
    for(int i=1;i<=b_vector.get_row();i++){
        b_vector.set_element(i, 1, 2*(strike_vector.get_element(i+1,1)-strike_vector.get_element(i,1)+strike_vector.get_element(i+2,1)-strike_vector.get_element(i+1,1)));
    }

    for(int i=1;i<=c_vector.get_row();i++){
        c_vector.set_element(i, 1, strike_vector.get_element(i+1,1)-strike_vector.get_element(i,1));
    }

    for(int i=1;i<=r_vector.get_row();i++){
        double r_value=3*(((vol_vector.get_element(i+2,1)-vol_vector.get_element(i+1,1))/(strike_vector.get_element(i+2,1)-strike_vector.get_element(i+1,1)))-((vol_vector.get_element(i+1,1)-vol_vector.get_element(i,1))/(strike_vector.get_element(i+1,1)-strike_vector.get_element(i,1))));
        r_vector.set_element(i, 1,r_value);
    }

    Matrix beta_vector=Matrix::nul_matrix(card_strike-1,1);
    beta_vector.set_sub_matrix(2,1,Matrix::static_solve_thomas_system(a_vector,b_vector,c_vector,r_vector));

    Matrix alpha_vector = Matrix::nul_matrix(card_strike-1,1);
    alpha_vector.set_element(card_strike-1,1, -(beta_vector.get_element(card_strike-1,1))/(3*((vol_vector.get_element(card_strike,1)-vol_vector.get_element(card_strike-1,1)))));

    for(int i=1;i<alpha_vector.get_row();i++){
        alpha_vector.set_element(i,1, (beta_vector.get_element(i+1,1)-beta_vector.get_element(i,1))/(3*(strike_vector.get_element(i+1,1)-strike_vector.get_element(i,1))));
    }
    Matrix gamma_vector=Matrix::nul_matrix(card_strike-1,1);
    for (int i=1;i<=gamma_vector.get_row();i++) {
        double delta_x=(strike_vector.get_element(i+1,1)-strike_vector.get_element(i,1));
        double gamma_value=((vol_vector.get_element(i+1,1)-vol_vector.get_element(i,1)))/(delta_x)-alpha_vector.get_element(i,1)*pow(delta_x,2.0)-beta_vector.get_element(i,1)*delta_x;
        gamma_vector.set_element(i, 1,gamma_value);
    }

    Matrix delta_vector = vol_vector.get_sub_matrix(1,1, card_strike-1,1);

    Matrix spline_param = alpha_vector.h_concat(beta_vector).h_concat(gamma_vector).h_concat(delta_vector);


    return(tenor_cubicspline(tenor,strike_vector,spline_param));
}

int tenor_cubicspline::which_strike_index(const double& strike){
    // Method to get the strike index
    int card_strike = m_strike_vector.get_row();

    if((strike>m_strike_vector.get_element(card_strike,1)) || (strike<m_strike_vector.get_element(1,1))){
        throw std::invalid_argument("Error in method which_strike_index, given strike is out of range");
    }
    if (strike==m_strike_vector.get_element(card_strike,1)) {
    return card_strike-1;
    }
    int strike_index=1;
    while(strike>=m_strike_vector.get_element(strike_index,1)){
        strike_index+=1;
    }
    return strike_index-1;

}

double tenor_cubicspline::interpolate_vol(const double &strike){
    // Method to interpolate the volatility and derive implied volatility
    if((strike>m_strike_vector.get_element(-1,1)) || (strike<m_strike_vector.get_element(1,1))){
        throw std::invalid_argument("Error in method interpolate, given strike is out of range");
    }

    int strike_index = this->which_strike_index(strike);
    Matrix strike_spline_param=m_spline_param.get_row_vector(strike_index);
    double implied_vol=0;
    for(int j=1;j<=4;j++){
        implied_vol+=strike_spline_param.get_element(1,j)*pow(strike-m_strike_vector.get_element(strike_index,1),4.0-j);
    }

    return(implied_vol);


}

double tenor_cubicspline::compute_implied_vol(const double &strike){
    // Method to compute the implied volatility
    if(strike<0){
        throw std::invalid_argument("Error in method compute_implied vol, strike is negative");
    }

    if (strike>m_strike_vector.get_element(-1,1)) {
        double left_derivative = 3*m_spline_param.get_element(-1,1)*pow(m_strike_vector.get_element(-1,1)-m_strike_vector.get_element(-2,1),2.0)+2*m_spline_param.get_element(-1,2)*(m_strike_vector.get_element(-1,1)-m_strike_vector.get_element(-2,1))+m_spline_param.get_element(-1,3);
        return(this->interpolate_vol(m_strike_vector.get_element(-1,1))+left_derivative*(strike-m_strike_vector.get_element(-1,1)));
    }
    if (strike<m_strike_vector.get_element(1,1)) {
        return(this->interpolate_vol(m_strike_vector.get_element(1,1))+(m_spline_param.get_element(1,3)*(strike-m_strike_vector.get_element(1,1))));
    }

    return(this->interpolate_vol(strike));
}

Matrix tenor_cubicspline::build_implied_vol_curve(const double& strike_mesh){
    // Method to build the implied volatility curve from mesh
    int card_strike = m_strike_vector.get_row();
    Matrix strike_linspace=Matrix::matrix_linspace_mesh(m_strike_vector.get_element(1,1),m_strike_vector.get_element(card_strike,1),strike_mesh,true);
    Matrix implied_vol_linspace=Matrix::nul_matrix(strike_linspace.get_row(),1);
    for(int i=1;i<=strike_linspace.get_row();i++){
        implied_vol_linspace.set_element(i,1,this->compute_implied_vol(strike_linspace.get_element(i,1)));
    }

    return strike_linspace.h_concat(implied_vol_linspace);
}

Matrix tenor_cubicspline::build_implied_vol_curve(const double& strike_start,const double& strike_end,const int& card){
    // Method to build the implied volatility curve from strike start and end
    Matrix strike_linspace=Matrix::matrix_linspace(strike_start,strike_end,card,true);
    Matrix implied_vol_linspace=Matrix::nul_matrix(strike_linspace.get_row(),1);
    for(int i=1;i<=strike_linspace.get_row();i++){
        implied_vol_linspace.set_element(i,1,this->compute_implied_vol(strike_linspace.get_element(i,1)));
    }

    return strike_linspace.h_concat(implied_vol_linspace);
}