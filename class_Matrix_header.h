#ifndef CLASSMATRIX_H
#define CLASSMATRIX_H
#include <cmath>
#include <vector>
#include <string>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

// ########################## MATRIX CLASS ###########################

class Matrix
{
    public:
    //Declaration constructeurs
    Matrix(); //Constructeur par defaut
    Matrix(const int &row,const int &col);
    Matrix(const int &row,const int &col,const std::vector<double> &value);//Constructeur principal
    Matrix(const Matrix &m); //Constructeur par recopie
    ~Matrix();//Destructeur
    Matrix* clone() const;


    //Methodes
    void assign_value(const Matrix &m);
    Matrix& operator=(const Matrix &m);
    Matrix operator+(const Matrix &m);
    Matrix operator*(const Matrix &m);
    Matrix operator^(const int &n);
    double operator()(const int& i, const int& j);
    bool isSquare();
    bool isSymmetric();
    bool isUpperTriangular();
    bool containNAN();
    double get_element(const int &row,const int &col) const ;
    void set_element(const int &row,const int &col,const double &element );
    int get_row() const ;
    int get_col() const ;
    std::vector<double> get_value();
    void print_Matrix() const;
    void write_matrix_in_file(std::string nom_fichier);
    static Matrix read_matrix_csv(const std::string &fname);
    static Matrix nul_matrix(const int &row,const int &col);
    static Matrix nul_matrix(const int &dim); // Square nul matrix
    static Matrix identity_matrix(const int &dim);
    static Matrix diagonal_matrix(const int &dim,const double& scale_param, const int& shift);
    static Matrix uniform_matrix(const int &row,const int &col,const double &value);
    static Matrix uniform_matrix(const int &dim,const double &value); // Square uniform matrix
    static Matrix base_matrix(const int& dim,const int& row,const int& col);
    static Matrix base_vector(const int& dim,const int& base_index);
    static Matrix permutation_matrix(const int& dim, const int& first_index,const int& second_index);
    static Matrix row_elimination_matrix(const int& dim,const int& pivot_index,const Matrix& elimation_vector);
    static Matrix matrix_linspace_mesh(const double& start,const double& end,const double& mesh,const bool& column_type);
    static Matrix matrix_linspace(const double& start,const double& end,const int& card,const bool& column_type);

    //static Matrix sample_standard_gaussian_vector(const int &row,const int &col);
    void set_sub_matrix(const int& idx_row,const int& idx_col,const Matrix& m);
    Matrix get_sub_matrix(const int& idx_row,const int& idx_col,const int& row_dim,const int& col_dim) const;
    Matrix get_diagonal(const int& shift);
    Matrix product(const Matrix &m);
    Matrix add(const Matrix &m);
    Matrix scale(const double &scale_parameter);
    Matrix transpose() const;
    Matrix cholesky();
    Matrix power_element_wise(const int &power_parameter);
    Matrix log_element_wise();
    Matrix exp_element_wise();
    Matrix power(const int &power_parameter);
    Matrix h_concat(const Matrix& m);
    Matrix v_concat(const Matrix& m);
    double mean();
    double dev();
    double trace();
    double norm();
    double dot_product(const Matrix &m);
    Matrix get_column_vector(const int &col) const;
    Matrix get_row_vector(const int &row) const;
    Matrix rot_90();
    int get_pivot_index(const int& pivot_step);
    Matrix get_elimination_vector(const int& pivot_step,const double& pivot_value);
    Matrix row_echelon_form(const bool& reduced);
    double det();
    int rank();
    Matrix inv(const double& margin_warning);
    static Matrix static_solve_thomas_system(Matrix const& a_vector,Matrix const& b_vector,Matrix const& c_vector,Matrix const& r_vector);


    //Attributs protected, protected members are like private members but it allows to use them in a derived class
    protected:

    const int m_row;
    const int m_col;
    double *m_value;

    /*
    Same specifications but with private members

    private:

    const int m_row;
    const int m_col;
    std::vector<double> m_value;



    */


};

#endif