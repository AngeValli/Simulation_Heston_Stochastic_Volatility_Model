#include "class_Matrix_header.h"


using namespace std;

// ########################## MATRIX CLASS ###########################

// Constructors & co

Matrix::Matrix() :m_row(),m_col()
{
    m_value=new double;
}

Matrix::Matrix(const int &row,const int &col):m_row(row),m_col(col)
{
    m_value=new double[row*col];

    for (int i=0; i<row*col; i++) {
        m_value[i]=0;
    }
}

Matrix::Matrix(const int &row,const int &col,const std::vector<double> &value):m_row(row),m_col(col)
{
    m_value=new double[row*col];

    for (int i=0; i<row*col; i++) {
        m_value[i]=value[i];
    }
}

Matrix::~Matrix(){

    delete [] m_value;

}

Matrix::Matrix(const Matrix &m): m_row(m.m_row),m_col(m.m_col){

    m_value=new double[m_row*m_col];

    for (int i=0; i<m_row*m_col; i++) {
        m_value[i]=m.m_value[i];
    }
}

Matrix* Matrix::clone() const{
    // Returns a pointer on a copy of this (*this), the clone must be deleted after use to delete the copy 
    return(new Matrix(*this));

}

//Principal Methods

void  Matrix::assign_value(const Matrix &m){
    // Method to assign a value in the Matrix
    if (m_col != m.m_col||m_row!= m.m_row ){
    throw std::invalid_argument( "Non comfortable arguments for matrix assignation");
    }

    for (int i=0; i<m_row*m_col; i++) {
        m_value[i]=m.m_value[i];
    }



}

Matrix& Matrix::operator=(const Matrix &m){
    if(this != &m){
    for (int i=0; i<m_row*m_col; i++) {
        m_value[i]=m.m_value[i];
    }

    }

    return(*this);
}

double Matrix::operator()(const int &i, const int &j){
    // Definition of the operator of the matrix to get access to a specific element (i,j) of the matrix via method get_element(i,j)
    return this->get_element(i,j);
}

bool Matrix::isSquare(){
    // Method to verify if the matrix is a square matrix
    return(bool (m_col == m_row));
}

bool Matrix::isUpperTriangular(){
    // Method to verify if the matrix is upper triangular
    for(int i=2;i<=m_row;i+=1){
        for(int j=1;j<i;j+=1){
            if(this->get_element(i,j)!=0){
                return(false);
            }
        }
    }
    return(true);
}

bool Matrix::isSymmetric(){
    // Method to verify if the square matrix is symmetric
    if(!(this->isSquare())){
        throw std::invalid_argument( "Matrix is not square, isSymmetric has no sense");
    }
    bool symmetric=true;
    for (int i = 1;i<=m_row; i++) {
    for (int j=i; j<=m_col; j++) {
    symmetric=(this->get_element(i,j)==this->get_element(j,i));
    if(symmetric==false){
        return(symmetric);
    }

    }
    }

    return(symmetric);
}

bool Matrix::containNAN(){
    // Method to verify if the matrix contains empty NaN value
    bool containNAN=false;
    for (int i = 1;i<=m_row; i++) {
    for (int j=1; j<=m_col; j++) {
    containNAN=isnan(this->get_element(i,j));
    if(containNAN==true){
        return(containNAN);
    }

    }
    }

    return(containNAN);
}


int Matrix::get_row() const{
    // Method to get a specific row of the matrix
    return(m_row);
}

int Matrix::get_col() const{
    // Method to get a specific column of the matrix
    return(m_col);
}



double Matrix::get_element(const int &row, const int &col) const{
    // Method to get access to a specific element (i,j) of the matrix
    int int_row=(row>=1)*row+(row<=-1)*(this->get_row()+row+1);
    int int_col=(col>=1)*col+(col<=-1)*(this->get_col()+col+1);

    if (int_row>m_row || int_col>m_col){
        throw std::invalid_argument("Error in method get_element given row or column index are out of range");
    }


    return(m_value[((int_row-1)*m_col)+int_col-1]);

}


void Matrix::set_element(const int &row,const int &col,const double &element ){
    // Method to define an element in the matrix
    int int_row=(row>=1)*row+(row<=-1)*(this->get_row()+row+1);
    int int_col=(col>=1)*col+(col<=-1)*(this->get_col()+col+1);

    if (int_row>m_row || int_col>m_col){
        throw std::invalid_argument("Error in method set_element given row or column index are out of range");
    }


    m_value[((int_row-1)*m_col)+int_col-1]=element;


}

void Matrix::print_Matrix() const{
  // Method to print the content of the matrix
    for (int i = 1; i <= m_row; i++) {
        for (int j = 1; j <= m_col; j++) {
            std::cout << this->get_element(i, j) << " " ;
    }
    std::cout << std::endl;

    }
    std::cout << std::endl;
}


void Matrix::write_matrix_in_file(std::string nom_fichier){
  // Method to output the content of the matrix in a file
    ofstream monFlux(nom_fichier);
    if(monFlux){
        for (int i = 1; i <= m_row; i++) {
            monFlux<<endl;
        for (int j = 1; j <= m_col; j++) {
            monFlux << this->get_element(i, j)<<",";
        }
    }
    }else{
        cout<<"Error file writing"<<endl;
    }
}

Matrix Matrix::read_matrix_csv(const string &fname){
    // Method to instantiate a matrix based on the content of a csv file
    vector<vector<string>> content;
	vector<string> row;
	string line, word;

	fstream file (fname, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();

			stringstream str(line);

			while(getline(str, word, ','))
				row.push_back(word);
			content.push_back(row);
		}
	}
	else
		cout<<"Could not open the file\n";


    Matrix res_mat=Matrix::nul_matrix(content.size(),content[0].size());

    for(int i=0;i<content.size();i++)
    {
        for (int j=0; j<content[i].size(); j++) {
            res_mat.set_element(i+1,j+1,std::stod(content[i][j]));
        }
    }
    return res_mat;
}

Matrix Matrix::get_column_vector(const int &col) const{
    // Method to get the vector of a specific column of the matrix
    if (col>m_col){
        throw std::invalid_argument("Error in method get_column_vector given column index is out of range");
    }

    Matrix column_vector=Matrix::nul_matrix(m_row,1);
    for (int k=1; k<=m_row;k++) {
        column_vector.set_element(k,1,this->get_element(k,col));
    }

    return(column_vector);
}
Matrix Matrix::get_row_vector(const int &row) const{
    // Method to get the vector of a specific row of the matrix
    if (row>m_row){
        throw std::invalid_argument("Error in method get_row_vector given row index is out of range");
    }
    Matrix row_vector=Matrix::nul_matrix(1,m_col);
    for (int k=1; k<=m_col;k++) {
        row_vector.set_element(1,k,this->get_element(row,k));
    }

    return(row_vector);
}


//Static methods

Matrix Matrix::nul_matrix(const int &row,const int &col){
    // Method to instantiate a nul matrix based on row and column size (unsquared matrix)
    std::vector<double> nul_value(row*col,0);
    Matrix nul_matrix=Matrix(row,col,nul_value);
    return(nul_matrix);
}
Matrix Matrix::nul_matrix(const int &dim){
    // Method to instantiate a nul matrix based on dimension of squared matrix
    std::vector<double> nul_value(pow(dim,2.0),0);
    Matrix nul_matrix=Matrix(dim,dim,nul_value);
    return(nul_matrix);
}

Matrix Matrix::identity_matrix(const int &dim){
    // Method to instantiate an identity squared matrix based on dimension
    Matrix mat=Matrix::nul_matrix(dim,dim);
    for (int i=1; i<=dim; i++) {
        mat.set_element(i,i,1.0);
    }
    return(mat);

}

Matrix Matrix::uniform_matrix(const int &row,const int &col,const double &value){
    // Method to instantiate a uniform matrix (having one value) based on row and column size (unsquared matrix)
    std::vector<double> unif_value(row*col,value);
    Matrix unif_matrix=Matrix(row,col,unif_value);
    return(unif_matrix);

}
Matrix Matrix::uniform_matrix(const int &dim,const double &value){
    // Method to instantiate a uniform matrix (having one value) based on dimension of squared matrix
    std::vector<double> unif_value(pow(dim,2.0),value);
    Matrix unif_matrix=Matrix(dim,dim,unif_value);
    return(unif_matrix);

}

Matrix Matrix::diagonal_matrix(const int &dim,const double& scale_param, const int& shift=0){
    // Method to instantiate a diagonal matrix based on dimension of squared matrix, with value scale_parameter on the dim - shift first rows and value 0 otherwise
    if (shift>dim){
        throw std::invalid_argument("Error in static method diagonal_matrix given shift is out of range");
    }
    Matrix mat=Matrix::nul_matrix(dim,dim);

    for (int i=1; i<=dim-abs(shift); i++) {
            mat.set_element(i+fmax(0,-shift),i+fmax(0,shift),scale_param);

    }
    return mat;
}

Matrix Matrix::get_diagonal(const int &shift=0){
    // Method which returns a vector of the dim-shift first elements of the diagonal of a squared matrix
    if (!this->isSquare()){
        throw std::invalid_argument("Error in method get_diagonal, Matrix is not square");
    }
    if (m_row-abs(shift)<=0) {
        throw std::invalid_argument("Error in method get_diagonal, shift is larger than m_row");
    }
    Matrix res_vect = Matrix::nul_matrix(m_row-abs(shift),1);
    for (int i=1; i<=m_row-abs(shift); i++) {
            res_vect.set_element(i,1, this->get_element(i+fmax(0,-shift),i+fmax(0,shift)));

    }
    return res_vect;

}

Matrix Matrix::base_matrix(const int& dim,const int& row,const int& col){
    // Method to instantiate a null matrix except on one coordinate (row, col) having value 1, based on dimension of squared matrix
        if (col>dim || row>dim){
        throw std::invalid_argument("Error in static method base_matrix given dim is out of range");
    }
    Matrix unit_mat=nul_matrix(dim);
    unit_mat.set_element(row,col,1.0);
    return(unit_mat);
}

Matrix Matrix::base_vector(const int& dim,const int& base_index){
    // Method to instantiate a null vector except on one coordinate base_index having value 1, based on length dim
    if (base_index>dim){
        throw std::invalid_argument("Error in method get_column_vector given column index is out of range");
    }
    Matrix base_vect = Matrix::nul_matrix(dim,1);
    base_vect.set_element(base_index,1,1.0);
    return base_vect;
}

// Maths methods

Matrix Matrix::product(const Matrix &m){
    // Method to compute a matrix multiplication
    if (m_col!= m.m_row){
        throw std::invalid_argument( "Non comfortable arguments for matrix product");
        }
    Matrix result_matrix=Matrix::nul_matrix(m_row, m.m_col);
    for(int i=1;i<=m_row;i++){
        for(int j=1;j<=m.m_col;j++){
            double s = 0.0;
            for (int k=1; k<=m_col; k++) {
                s=s+this->get_element(i,k)*m.get_element(k,j);
            }
            result_matrix.set_element(i,j,s);
        }
    }

    return(result_matrix);
}

Matrix Matrix::operator*(const Matrix &m){
    // Method to compute the matrix multiplication operator
    return(this->product(m));
}

Matrix Matrix::add(const Matrix &m){
    // Method which returns the sum of the two input matrices having the same dimensions
    if (m_col != m.m_col||m_row != m.m_row){
        throw std::invalid_argument( "Non comfortable arguments for matrix addition");
        }
    Matrix result_matrix=Matrix::nul_matrix(m_row,m_col);
    for(int i=1;i<=m_row;i++){
        for(int j=1;j<=m_col;j++){
            result_matrix.set_element(i,j,this->get_element(i, j)+m.get_element(i, j));
        }
    }
    return(result_matrix);
}

Matrix Matrix::operator+(const Matrix &m){
    // Method to compute the matrix sum operator
    return(this->add(m));
}

Matrix Matrix::matrix_linspace_mesh(const double& start,const double& end,const double& mesh,const bool& column_type=false){
    // Method which returns a matrix with linearly spaced values between values start and end, with space mesh. Argument column_type is a boolean to perform on rows or columns.

    // Check on inputs arguments
    double ind = floor((end-start)/mesh)-((end-start)/mesh);
    if(ind!=0){
        throw std::invalid_argument("Error in method matrix_linspace, meshing can't be performed due to bad arguments");
    }
    int n = floor((end-start)/mesh);
    Matrix res_mat=Matrix::nul_matrix(column_type*n+1,(!column_type)*n+1);
    if(column_type){
        for(int i=0;i<=n;i++){
            res_mat.set_element(i+1,1, start+mesh*i);
        }
    } else {
        for(int i=0;i<=n;i++){
            res_mat.set_element(1,i+1, start+mesh*i);
        }

    }
    return res_mat;
}

Matrix Matrix::matrix_linspace(const double& start,const double& end,const int& card,const bool& column_type){
    // Method which returns a matrix with linearly spaced values between values start and end, with space mesh computed from cardinal card. Argument column_type is a boolean to perform on rows or columns.

    Matrix res_mat=Matrix::nul_matrix(column_type*card+(!column_type),(!column_type)*card+column_type);

    double mesh = (end-start)/(card-1);

    if(column_type){
        for(int i=0;i<card;i++){
            res_mat.set_element(i+1,1, start+mesh*i);
        }
    } else {
        for(int i=0;i<card;i++){
            res_mat.set_element(1,i+1, start+mesh*i);
        }
    }
    return res_mat;
}


Matrix Matrix::scale(const double &scale_parameter){
    // Method which returns a matrix with values scaled by scale_parameter.
    Matrix result_matrix=Matrix::nul_matrix(m_row,m_col);
    for(int i=1;i<=m_row;i++){
        for(int j=1;j<=m_col;j++){
            result_matrix.set_element(i,j,scale_parameter*(this->get_element(i, j)));
        }
    }
    return(result_matrix);
}

Matrix Matrix::transpose() const{
    // Method to compute transposition of a matrix
    Matrix trans_mat=Matrix::nul_matrix(m_col,m_row);
    for (int i=1; i<=m_col; i++) {
        for (int j=1; j<=m_row; j++) {
            trans_mat.set_element(i, j, this->get_element(j,i));
        }
    }

    return(trans_mat);
}

Matrix Matrix::cholesky(){
    // Method to compute cholesky decomposition of a matrix, based on sufficient regularity of the input matrix (squared, symmetric and positive)
    if (!(this->isSquare())){
        throw std::invalid_argument( "Non comfortable arguments for cholesky decomposition, matrix is not square");
    }
    if (!(this->isSymmetric())){
        throw std::invalid_argument( "Non comfortable arguments for cholesky decomposition, matrix is not symmetric");
    }

    Matrix cholesky_mat=Matrix::nul_matrix(m_row, m_col);
    for (int index=1;index<=m_col;index++){

        double s=0.0;

        for(int k=1;k<index;k++){
            s=s+pow(cholesky_mat.get_element(index, k),2.0);
        }

        cholesky_mat.set_element(index,index,sqrt(this->get_element(index, index)-s));

        for(int j=index+1;j<=m_col;j++){

            double s=0.0;

            for(int k=1;k<index;k++){
                s=s+cholesky_mat.get_element(index, k)*cholesky_mat.get_element(j, k);
            }

            cholesky_mat.set_element(j,index, (this->get_element(index,j)-s)/cholesky_mat.get_element(index,index));

        }


    }
    if (cholesky_mat.containNAN()) {
    cout<<( "#####WARNING######: Error in computing Cholesky decomposition: result matrix contains NaN, argument matrix might not be positive nor definite");
    }
    return(cholesky_mat);
}

Matrix Matrix::power_element_wise(const int &power_parameter){
    // Method to compute the element-wise power of a matrix values
    Matrix result_matrix=Matrix::nul_matrix(m_row, m_col);
    for(int i=1;i<=m_row;i++){
        for(int j=1;j<=m_col;j++){

            result_matrix.set_element(i,j,pow(this->get_element(i, j),power_parameter));

        }
    }

    return(result_matrix);
}

Matrix Matrix::power(const int &power_parameter){
    // Method to compute the power multiplication of a matrix
    Matrix result_matrix=Matrix(*this);
    for(int k=2;k<=power_parameter;k++){
        result_matrix=(*this)*result_matrix;
    }

    return(result_matrix);
}

Matrix Matrix::operator^(const int &n){
    // Method to compute the matrix power operation
    return(this->power(n));
}

Matrix Matrix::log_element_wise(){
    // Method to compute the element-wise logarithm of a matrix values
    Matrix result_matrix=Matrix::nul_matrix(m_row, m_col);
    for(int i=1;i<=m_row;i++){
        for(int j=1;j<=m_col;j++){

            result_matrix.set_element(i,j,log(this->get_element(i, j)));

        }
    }

    return(result_matrix);
}

Matrix Matrix::exp_element_wise(){
    // Method to compute the element-wise exponential of a matrix values
    Matrix result_matrix=Matrix::nul_matrix(m_row, m_col);
    for(int i=1;i<=m_row;i++){
        for(int j=1;j<=m_col;j++){

            result_matrix.set_element(i,j,exp(this->get_element(i, j)));

        }
    }

    return(result_matrix);
}

double Matrix::mean(){
    // Method to compute the mean of all values of a matrix
    double somme=0.0;
    for(int i=1;i<=m_row;i++){
	for(int j=1;j<=m_col;j++){
		somme=somme+(this->get_element(i,j));
		}
    }
    double mass = (m_col)*(m_row);
    return(somme/mass);
}

double Matrix::dev(){
    // Method to compute the standard deviation of all values of a matrix
    Matrix quadratic_mat=*this;
    quadratic_mat.assign_value(quadratic_mat.add(Matrix::uniform_matrix(m_row, m_col, this->mean()).scale(-1.0)));
    quadratic_mat.assign_value(quadratic_mat.power_element_wise(2));
    double n = (m_col)*(m_row);
    return((n/(n-1))*sqrt(quadratic_mat.mean()));
}

double Matrix::trace(){
    // Method to compute the trace of a matrix
    double somme=0.0;
    for (int k=1;k<=m_row;k++) {
        somme=somme+this->get_element(k,k);
    }

    return(somme);
}

double Matrix::norm(){
    // Method to compute the Euclidean norm of a matrix
    return(sqrt(this->transpose().product(*this).trace()));
}



double Matrix::dot_product(const Matrix &m){
    // Method to compute the dot product between two vectors (represented as unicolumn matrix)
    if (m_col!= 1 || m.m_col!= 1 || m_row != m.m_row){
        throw std::invalid_argument( "Non comfortable arguments for dot product");
    }

    Matrix mat_value= (this->transpose()).product(m);

    return(mat_value.get_element(1,1));

}

void Matrix::set_sub_matrix(const int& idx_row,const int& idx_col,const Matrix& m){
    // Method to set a matrix m as the submatrix starting from coordinates (idx_row, idx_col)
    if(m.m_col > m_col || m.m_row > m_row ){
        throw std::invalid_argument( "Non comfortable arguments in set_sub_matrix");
    }
    for(int i = 1;i<= m.m_row;i++){
        for (int j = 1; j<= m.m_col; j++) {
        this->set_element(i+idx_row-1,j+idx_col-1,m.get_element(i,j));
        }
    }
}

Matrix Matrix::h_concat(const Matrix& m){
    // Method to concatenate the columns of a matrix m to the original matrix
    if (m_row!=m.m_row) {
    throw std::invalid_argument( "Non comfortable arguments in h_concat, row sizes do not match");
    }
    Matrix result = Matrix::nul_matrix(m_row,m_col+m.m_col);
    result.set_sub_matrix(1,1,*this);
    result.set_sub_matrix(1,m_col+1, m);
    return result;
}
Matrix Matrix::v_concat(const Matrix& m){
    // Method to concatenate the rows of a matrix m to the original matrix
    if (m_col!=m.m_col) {
    throw std::invalid_argument( "Non comfortable arguments in h_concat, col sizes do not match");
    }
    Matrix result = Matrix::nul_matrix(m_row+m.m_row,m_col);
    result.set_sub_matrix(1,1,*this);
    result.set_sub_matrix(m_row+1,1,m);
    return result;
}

Matrix Matrix::permutation_matrix(const int& dim, const int& first_index,const int& second_index){
    // Method to define a permutation matrix based on rows indexes to exchange
    if (first_index>dim || second_index>dim){
        throw std::invalid_argument("Error in static method permutation_matrix first_index or second_index is out of range");
    }
    Matrix res_mat = Matrix::identity_matrix(dim);
    res_mat.set_element(first_index,second_index,1.0);
    res_mat.set_element(first_index,first_index,0);
    res_mat.set_element(second_index,first_index,1.0);
    res_mat.set_element(second_index,second_index,0);
    return(res_mat);
}

Matrix Matrix::row_elimination_matrix(const int& dim,const int& pivot_index,const Matrix& elimination_vector){
    // Method to compute a Gaussian elimination (or row reduction) to solve a system of linear equations
    if(pivot_index>dim){
        throw std::invalid_argument("Error in static method row_elimination_matrix given pivot_index is out of range");
    }
    if(elimination_vector.get_row()!=(dim-pivot_index)){
        throw std::invalid_argument("Error in static method row_elimination_matrix dimension of elimination_vector doesn't fit");
    }

    Matrix res_mat = Matrix::identity_matrix(dim);
    for(int i=1;i<=elimination_vector.get_row();i+=1){
        res_mat.set_element(i+pivot_index,pivot_index,elimination_vector.get_element(i,1));
    }

    return(res_mat);
}

int Matrix::get_pivot_index(const int& pivot_step)
    // Method to get pivot index based on pivot step
{
    double current_max_value=-1;
    double current_max_index=-1;
    for(int i=pivot_step;i<=m_row;i+=1)
    {
        if ( (abs(this->get_element(i,pivot_step))>fmax(current_max_value,0)))
        {
            current_max_value=this->get_element(i,pivot_step);
            current_max_index=i;
        }
    }
    return(current_max_index);
}

Matrix Matrix::get_elimination_vector(const int& pivot_step,const double& pivot_value){
    // Method to retrieve elimination vector based on current pivot step and pivot value
    Matrix res_vect = Matrix::nul_matrix(m_row-pivot_step,1);
    for(int i=1;i<=m_row-pivot_step;i+=1){
        res_vect.set_element(i,1, -(this->get_element(i+pivot_step,pivot_step))/pivot_value);

    }
    return(res_vect);
}

Matrix Matrix::row_echelon_form(const bool& reduced=false) {
    // Method to transform a matrix to its row echelon form (or reduced row echelon form)
    Matrix working_matrix=*this;
    int pivot_index=0;
    double pivot_value=0;
    for (int j=1;j<=m_col;j+=1){
        pivot_index=working_matrix.get_pivot_index(j);
        if(pivot_index!=-1){
            pivot_value=working_matrix.get_element(pivot_index,j);
            if(pivot_index!=j){
                working_matrix=permutation_matrix(m_row, pivot_index,j)*working_matrix;
            }
            Matrix elimination_vector = working_matrix.get_elimination_vector(j, pivot_value);
            working_matrix=row_elimination_matrix(m_row,j, elimination_vector)*working_matrix;
            if(reduced==true){
                Matrix reduce_mat=Matrix::identity_matrix(m_row);
                reduce_mat.set_element(j, j, 1/pivot_value);
                working_matrix=reduce_mat*working_matrix;
            }

        }
    }
    return(working_matrix);
}

double Matrix::det(){
    // Method to compute the determinant of a matrix by deriving it from its row echelon form (except for upper triangular matrix)
    if (!(this->isSquare())) {
        throw std::invalid_argument("Error in det method, matrix is not square determinant can't be computed");
    }
    if(this->isUpperTriangular()){
        double det = 1.0;
        for(int i=1;i<=m_row;i+=1){
            det*=this->get_element(i,i);
        }
        return(det);

    }

    int compt_permutation=0;
    Matrix working_matrix=*this;
    int pivot_index=0;
    double pivot_value=0;
    for (int j=1;j<=m_col;j+=1){
        pivot_index=working_matrix.get_pivot_index(j);
        if(pivot_index!=-1){
            pivot_value=working_matrix.get_element(pivot_index,j);
            if(pivot_index!=j){
                working_matrix=permutation_matrix(m_row, pivot_index,j)*working_matrix;
                compt_permutation+=1;
            }
            Matrix elimination_vector = working_matrix.get_elimination_vector(j, pivot_value);
            working_matrix=row_elimination_matrix(m_row,j, elimination_vector)*working_matrix;

        }
    }

    double det = std::pow(-1,compt_permutation);
    for(int i=1;i<=m_row;i+=1){
        det*=working_matrix.get_element(i,i);
    }
    return(det);
}

int Matrix::rank(){
    // Method to retrieve rank of a matrix based on its row echelon form
    Matrix row_echelon_matrix = this->row_echelon_form();
    int rank=0;
    for(int i=1;i<=m_row;i+=1){
        int non_nul_pivot=false;
        for(int j=i;j<=m_row;j+=1){
            non_nul_pivot+=(abs(row_echelon_matrix.get_element(i,j))>0);
        }
        rank+=(non_nul_pivot>0);

    }
    return(rank);

}

Matrix Matrix::rot_90(){
    // Method to rotate a matrix by 90°
    if(!this->isSquare()){
        throw std::invalid_argument("Error in method rot_90 cannot perform rotation on a non square matrix");
    }
    Matrix res_mat = Matrix::nul_matrix(m_row);
    for(int i=1;i<=m_row;i+=1){
        for(int j=1;j<=m_col;j+=1){
            res_mat.set_element(i,j,this->get_element(j, this->get_col()-i+1));
        }
    }
    return res_mat;
}

Matrix Matrix::get_sub_matrix(const int& idx_row,const int& idx_col,const int& row_dim,const int& col_dim) const{
    // Method to get a sub matrix from a specified position (idx_row,idx_col).
    if(idx_row+row_dim-1>m_row || col_dim+idx_col-1>m_col){
        throw std::invalid_argument("Error in method get_sub_matrix, index are out of range");
    }
    Matrix res_mat=nul_matrix(row_dim,col_dim);
    for (int i=1; i<=row_dim; i+=1) {
        for(int j=1;j<=col_dim;j+=1){
            res_mat.set_element(i,j,this->get_element(idx_row+i-1, idx_col+j-1));
        }
    }
    return res_mat;

}


Matrix Matrix::inv(const double& margin_warning=0.5){
    // Method to invert a matrix based on its reduced row echelon form (throw a warning if the matrix determinant is inferior to a certain margin, for stability purposes)
    if(!this->isSquare()){
        throw std::invalid_argument("Error in inv method, matrix is not square");
    }

    Matrix working_matrix=this->h_concat(Matrix::identity_matrix(m_row));
    working_matrix=working_matrix.row_echelon_form(true);
    double det = working_matrix.get_sub_matrix(1,1, m_row,m_col).det();
    if(det==0){
        throw std::invalid_argument("Error in inv method, determinant of matrix is 0 can't invert matrix");
    }
    if(det<margin_warning){
        cout<<"########### WARNING ########### In method inv: matrix determinant is near 0, inverse can be instable"<<endl;
    }
    for(int i=1;i<=m_row;i+=1){
        for(int j=i+1;j<=m_col;j+=1){
            if(abs(working_matrix.get_element(i,j))>0){
                Matrix elimination_matrix=Matrix::identity_matrix(m_row);
                elimination_matrix.set_element(i,j,-working_matrix.get_element(i,j));
                working_matrix=elimination_matrix*working_matrix;

            }
        }
    }


    return(working_matrix.get_sub_matrix(1,m_col+1, m_row, m_col));
}

Matrix Matrix::static_solve_thomas_system(Matrix const& a_vector,Matrix const& b_vector,Matrix const& c_vector,Matrix const& r_vector){
    // Method to perform a Thomas algorithm (tridiagonal matrix algorithm) to solve tridiagonal system of equations
    int n=b_vector.m_row;
    if(a_vector.m_col != 1 || b_vector.m_col != 1 || c_vector.m_col != 1 || r_vector.m_col != 1){
        throw std::invalid_argument("Error in method solve_thomas_system method, given arguments are not vectors");

    }
    if(a_vector.m_row != n-1 || c_vector.m_row != n-1 || r_vector.m_row!=n){
        throw std::invalid_argument("Error in method solve_thomas_system method, given arguments dimensions do not match");
    }
    Matrix solution_vector=Matrix::nul_matrix(n,1);
    Matrix r_prime_vector=Matrix::nul_matrix(n,1);
    Matrix c_prime_vector=Matrix::nul_matrix(n-1,1);
    c_prime_vector.set_element(1,1,c_vector.get_element(1,1)/b_vector.get_element(1, 1));
    for(int i=2;i<n;i+=1){
        c_prime_vector.set_element(i,1, c_vector.get_element(i,1)/(b_vector.get_element(i,1)-(a_vector.get_element(i-1,1)*c_prime_vector.get_element(i-1,1))));
    }
    r_prime_vector.set_element(1,1,r_vector.get_element(1,1)/b_vector.get_element(1,1));
    for(int i=2;i<=n;i+=1){
        double numerator=r_vector.get_element(i,1)-(a_vector.get_element(i-1,1)*r_prime_vector.get_element(i-1,1));
        double denominator=b_vector.get_element(i,1)-(a_vector.get_element(i-1,1)*c_prime_vector.get_element(i-1,1));
        r_prime_vector.set_element(i,1, numerator/denominator);
    }
    solution_vector.set_element(n,1,r_prime_vector.get_element(n,1));
    for(int j=n-1;j>0;j-=1){
        solution_vector.set_element(j,1,r_prime_vector.get_element(j, 1)-(c_prime_vector.get_element(j,1)*solution_vector.get_element(j+1,1)));
    }
    return solution_vector;
}