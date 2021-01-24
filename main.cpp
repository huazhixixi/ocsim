#include <iostream>
#include <armadillo>
namespace ar = arma;



#include <chrono>
using namespace std;
using namespace chrono;




int main()
{
    arma::cx_mat mat(10,1,arma::fill::zeros);
    mat.col(0) = {0,1,2,3,4,5,6,7,8,9};

    // reverse a function
    mat = arma::reverse(mat);
    mat.print();
}
