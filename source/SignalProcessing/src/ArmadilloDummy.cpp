#include <SignalProcessing/ArmadilloDummy.h>

#include <armadillo>

#include <iostream>

using namespace std;

int det()
{
    arma::arma_rng::set_seed_random();
    arma::Mat<double> A = arma::randu(4,4);
    cout << A << endl;

    return arma::det(A);
}
