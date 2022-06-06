//
// Created by Simone Ulzega on 07.06.21.
//

#include <vector>
#include <cmath>
using namespace std;

//// **********************************************************************
//// From xi to rain: r(xi) = lambda*(xi-xr)^(1+gamma)
double r(double xi, const vector<double> & theta){

    double lambda = theta[4];
    double gamma  = theta[5];
    double xir    = theta[6];

    double out;

    if (xi <= xir){
        out = 0.0;
    } else {
        out = lambda * pow(xi-xir, 1+gamma);
    };

    return out;
}
//// **********************************************************************
//// Inverse transformation, from rain (x) to xi0: xi0 = (x/lambda)^(1/(1+gamma)) + xir
/*
double invr(double x, const vector<double> & theta){

    double lambda = theta[4];
    double gamma  = theta[5];
    double xir    = theta[6];

    double out;

    if (x <= 0){
        cout << "ERROR in function invr: rain must be positive" << endl;
        exit(EXIT_FILE_ERROR);
    } else {
        out = pow(x / lambda, 1.0/(1.0+gamma)) + xir;
    };

    return out;
}*/
