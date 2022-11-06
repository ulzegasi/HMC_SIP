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
