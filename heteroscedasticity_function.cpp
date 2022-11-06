//
// Created by Simone Ulzega on 19.03.21.
//

#include <cmath>
#include "global_variables.h"
#include "adept.h"
using namespace std;
using adept::adouble;

//// **********************************************************************
//// Heteroscedasticity function
double g(double y){
    double out = beta * log( sinh( (alpha+y)/beta ) );
    return out;
}
//// **********************************************************************
//// Heteroscedasticity function to be used in the adept framework for derivatives calculations
adouble ag(adouble y){
    adouble out = beta * log( sinh( (alpha+y)/beta ) );
    return out;
}
//// **********************************************************************