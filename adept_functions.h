//
// Created by Simone Ulzega on 24.06.21.
//

#ifndef HMC_SIP_ADEPT_FUNCTIONS_H
#define HMC_SIP_ADEPT_FUNCTIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include "global_variables.h"
#include "heteroscedasticity_function.h"
#include "adept.h"
using namespace std;
using adept::adouble;

//// ***************************************************************** ////
//// ***************************************************************** ////
adouble aV_n_1(int, int, int, int, int, int, double,
               const vector<double> &, const vector<vector<double>> &,
               const vector<double> &, const vector<double> &, const vector<adouble> &,
                       const vector<size_t> &, const vector<size_t> & );

void dV_fun(adept::Stack &, int, int, int, int,  double,
            const vector<double> &, const vector<vector<double>> &,
            const vector<double> &, const vector<double> &,
            const vector<double> &, const vector<double> &, vector<adouble> &, vector<double> &,
                    const vector<size_t> &, const vector<size_t> &);

#endif //HMC_SIP_ADEPT_FUNCTIONS_H
