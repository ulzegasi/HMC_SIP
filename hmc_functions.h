//
// Created by Simone Ulzega on 18.03.21.
//

#ifndef HMC_SIP_HMC_FUNCTIONS_H
#define HMC_SIP_HMC_FUNCTIONS_H

#include <vector>
#include <cmath>
#include "global_variables.h"
#include "heteroscedasticity_function.h"
#include "rainfall_potential_transformation.h"
#include "xi_u_transformation.h"
using namespace std;

double V_N(int, int, double, const vector<double> &);

double V_n(int, int, int, int, double,
           const vector<double> &, const vector<vector<double>> &, const vector<double> &, const vector<double> &,
                   const vector<size_t> &, const vector<size_t> &);

double V_1(int, int, int, double, const vector<double> &);

double V_p(const vector<double> &, const vector<double> &, const vector<double> &);

#endif //HMC_SIP_HMC_FUNCTIONS_H
