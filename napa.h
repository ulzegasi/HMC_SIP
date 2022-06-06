//
// Created by Simone Ulzega on 11.06.21.
//

#ifndef HMC_SIP_NAPA_H
#define HMC_SIP_NAPA_H

#include <vector>
#include <cmath>
#include "adept.h"
#include "global_variables.h"

using namespace std;
using adept::adouble;

//// ***************************************************************** ////
//// ***************************************************************** ////
void napa(int, int, int, int, int, int, double, double, double, double,
          const vector<double> &, const vector<double> &, const vector<vector<double>> &,
          const vector<double> &, const vector<double> &,
          vector<double> &, vector<double> &, vector<double> &, vector<double> &, vector<adouble> &,
          adept::Stack &, const vector<size_t> &, const vector<size_t> &, vector<double> &);
//// ***************************************************************** ////
//// ***************************************************************** ////

#endif //HMC_SIP_NAPA_H
