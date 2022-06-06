//
// Created by Simone Ulzega on 09.06.21.
//

#ifndef HMC_SIP_XI_U_TRANSFORMATION_H
#define HMC_SIP_XI_U_TRANSFORMATION_H

#include <vector>
using namespace std;

//// **********************************************************************
//// Forward transformation xi -> u
vector<double> xi2u(int, int, const vector<double> &);
//// **********************************************************************

//// **********************************************************************
//// Back transformation u -> xi
vector<double> u2xi(int, int, const vector<double> &);
//// **********************************************************************

#endif //HMC_SIP_XI_U_TRANSFORMATION_H
