//
// Created by Simone Ulzega on 09.06.21.
//
#include <vector>
using namespace std;

//// **********************************************************************
//// Forward transformation xi -> u
vector<double> xi2u(int nx, int jx, const vector<double> & xi){

    vector<double> u(nx*jx+1);

    for (size_t s = 0; s < nx; ++s)
    {
        u[s*jx] = xi[s*jx];
        for (int k = 2; k <= jx; ++k)
            u[s*jx+k-1] = xi[s*jx+k-1] - ( ((double)k-1)*xi[s*jx+k] + xi[s*jx] )/(double)k;
    }
    u[nx*jx] = xi[nx*jx];

    return u;
}
//// **********************************************************************

//// **********************************************************************
//// Back transformation u -> xi
vector<double> u2xi(int nx, int jx, const vector<double> & u){

    double temp;
    vector<double> out(nx*jx+1);

    for (size_t s = 0; s < nx; ++s)
    {
        out[s*jx] = u[s*jx];
        for (size_t k = 2; k <= jx; ++k){
            temp = 0.0;
            for (size_t l = k; l <= jx + 1; ++l){
                temp += (((double)k-1.0)/((double)l-1.0)) * u[s*jx+l-1];
            };
            out[s*jx+k-1] = temp + (((double)jx-(double)k+1.0)/(double)jx) * u[s*jx];
        }

    }
    out[nx*jx] = u[nx*jx];

    return out;
}
//// **********************************************************************
