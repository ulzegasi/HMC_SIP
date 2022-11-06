//
// Function 'aV_n_1': potential energy function to be used in the Adept framework for AD
// It does not include V_N since the fast dynamics is solved analytically
//
// Function 'dV_fun': returns the derivatives of the potential energy
//
// Created by Simone Ulzega on 15.06.21.
//

#include <vector>
#include <cmath>
#include "global_variables.h"
#include "heteroscedasticity_function.h"
#include "adept.h"
#include <limits>
#include <boost/math/special_functions/erf.hpp>
using namespace std;
using adept::adouble;

//// ***************************************************************** ////
//// ***************************************************************** ////
//// INSIDE aV_n_1, TERM V_n SHOULD BE MODIFIED FOR DATA-LESS OR RAIN-LESS RUNS
//// Just comment out data-dependent (or rain-dependent) terms
//// ***************************************************************** ////
//// ***************************************************************** ////

adouble aV_n_1(int N, int nx, int ny, int jx, int jy, int np, double dt,
               const vector<double> & x0, const vector<vector<double>> & y,
               const vector<double> & mu_thetas, const vector<double> & sigma_thetas, const vector<adouble> & x,
               const vector<size_t> & non_zeros, const vector<size_t> & zeros){

    adouble K        = x[0];
    adouble xgw      = x[1];
    adouble sigma_z  = x[2];
    adouble sigma_xi = x[3];
    adouble lambda   = x[4];
    adouble gamma    = x[5];
    adouble xir      = x[6];
    adouble S0       = x[7];

    adouble out = 0.0;

    //// *********************************************
    //// ******** V_p ********
    //// *********************************************
    for (int theta_ind = 0; theta_ind < (np-2); ++theta_ind){
        out += log(x[theta_ind]) + (pow(log(x[theta_ind])-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2)));
    };

    for (int theta_ind = np-2; theta_ind < np-1; ++theta_ind){
        out += (pow(x[theta_ind]-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2)));
    };

    for (int theta_ind = np-1; theta_ind < np; ++theta_ind){
        if (x[theta_ind] >= 0){
            out += (pow(x[theta_ind]-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2)));
        } else {
            out += inf;
        }
    };

    //// *********************************************
    //// ******** V_1 ********
    //// *********************************************
    adouble temp1, temp2;
    out += (pow(x[np+N-1],2) + pow(x[np+0],2))/4.0;
    for (int s = 1; s <= nx; ++s){
        temp1 = 0.0;
        for (int kind = 2; kind <= jx; ++kind){
            temp2 = 0.0;
            for (int lind = kind; lind <= jx + 1; ++lind){
                temp2 += (((double)kind-1.0)/((double)lind-1.0)) * x[np+(s-1)*jx+lind-1];
            };
            temp1 += pow(temp2 + (((double)jx-(double)kind+1.0)/(double)jx) * x[np+(s-1)*jx], 2);
        };
        out += dt/(4.0*tau) * (pow(x[np+s*jx],2) + temp1);
    };

    //// *********************************************
    //// ******** V_n ********
    //// *********************************************
    adouble ym0;
    adouble yxi;
    adouble yxi_old = 0.0;
    adouble temp_sum_1, temp_sum_2, temp_3;
    vector<adouble> x_temp(jy);
    int n_pos_rain_points = non_zeros.size();

    for (size_t s = 1; s <= nx; ++s){
        out += (tau/(4.0*(double)jx*dt)) * pow(x[np + (s-1)*jx]-x[np + s*jx], 2);
    }

    //// ***************************************************************************** ////
    //// ***************************************************************************** ////
    //// DATA-DEPENDENT TERMS BEGIN HERE
    //// Comment this out for a data-less run
    //// Or comment out only rain-dependent terms for a run with only discharge observations
    //// ***************************************************************************** ////
    //// ***************************************************************************** ////

    //// ****************** RAIN ***************** ////
    //// ************* Observational error and Jacobian! (POSITIVE rain) ************* ////
    for (auto s: non_zeros){
        //out += log(lambda*(1+gamma)*(xi0[s-1]-xir));
        out += 0.5 * pow(   (pow(x0[s-1] / lambda, 1.0/(1.0+gamma)) + xir)   -x[np + (s-1)*jx],2)/pow(sigma_xi,2)
                + log(lambda*(1+gamma)*pow(x0[s-1] / lambda, gamma/(1.0+gamma)));
    }
    out += (n_pos_rain_points)*log(sigma_xi);
    //// ************* Integration of zero rain probability density ************* ////
    for (auto s: zeros){
        out -= log( 0.5 * (1 + erf( (xir-x[np+(s-1)*jx])/(sqrt(2)*sigma_xi) ) ) );
    }
    //// ************* END OF RAIN PART ************* ////

    //// ****************** DISCHARGE ***************** ////
    out += pow(g(y[0][1]) - ag(S0/K), 2)/(2*pow(sigma_z, 2)) ;
    for (size_t s = 2; s <= ny + 1; ++s){
        ym0 = (S0/K)*pow((1-dt/K), (s-1)*jy) + (1-pow((1-dt/K), (s-1)*jy))*xgw;
        //// **************************
        //// Beginning calculation of the xi-dependent part of y_{M,(s-1)jy+1}
        //// First one needs to find out the indexes of the xis in terms of jx, instead of jy,
        //// in order to express them as u's
        for (int id = 0; id < jy; ++id){
            int q = floor(((s-2)*jy+1+id)/jx);
            int res = (int)((s-2)*jy+1+id)%jx;
            if (res == 1){
                x_temp[id] = x[np + q*jx];   // Boundary bead
            } else if (res > 1) {
                temp_sum_1 = 0.0;
                for (size_t l = res; l <= jx + 1; ++l){
                    temp_sum_1 += (((double)res-1.0)/((double)l-1.0)) * x[np + q*jx+l-1];
                };
                x_temp[id] = temp_sum_1 + (((double)jx-(double)res+1.0)/(double)jx) * x[np + q*jx];   // Staging bead
            } else if (res == 0) {
                temp_sum_1 = 0.0;
                for (size_t l = jx; l <= jx + 1; ++l){
                    temp_sum_1 += ((jx-1.0)/(l-1.0)) * x[np + (q-1)*jx+l-1];
                };
                x_temp[id] = temp_sum_1 + ((jx-jx+1.0)/jx) * x[np + (q-1)*jx];   // Last staging bead: (sx-1)*jx + jx
            }
        }
        temp_sum_2 = 0.0;  // reset sum over index k
        for (size_t ix = 1; ix <= jy; ++ix){
            if (x_temp[ix-1] > x[6])
                temp_sum_2 += pow((1.0-dt/K), jy-ix) * lambda * pow(x_temp[ix-1]-xir, 1+gamma);
        };
        yxi = pow(1.0-dt/K, jy) * yxi_old + (A*dt/K)*temp_sum_2;

        //// End of calculation of the xi-dependent part of y_{M,(s-1)jy+1}
        //// **************************
        out += pow(g(y[s-1][1]) - ag(ym0 + yxi), 2)/(2*pow(sigma_z, 2)) ;
        yxi_old = yxi;
    }
    out += (ny+1)*log(sigma_z);
    //// ************* END OF DISCHARGE PART ************* ////

    //// ***************************************************************************** ////
    //// ***************************************************************************** ////
    //// END OF DATA-DEPENDENT TERMS
    //// ***************************************************************************** ////
    //// ***************************************************************************** ////

    //// ************************************************************************************
    // out *= (-1); // Change sign of the output => F = (-1)*dV = d(-1*V)
    return (-1)*out;
}

//// ***************************************************************** ////
//// *************************** Full Jacobian *********************** ////
//// ***************************************************************** ////

void dV_fun(adept::Stack & stack, int nx, int ny, int jx, int jy, double dt,
            const vector<double> & x0, const vector<vector<double>> & y,
            const vector<double> & mu_thetas, const vector<double> & sigma_thetas, const vector<double> & theta,
            const vector<double> & u, vector<adouble> & x, vector<double> & dy_dx,
            const vector<size_t> & non_zeros, const vector<size_t> & zeros)
{
    int n_theta = (int)theta.size();
    int n_u = (int)u.size();
    int n_x = n_theta + n_u;
    adept::set_values(&x[0], n_theta,&theta[0]);  // Construct a vector of adoubles where the first n_theta elements are theta
    adept::set_values(&x[n_theta], n_u, &u[0]);     // and the remaining n_u elements are u
    stack.new_recording();
    (aV_n_1(n_u, nx, ny, jx, jy, n_theta, dt, x0, y, mu_thetas, sigma_thetas, x, non_zeros, zeros)).set_gradient(1.0);
    stack.compute_adjoint();
    adept::get_gradients(&x[0], n_x, &dy_dx[0]);
}


