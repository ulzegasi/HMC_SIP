//
// Created by Simone Ulzega on 18.03.21.
//
#include <iostream>
#include <vector>
#include <cmath>
#include "global_variables.h"
#include "heteroscedasticity_function.h"
#include "rainfall_potential_transformation.h"
#include "xi_u_transformation.h"
#include <limits>
#include <boost/math/special_functions/erf.hpp>
using namespace std;

//// ***************************************************************** ////
//// ***************************************************************** ////
double V_N(int nx, int jx, double dt, const vector<double> & u)
{
    double out = 0.0;
    for (int s = 1; s <= nx; ++s)
    {
        for (int k = 2; k <= jx; ++k)
        {
            out += 0.25*(tau/dt) * ((double)k/(k-1.0)) * pow(u[(s-1)*jx+k-1],2);
        }
    }
    return out;
};

//// ***************************************************************** ////
//// ***************************************************************** ////

//// THERE ARE TWO POSSIBLE FUNCTIONS V_n, depending on whether we want to use data or not ////
//// ONE OF THE TWO MUST BE COMMENTED OUT !!! ////

//// ********************* FULL INFERENCE, WITH DATA ********************* ////
double V_n(int nx, int ny, int jx, int jy, double dt,
	const vector<double> & x0, const vector<vector<double>> & y, const vector<double> & theta, const vector<double> & u,
	const vector<size_t> & non_zeros_event1, const vector<size_t> & zeros_event1)
{
    double K        = theta[0];
    double xgw      = theta[1];
    double sigma_z  = theta[2];
    double sigma_xi = theta[3];
    double lambda   = theta[4];
    double gamma    = theta[5];
    double xir      = theta[6];
    double S0       = theta[7];

    double ym0;
    double yxi;
    double yxi_old = 0.0;
    double temp_sum = 0.0;

    vector<double> xi_temp = u2xi(nx, jx, u);

    double out = 0.0;

    int n_pos_rain_points = non_zeros_event1.size();

    //// xi0 -> pow(x0 / lambda, 1.0/(1.0+gamma)) + xir

	for (size_t s = 1; s <= nx; ++s){
	    out += (tau/(4.0*jx*dt)) * pow(u[(s-1)*jx]-u[s*jx], 2);
	}

	//// ************* Observational error and Jacobian! (POSITIVE rain) ************* ////
    for (auto s: non_zeros_event1){
        //out += log(lambda*(1+gamma)*(xi0[s-1]-xir));
        out += 0.5 * pow(   (pow(x0[s-1] / lambda, 1.0/(1.0+gamma)) + xir)   -u[(s-1)*jx],2)/pow(sigma_xi,2)
                + log(lambda*(1+gamma)*pow(x0[s-1] / lambda, gamma/(1.0+gamma)));
    }
    out += (n_pos_rain_points)*log(sigma_xi);
    //// ************* Integration of zero rain probability density ************* ////
    for (auto s: zeros_event1){
        out -= log( 0.5 * (1 + boost::math::erf( (xir-u[(s-1)*jx])/ (sqrt(2)*sigma_xi) ) ) );
    }
    //// ************* ************* ************* ////

	out += pow(g(y[0][1]) - g(S0/K), 2)/(2*pow(sigma_z, 2)) ;
	for (size_t s = 2; s <= ny + 1; ++s){
	    ym0 = S0/K*pow((1-dt/K), (s-1)*jy) + (1-pow((1-dt/K), (s-1)*jy))*xgw;
	    //// **************************
	    //// Beginning calculation of the xi-dependent part of y_{M,(s-1)jy+1}
	    for (size_t kind = 1; kind <= jy; ++kind){
            temp_sum += pow((1.0-dt/K), jy-kind) * r(xi_temp[(s-2)*jy + kind - 1], theta);};
        yxi = pow(1.0-dt/K, jy) * yxi_old + (A*dt/K) * temp_sum;
        //// End of calculation of the xi-dependent part of y_{M,(s-1)jy+1}
        //// **************************
	    out += pow(g(y[s-1][1]) - g(ym0 + yxi), 2)/(2*pow(sigma_z, 2)) ;
	    yxi_old = yxi;
        temp_sum = 0.0;  // reset sum over index k
	}
    out += (ny+1)*log(sigma_z);

    return out;
};

//// ********************* PRIORS INFERENCE, WITHOUT DATA ********************* ////
/*double V_n(int nx, int ny, int jx, int jy, double dt,
           const vector<double> & xi0, const vector<vector<double>> & y, const vector<double> & theta, const vector<double> & u)
{
    double out = 0.0;

    //// **************************************************** ////
    //// **************************************************** ////
    //// Comment this out to get rid of data in the inference ////
    //out += 0.5 * pow(xi0[nx]-u[nx*jx],2)/pow(sigma_xi,2) + (ny+1)*log(sigma_z) + (nx+1)*log(sigma_xi);
    //// **************************************************** ////
    //// **************************************************** ////

    for (size_t s = 1; s <= nx; ++s){
        //// **************************************************** ////
        //// **************************************************** ////
        //// WITH data it would be: ////
        //out += 0.5 * pow(xi0[s-1]-u[(s-1)*jx],2)/pow(sigma_xi,2) + (tau/(4.0*jx*dt)) * pow(u[(s-1)*jx]-u[s*jx], 2);
        //// **************************************************** ////
        //// **************************************************** ////
        //// instead, WITHOUT data: ////
        out += (tau/(4.0*jx*dt)) * pow(u[(s-1)*jx]-u[s*jx], 2);
        //// N.B.:ONE OF THE TWO MUST BE COMMENTED OUT !!!
        //// **************************************************** ////
        //// **************************************************** ////
    }

    //// All of the following goes away in a data-less framework
    *//*out += pow(g(y[0][1]) - g(S0/K), 2)/(2*pow(sigma_z, 2)) ;
    for (size_t s = 2; s <= ny + 1; ++s){
        ym0 = S0/K*pow((1-dt/K), (s-1)*jy) + (1-pow((1-dt/K), (s-1)*jy))*xgw;
        //// **************************
        //// Beginning calculation of the xi-dependent part of y_{M,(s-1)jy+1}
        for (size_t kind = 1; kind <= jy; ++kind){
            temp_sum += pow((1.0-dt/K), jy-kind) * r(xi_temp[(s-2)*jy + kind - 1], theta);};
        yxi = pow(1.0-dt/K, jy) * yxi_old + (A*dt/K) * temp_sum;
        //// End of calculation of the xi-dependent part of y_{M,(s-1)jy+1}
        //// **************************
        out += pow(g(y[s-1][1]) - g(ym0 + yxi), 2)/(2*pow(sigma_z, 2)) ;
        yxi_old = yxi;
        temp_sum = 0.0;  // reset sum over index k
    }*//*

    return out;
};*/

//// ***************************************************************** ////
//// ***************************************************************** ////

double V_1(int N, int nx, int jx, double dt, const vector<double> & u)
{
    double out;
    double temp1, temp2;
    out = (pow(u[N-1], 2) + pow(u[0], 2))/4.0;
    for (int s = 1; s <= nx; ++s){
        temp1 = 0.0;
        for (int kind = 2; kind <= jx; ++kind){
            temp2 = 0.0;
            for (int lind = kind; lind <= jx + 1; ++lind){
                temp2 += ((kind-1.0)/(lind-1.0)) * u[(s-1)*jx+lind-1];
            };
            temp1 += pow(temp2 + ((jx-kind+1.0)/(double)jx) * u[(s-1)*jx], 2);
        };
        out += dt/(4.0*tau) * (pow(u[s*jx],2) + temp1);
    };

	return out;
}

//// ***************************************************************** ////
//// ***************************************************************** ////

double V_p(const vector<double> & theta, const vector<double> & mu_thetas, const vector<double> & sigma_thetas)
{
    int np = theta.size();

    double out = 0.0;
    //cout.precision(17);

    //// Priors:
    //// First, log-normal distributions:
    for (int theta_ind = 0; theta_ind < (np-2); ++theta_ind){
        //cout << "Parameter " << theta_ind + 1 << ": " << - log(theta[theta_ind]) - (pow(log(theta[theta_ind])-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2))) - log(sigma_thetas[theta_ind]* sqrt(2*pi)) << endl;
        out += log(theta[theta_ind]) + (pow(log(theta[theta_ind])-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2)));
    };
    //// then, normal distributions (last two parameters):
    for (int theta_ind = np-2; theta_ind < np-1; ++theta_ind){
        //cout << "Parameter " << theta_ind + 1 << ": " << - (pow(theta[theta_ind]-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2))) - log(sigma_thetas[theta_ind]* sqrt(2*pi)) << endl;
        out += (pow(theta[theta_ind]-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2)));
    };
    for (int theta_ind = np-1; theta_ind < np; ++theta_ind){
        if (theta[theta_ind] >= 0){
            //cout << "Parameter " << theta_ind + 1 << ": " << - (pow(theta[theta_ind]-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2))) - log(sigma_thetas[theta_ind] * sqrt(pi) * (1 - boost::math::erf(-mu_thetas[theta_ind]/(sqrt(2.0)*sigma_thetas[theta_ind])))) + log(sqrt(2)) << endl;
            out += (pow(theta[theta_ind]-mu_thetas[theta_ind], 2)/(2*pow(sigma_thetas[theta_ind], 2)));
        } else {
            out += inf;
        }
    };

    return out;
}