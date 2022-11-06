//
// MD integration routine
//
// Created by Simone Ulzega on 10.06.21.
//

#include <vector>
#include <cmath>
#include "global_variables.h"
#include "adept.h"
#include "adept_functions.h"

using namespace std;
using adept::adouble;

//// ***************************************************************** ////
//// ***************************************************************** ////
void napa(int nx, int ny, int jx, int jy, int N, int n_params,
          double dt, double dtau, double m_stg, double m_bdy,
          const vector<double> & mp, const vector<double> & x0, const vector<vector<double>> & y,
          const vector<double> & mu_thetas, const vector<double> & sigma_thetas,
          vector<double> & theta, vector<double> & u, vector<double> & pp, vector<double> & p,
          vector<adouble> & x, adept::Stack & stack,
          const vector<size_t> & non_zeros_event1, const vector<size_t> & zeros_event1, vector<double> & t_derivatives_tot){

    static vector<double> force_old(n_params + N);
    static vector<double> force_new(n_params + N);

    clock_t t_derivative_calc;

    //// Fast outer propagator (V_N), step dtau/2
    double w_stg;
    double u_old, p_old;
    for (size_t s = 1; s <= nx; ++s){
        for (size_t k = 2; k <=jx; ++k){
            w_stg = sqrt(tau*k/(2*(k-1)*dt*m_stg));
            u_old = u[(s-1)*jx+k-1];
            p_old = p[(s-1)*jx+k-1];
            u[(s-1)*jx+k-1] = u_old * cos(w_stg*dtau/2.0) + p_old * sin(w_stg*dtau/2.0) / (m_stg*w_stg);
            p[(s-1)*jx+k-1] = p_old * cos(w_stg*dtau/2.0) - m_stg * w_stg * u_old * sin(w_stg*dtau/2.0);
        }
    }

    //// Slow inner propagator (V_n, V_1), step dtau
    t_derivative_calc = clock();
    dV_fun(stack, nx, ny, jx, jy, dt, x0, y, mu_thetas, sigma_thetas, theta, u, x, force_old, non_zeros_event1, zeros_event1);
    t_derivatives_tot[0] += 1;
    t_derivatives_tot[1] += ((float)(clock()-t_derivative_calc)/CLOCKS_PER_SEC);

    for (int s = 1; s <= (nx+1); ++s)
        u[(s-1)*jx] += dtau * ( p[(s-1)*jx]  + (dtau/2.0) * force_old[n_params+(s-1)*jx] ) / m_bdy;
    for (int ix = 0; ix < n_params; ++ix){
        theta[ix] += dtau * ( pp[ix]  + (dtau/2.0) * force_old[ix] ) / mp[ix];
        // cout << theta[ix] << endl;
    }

    t_derivative_calc = clock();
    dV_fun(stack, nx, ny, jx, jy, dt, x0, y, mu_thetas, sigma_thetas, theta, u, x, force_new, non_zeros_event1, zeros_event1);
    t_derivatives_tot[0] += 1;
    t_derivatives_tot[1] += ((float)(clock()-t_derivative_calc)/CLOCKS_PER_SEC);

    for (int ix = 0; ix < n_params; ++ix)
        pp[ix] += (dtau/2)*( force_old[ix] + force_new[ix] );
    for (int ix = 0; ix < N; ++ix)
        p[ix] += (dtau/2)*( force_old[n_params + ix] + force_new[n_params + ix] );


    //// Again fast outer propagator (V_N), step dtau/2
    for (size_t s = 1; s <= nx; ++s){
        for (size_t k = 2; k <=jx; ++k){
            w_stg = sqrt(tau*k/(2*(k-1)*dt*m_stg));
            u_old = u[(s-1)*jx+k-1];
            p_old = p[(s-1)*jx+k-1];
            u[(s-1)*jx+k-1] = u_old * cos(w_stg*dtau/2.0) + p_old * sin(w_stg*dtau/2.0) / (m_stg*w_stg);
            p[(s-1)*jx+k-1] = p_old * cos(w_stg*dtau/2.0) - m_stg * w_stg * u_old * sin(w_stg*dtau/2.0);
        }
    }

}

//// ***************************************************************** ////
//// ***************************************************************** ////
