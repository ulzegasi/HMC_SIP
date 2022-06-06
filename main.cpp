////
////  main.cpp
////  Bayesian parameter inference in hydrological modelling
////  using a Hamiltonian Monte Carlo approach with a stochastic rain model
////
////  Created by Ulzega Simone, February 2021
////  simone.ulzega@zhaw.ch
////

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>
#include "vector_operations.h"
#include "hmc_functions.h"
#include "xi_u_transformation.h"
#include "napa.h"
#include "adept_source.h"
#include <limits>
#include "adept_functions.h"
#define EXIT_FILE_ERROR (1)
using namespace std;
using adept::adouble;


//// ********************************************************************* ////
//// *************************** Boost Libraries ************************* ////
//// ********************************************************************* ////
//// https://www.boost.org/doc/libs/1_75_0/more/getting_started/unix-variants.html
////
//// The most reliable way to get a copy of Boost is to download a distribution from SourceForge:
//// 1. Download boost_1_75_0.tar.bz2.
//// 2. In the directory where you want to put the Boost installation, execute:
//// tar --bzip2 -xf /path/to/boost_1_75_0.tar.bz2
////
//// > cd boost_1_75_0
//// > ./bootstrap.sh
//// > ./b2
//// > ./b2 install
//// ********************************************************************* ////
//// ********************************************************************* ////

//// ********************************************************************* ////
//// *************************** Data-less test ************************* ////
//// ********************************************************************* ////
//// 1) Modify "hmc_functions.cpp" and "adept_functions.cpp"
//// (instructions in the functions)
//// ********************************************************************* ////
//// ********************************************************************* ////

//// *************************** Global variables ************************* ////
//// ******************* (declared in global_variables.h) ***************** ////
double A, alpha, beta, tau, pi, inf;
//// ********************************************************************** ////

//// ****************************** Functions ***************************** ////
//// **********************************************************************
//// To sample from truncated normal
//double sample_from_truncated_normal(double b, double mu, double si){
//    //// The distribution is defined over the support (-inf, b] with mean 'mu' and standard deviation 'si'
//
//    static mt19937_64 engine1(0);
//    static uniform_real_distribution<double> r(0,1);
//    double r1 = r(engine1);
//    double beta =  (b-mu)/si;
//    double arg = (1 + boost::math::erf(beta/sqrt(2.0))) * r1 - 1;
//    double res = mu + si * sqrt(2.0) * boost::math::erf_inv(arg);
//    return res;
//}
//// **********************************************************************

//// **********************************************************************
//// To fill in a vector with normally distributed random numbers (used to generate momenta in HMC loop)
void norm_rand(vector<double> & vec, double mu = 0.0, double std = 1.0)
{
    static mt19937_64 engine2(101);
    // static mt19937_64 engine(time(NULL));
    static normal_distribution<double> norm_dist_vec(mu, std);
    for (size_t ix = 0; ix < vec.size(); ++ix)
        vec[ix] = norm_dist_vec(engine2);
}
//// **********************************************************************

//// **********************************************************************
//// To generate random real numbers from normal distribution N[0,1]
double n_rand(double mu = 0.0, double std = 1.0)
{
    static mt19937_64 engine3(102);
    // static mt19937_64 engine(time(NULL));
    static normal_distribution<double> norm_dist(mu, std);
    double res = norm_dist(engine3);
    return res;
}
//// **********************************************************************

//// **********************************************************************
//// To generate random real numbers from uniform distribution U[0,1]
double u_rand(double a = 0.0, double b = 1.0)
{
    static mt19937_64 engine4(103);
    // static mt19937_64 engine(time(NULL));
    static uniform_real_distribution<double> uniform_dist(a,b);
    double res = uniform_dist(engine4);
    return res;
}
//// **********************************************************************
//// *************************** End of functions ************************* ////

//// ************************* Useful directories ************************* ////
// Local:
const string datadir  = "/Users/ulzg/CLionProjects/HMC_SIP/data/";  // change this when moving to cluster
const string outdir  = "/Users/ulzg/CLionProjects/HMC_SIP/output/"; // change this when moving to cluster
// Cluster:
//const string datadir  = "/cfs/earth/scratch/ulzg/hmc_sip/data/";
//const string outdir  = "/cfs/earth/scratch/ulzg/hmc_sip/output/";
//// ********************************************************************** ////


//// ********************************************************************* ////
//// ******************************* Main ******************************** ////
//// ********************************************************************* ////
int main() {

    cout << endl;
    cout << "************************************" << endl;
    cout << "************* Starting *************" << endl;
    cout << "************************************" << endl;
    cout << endl;

    bool save_init_polymer = true;  // Flag 'save / not save' initial config
    bool continue_from_previous = true;  // Flag 'continue / from scratch'
    const string previous_file_name = "burnin_retuned_nogibbs_jacob";  // Used only if continue_from_previous = true
    bool use_sc2 = true;  // Flag 'Sc2 (inaccurate, set to true) / Sc1 (accurate, set to false)'
    string scenario;
    if (use_sc2){
        scenario = "Sc2";
    } else {
        scenario = "Sc1";
    }

    //// ********************************************************************* ////
    //// ***************************** Load data ***************************** ////
    //// ********************************************************************* ////

    cout << "*** Loading data ***" << endl;

    //// **************************** Rain events **************************** ////
    vector<vector<double>> x01;

    // Open file streams //
    ifstream ifs_rain1( datadir + "event1_rain_" + scenario + ".dat" );

    if (!ifs_rain1.is_open())
    {
        cout << "Exiting: unable to open file for rain event" << endl;
        exit(EXIT_FILE_ERROR);
    }

    string line;

    while (getline(ifs_rain1, line))   // read one line from ifs
    {
        stringstream ss(line); // access line as a stream
        std::vector<double> values;
        double value;
        while (ss >> value)
        {
            values.push_back(value);
        }
        x01.push_back(values);
    }
    x01.erase(x01.begin());

    // Close file streams
    ifs_rain1.close();

    //// ******************************************************************************** ////
    //// ******************************************************************************** ////
    //// ******************************************************************************** ////
    //// ****** Getting rid of initial data points ****** ////
    //// This is done to have rain and discharges starting at the same time point ////
    //// IT IS NOT IDEAL, ONLY A TEMPORARY SOLUTION, THE CODE SHOULD BE ADAPTED TO ALLOW Y STARTING AT LATER TIME ////
    if (!use_sc2){
        x01.erase(x01.begin(),x01.begin()+3);
        x01.erase(x01.end() - 3, x01.end());
    }
    //// ******************************************************************************** ////
    //// ******************************************************************************** ////
    //// ******************************************************************************** ////

    // Print file dimensions
    cout << "Dimensions of rain event datasets: (" << x01.size() << "," << x01[0].size() << ")" << endl;

    //// To have rain inputs in mm/s instead of mm/min
    for (size_t ix = 0; ix < x01.size(); ++ix)
        x01[ix][1] = x01[ix][1]/60.;

    /*for (size_t ix = 0; ix < x01.size(); ++ix)
        cout << x01[ix][1] << ' ';
    return(0);*/

    //// Create a vector containing observed rain without times
    vector<double> x0;
    for (size_t ix = 0; ix < x01.size(); ++ix)
        x0.push_back(x01[ix][1]);

    //// ********************************************************************* ////

    //// ************************** Discharge events ************************** ////
    vector<vector<double>> y1;

    // Open file streams //
    ifstream ifs_out1( datadir + "event1_discharge.dat" );
    if (!ifs_out1.is_open())
    {
        cout << "Exiting: unable to open file for discharge event" << endl;
        exit(EXIT_FILE_ERROR);
    }

    while (getline(ifs_out1, line))   // read one line from ifs
    {
        stringstream ss(line); // access line as a stream
        std::vector<double> values;
        double value;
        while (ss >> value)
        {
            values.push_back(value);
            // std::cout << value << std::endl;
        }
        y1.push_back(values);
    }
    y1.erase(y1.begin());

    // Close file streams
    ifs_out1.close();

    //// ****** Getting rid of first discharge point ****** ////
    //// This is needed to ensure that the first xi corresponds to an input data point, which is needed  ////
    //// to apply Tuckerman coordinate transformation (Tuckerman et al., J. Chem. Phys. 99, 1993) ////

    if (use_sc2){
        y1.erase(y1.begin());
    }; /*else {
        y1.erase(y1.begin());
    }*/

    //// ****** Getting rid of last discharge points ****** ////
    //// We don't need discharge series that are longer than the rain series ////
    //// Longer ys would cause problems within the function V_n ////

    if (use_sc2){
        // y1 is good as it is
    }

    // Print file dimensions
    cout << "Dimensions of discharge datasets: (" << y1.size() << "," << y1[0].size() << ")" << endl;
    cout << endl;

    //// ********************************************************************* ////

    //// ************************ Get measurements times ************************ ////
    vector<double> tx01;
    vector<double> ty1;

    for ( size_t ir = 0; ir < x01.size(); ir++ ) {
        tx01.push_back(x01[ir][0]);
    }


    for ( size_t ir = 0; ir < y1.size(); ir++ ) {
        ty1.push_back(y1[ir][0]);
    }

    /*for (auto vi: tx01)
        cout << vi*60 << ' ';
    cout << endl << endl;
    for (auto vi: ty1)
        cout << vi*60 << ' ';
    return(0);*/
    //// ********************************************************************* ////

    //// ********************************************************************* ////
    //// ***************************** Parameters **************************** ////
    //// ********************************************************************* ////
    const int n_params = 8;

    //// Parameters are (LN = log-normal, N = normal):
    /// K (LN), xgw (LN), sigma_z (LN), sigma_xi (LN), lambda (LN), xir (N), gamma (LN), S0 (N)

    //// **************************** Model parameters: fixed values *************************** ////
    A = 11815.8;   // Area [m^2]
    alpha = 25.0;  // Coefficients of the discharge heteroscedasticity transformation H [l/s]
    beta = 50.0;   // Coefficients of the discharge heteroscedasticity transformation H [l/s]
    tau = 636.0;     // Auto correlation coefficient of the OU process [s]
    pi = boost::math::constants::pi<double>();
    inf = std::numeric_limits<double>::infinity();

    //// **************** Model parameters: mu and sigma for log-normal priors **************** ////
    vector<double> mu_thetas_temp(n_params), mu_thetas(n_params);
    vector<double> sigma_thetas_temp(n_params), sigma_thetas(n_params);

    //// Parameters mean and sd (they need to be transformed for LN distributions)
    mu_thetas_temp = {284.4, 6.0, 4.5, 0.65, 0.1/60.0, 0.5, 0.5, 0.0};
    sigma_thetas_temp = {57.6, 1.0, 0.45, 0.3, 0.05/60.0, 0.25, 0.1, 5000.0};

    for (size_t ix = 0; ix < 6; ++ix){
        mu_thetas[ix] = log(pow(mu_thetas_temp[ix],2)/sqrt(pow(mu_thetas_temp[ix],2)+
                                                           pow(sigma_thetas_temp[ix],2)));
        sigma_thetas[ix] = sqrt(log(1+(pow(sigma_thetas_temp[ix],2)/pow(mu_thetas_temp[ix],2))));
    }
    for (size_t ix = 6; ix < 8; ++ix){
        mu_thetas[ix] = mu_thetas_temp[ix];
        sigma_thetas[ix] = sigma_thetas_temp[ix];
    }

    //// **************** Model parameters: init values for inferred parameters **************** ////
    vector<double> theta(n_params);

    double K;
    double xgw;
    double sigma_z;
    double sigma_xi;
    double lambda;
    double gamma;
    double xir;
    double S0;

    if (continue_from_previous == true){

        cout << "*** Loading parameters from previous run ***" << endl;

        vector<vector<double>> thetas_from_previous;
        ifstream ifs_theta( outdir + "thetas_" + previous_file_name + ".dat" );
        // Reads the whole thetas file, while we need only last line, obviously not efficient
        // It's not a problem (we read it only once before beginning HMC), but should be changed
        if (!ifs_theta.is_open())
        {
            cout << "Exiting: unable to open file for parameter values" << endl;
            exit(EXIT_FILE_ERROR);
        }
        while (getline(ifs_theta , line))   // read one line from ifs
        {
            stringstream ss(line); // access line as a stream
            std::vector<double> values;
            double value;
            while (ss >> value)
            {
                values.push_back(value);
            }
            thetas_from_previous.push_back(values);
        }

        K = thetas_from_previous[thetas_from_previous.size()-1][0];       // Reservoir retention rate [s]
        xgw =thetas_from_previous[thetas_from_previous.size()-1][1];       // Groundwater base flow [l/s]
        sigma_z = thetas_from_previous[thetas_from_previous.size()-1][2];   // Standard deviation of the observational error model for H(y) [l/s]
        sigma_xi = thetas_from_previous[thetas_from_previous.size()-1][3];  //// Does starting with a larger sigma_xi change the posterior?
        lambda = thetas_from_previous[thetas_from_previous.size()-1][4];    // Scaling factor within rainfall transformation r [l/(s*m^2)]
        gamma = thetas_from_previous[thetas_from_previous.size()-1][5];     // Exponential factor within rainfall transformation r
        xir = thetas_from_previous[thetas_from_previous.size()-1][6];       // Domain split location within rainfall transformation r
        S0 = thetas_from_previous[thetas_from_previous.size()-1][7];  // Reservoir level at t=0 [l] -> mean of N(0, 5000) truncated to [0, inf)

        theta[0] = K;
        theta[1] = xgw;
        theta[2] = sigma_z;
        theta[3] = sigma_xi;
        theta[4] = lambda;
        theta[5] = gamma;
        theta[6] = xir;
        theta[7] = S0;

    } else {

        K = 284.4;       // Reservoir retention rate [s]
        xgw = 6.0;       // Groundwater base flow [l/s]
        sigma_z = 4.5;   // Standard deviation of the observational error model for H(y) [l/s]
        sigma_xi = 0.65;  
        lambda = 0.1/60.0;    // Scaling factor within rainfall transformation r [l/(s*m^2)]
        gamma = 0.5;     // Exponential factor within rainfall transformation r
        xir = 0.5;       // Domain split location within rainfall transformation r
        S0 = 5000.0 * sqrt(2.0/pi);  // Reservoir level at t=0 [l] -> mean of N(0, 5000) truncated to [0, inf)

        theta[0] = K;
        theta[1] = xgw;
        theta[2] = sigma_z;
        theta[3] = sigma_xi;
        theta[4] = lambda;
        theta[5] = gamma;
        theta[6] = xir;
        theta[7] = S0;
    }

    if (continue_from_previous == true){
        cout << "*** Starting from file *" << previous_file_name << "* with the following parameter values: ***" << endl;
    } else {
        cout << "*** Starting from scratch with the following parameter values: ***" << endl;
    }
    cout << "K = " << theta[0] << endl;
    cout << "Xgw = " << theta[1] << endl;
    cout << "sigma_z = " << theta[2] << endl;
    cout << "sigma_xi = " << theta[3] << endl;
    cout << "lambda = " << theta[4] << endl;
    cout << "gamma = " << theta[5] << endl;
    cout << "xi_r = " << theta[6] << endl;
    cout << "S0 = " << theta[7] << endl;
    cout << endl;

    //// Set parameter limits (used a bit artificially in HMC to prevent divergence issues)
    double K_min = 0.0, K_max = 5e3;
    double xgw_min = 0.0, xgw_max = 200;
    double sigma_z_min = 0.0, sigma_z_max = 100;
    double sigma_xi_min = 0.0, sigma_xi_max = 50;
    double lambda_min = 0.0, lambda_max = 100;
    double gamma_min = 0.0, gamma_max = 50;
    double xir_min = -50.0, xir_max = 50.5;
    double S0_min = -1e4, S0_max = 5e4;

    //// ********************************************************************* ////
    //// PDF, mean and SD for the parameters
    //// (number in parenthesis are mean and sd of the random variable)
    //// ********************************************************************* ////
    //// K ->        LN(284.4, 57.6)
    //// xgw ->      LN(6, 1)
    //// sigma_z ->  LN(4.5, 0.45)
    //// sigma_xi -> LN(0.65, 0.3)
    //// lambda ->   LN(0.1, 0.05)
    //// gamma ->    LN(0.5, 0.25)
    //// xir ->      N(0.5, 0.1)
    //// S0 ->       N(0, 5000) truncated to [0,inf)
    //// ********************************************************************* ////

    //// *********************** Discretization parameters ********************** ////
    const int nx1= x01.size() - 1;
    const int ny1= y1.size() - 1;

    int jx;

    if (use_sc2){
        jx = 60;  // Delta_t between data points = 10 minutes = 600 seconds -> fine grid step dt = 10 seconds
    } else {
        jx = 6;  // Delta_t between data points = 1 minute = 60 seconds -> fine grid step dt = 10 seconds
    }

    const int jy = 24;  // Delta_t between data points = 4 minutes = 240 seconds -> fine grid step dt = 10 seconds


    const int N1 = nx1 * jx + 1;

    cout << "*** Total number of discretization points (Nx): " << N1 << " ***" << endl;

    //// *********************** Time unit ********************** ////
    double dt = round((tx01[1]-tx01[0])/jx *3600);

    cout << "*** Time unit dt (for all events): " << dt << " seconds ***" << endl;
    cout << endl;

    //// ********************************************************************* ////
    //// ******************* Indexes of measurement points ****************** ////
    //// ************************ and zero-rain points *********************** ////
    //// ********************************************************************* ////

    //// Indexes of data points within global discretization framework (size N)
    //// Indexes start at 1
    vector<size_t> obs_ind_1(nx1+1);
    for (size_t ix = 0; ix < nx1+1; ++ix)
        obs_ind_1[ix] = (int)ix * jx + 1;

    /*for (auto i: obs_ind_1)
        cout << i << ' ';
    return 0;*/

    //// ********************************************************************* ////

    //// Find zero-rain points (indices in the "event files" of size nx + 1)
    //// Indexes start at 1
    vector<size_t> zeros_event1;
    auto it1 = find_if(x01.begin(), x01.end(), [](vector<double> v){return v[1] == 0;});
    while (it1 != x01.end()) {
        zeros_event1.push_back(distance(x01.begin(), it1) + 1);
        it1 = find_if(next(it1), x01.end(), [](vector<double> v){return v[1] == 0;});
    }

    vector<size_t> non_zeros_event1;
    it1 = find_if(x01.begin(), x01.end(), [](vector<double> v){return v[1] != 0;});
    while (it1 != x01.end()) {
        non_zeros_event1.push_back(distance(x01.begin(), it1) + 1);
        it1 = find_if(next(it1), x01.end(), [](vector<double> v){return v[1] != 0;});
    }

    /*for (auto i: zeros_event1)
        cout << i << ' ';
    cout << endl;
    for (auto i: non_zeros_event1)
        cout << i << ' ';
    cout << endl;
    return(0);*/
    //// ********************************************************************* ////

    //// Now zero-rain indexes in the global discretization framework (size N)
    //// As usual, indexes start at 1
    vector<size_t> zero_ind_1(zeros_event1.size());
    for (time_t ix = 0; ix < zeros_event1.size(); ++ix)
        zero_ind_1[ix] = ((int)zeros_event1[ix] - 1) * jx + 1;

    cout << setprecision(3) << "*** Number of zero-rain points in events 1, 2 and 3: "
         << zeros_event1.size() << "(" << (double)zeros_event1.size()/(nx1+1)*100 << "%) " << "***" << endl;
    cout << endl;
    //// ********************************************************************* ////

    //// ************************************************************************* ////
    //// ***************************** Initial state ***************************** ////
    //// ************************ (linear interpolation) ************************* ////
    //// ************************************************************************* ////

    cout << "*** Generating initial configuration ***" << endl;

    //// Containers for rainfall potential over all discretization points
    vector<double> xi1(N1);
    double previous_xi;

    //// ************ Event 1 ************ ////
    if (continue_from_previous == true) {

        cout << "*** Loading xi configuration from previous run ***" << endl;

        vector<double> xi_from_previous;
        ifstream ifs_xi_last( outdir + "pred_xi_last_" + previous_file_name + ".dat" );
        if (!ifs_xi_last.is_open())
        {
            cout << "Exiting: unable to open file for xi values" << endl;
            exit(EXIT_FILE_ERROR);
        }
        while (getline(ifs_xi_last , line))   // read one line from ifs
        {
            stringstream ss(line); // access line as a stream
            // std::vector<double> values;
            double value;
            while (ss >> value)
                xi_from_previous.push_back(value);
        }
        for (size_t ix = 0; ix < N1; ++ix)
            xi1[ix] = xi_from_previous[ix];

    } else {

        //// Generate OU process
        xi1[0] = n_rand();
        for (size_t ix = 0; ix < N1-1; ++ix)
            xi1[ix+1] = xi1[ix] - dt * (xi1[ix]/tau) + sqrt(dt) * sqrt(2/tau) * n_rand();
    }

    if (save_init_polymer == true)
    {
        ofstream ofs_init_xi1(datadir + "init_polymer_xi1.dat");
        if (!ofs_init_xi1.is_open())
        {
            cout << "Exiting: unable to open file to write initial configuration" << endl;
            exit(EXIT_FILE_ERROR);
        }
        for (size_t ix = 0; ix < N1; ++ix)
            ofs_init_xi1 << xi1[ix] << endl;
        ofs_init_xi1.close();
    }

    //// ************************************************************************* ////
    //// ****************** Coordinate transformation: xi -> u ******************* ////
    //// ************ (eqs 2.16-17 in Tuckerman et al., JCP 99, 1993) ************ ////
    //// ************************************************************************* ////

    cout << "*** Coordinates transformation xi -> u ***" << endl;

    //// Containers for discretized rainfall potential in the transformed coordinates u
    vector<double> u1(N1);

    /// IMPORTANT: the transformation that should be applied to the intermediate points is:
    //// u[s*j+k-1] = xi[s*j+k-1] - ( (k-1)*xi[s*j+k] + xi[s*j] )/k;
    //// BUT: it is always = 0 when the xi[sj+k-1] between data points are
    //// a linear interpolation of the data points xi[sj]
    //// For staging beads as linear interpolation of data points, therefore one could use
    //// u[s*jx + k - 1] = 0.0;

    for (size_t s = 0; s < nx1; ++s)
    {
        u1[s*jx] = xi1[s*jx];
        for (int k = 2; k <= jx; ++k)
            u1[s*jx+k-1] = xi1[s*jx+k-1] - ( ( (k-1.0)*xi1[s*jx+k] + xi1[s*jx] )/(double)k );
    }
    u1[N1-1] = xi1[N1-1];

    if (save_init_polymer == true)
    {
        ofstream ofs_init_u1(datadir + "init_polymer_u1.dat");
        if (!ofs_init_u1.is_open())
        {
            cout << "Exiting: unable to open file to write initial configuration" << endl;
            exit(EXIT_FILE_ERROR);
        }
        for (size_t ix = 0; ix < N1; ++ix)
            ofs_init_u1 << u1[ix] << endl;
        ofs_init_u1.close();
    }

    //// ************************************************************************* ////
    //// ********************** Create containers for HMC ************************ ////
    //// ************************************************************************* ////

    cout << "*** Preparing HMC ***" << endl;

    //// Containers for Markov chains
    vector< vector<double> > theta_sample;
    vector< vector<double> > u1_sample;
    vector<double> energies;
    vector<double> u0_sample;
    //// Store initial values
    theta_sample.push_back(theta);
    u1_sample.push_back(u1);

    //// Containers for temporary values during HMC steps
    vector<double> theta_save;
    vector<double> u1_save;

    //// Containers for masses and momenta
    vector<double> mp(n_params), m1(N1);
    vector<double> sqrt_mp(n_params), sqrt_m1(N1);
    vector<double> pp, p1;

    //// Containers for random numbers (to generate momenta p)
    vector<double> rand_vec_p(n_params), rand_vec_1(N1);

    //// Containers for energies and probabilities
    double V_old, V_new, H_old, H_new;
    double accept_prob;

    //// Containers for execution times
    vector<double> time_napa;

    //// Containers for AD
    adept::Stack stack;
    vector<adouble> x(n_params+N1);

    //// ************************************************************************* ////
    //// ****************************** HMC block ******************************** ////
    //// ************************************************************************* ////

    //// Definition of masses
    double m_bdy = 1.6;   // m = m_q / dt
    double m_stg = 0.4;   // we assume m_q prop. to dt ==> m = costant

    mp[0] = 1e-5;  // refers to K
    mp[1] = 1.0;  // refers to Qgw
    mp[2] = 1.0;  // refers to sigma_z
    mp[3] = 1.0;  // refers to sigma_xi
    mp[4] = 2e5;  // refers to lambda
    mp[5] = 0.5;  // refers to gamma
    mp[6] = 15.0;  // refers to xir
    mp[7] = 1e-7;  // refers to S0

    //// Masses for coordinates
    for (int s = 0; s < nx1; ++s){
        m1[s*jx] = m_bdy;
        for (int k = 2; k <= jx; ++k)
            m1[s*jx + k - 1] = m_stg;
    }
    m1[N1-1] = m_bdy;
    //// Square roots
    sqrt_mp = vsqrt(mp);
    sqrt_m1 = vsqrt(m1);

    //// Simulation parameters
    // const int n_samples = 5e4; // Total number of points in the MCMC
    const int n_samples = 10000;
    const double dtau = 0.015;
    const int n_napa = 3;

    //// ************************************************************************* ////
    //// *********************** Get ready to save results *********************** ////
    //// ************************************************************************* ////
    const string output_file_name = "file_name_here";
    //// **************************** ////
    ofstream ofs_thetas(outdir + "thetas_" + output_file_name + ".dat");
    if (!ofs_thetas.is_open())
    {
        cout << "Exiting: unable to open file to write parameter samples" << endl;
        exit(EXIT_FILE_ERROR);
    }
    ofs_thetas << "K" << "\t" << "xgw" << "\t" << "sigma_z" << "\t" << "sigma_xi" << "\t" << "lambda" << "\t" << "gamma" << "\t" <<
               "xi_r" << "\t" << "S_0" << "\t" << endl;
    //// **************************** ////
    ofstream ofs_energies(outdir + "energies_" + output_file_name + ".dat");
    if (!ofs_energies.is_open())
    {
        cout << "Exiting: unable to open file to write energy samples" << endl;
        exit(EXIT_FILE_ERROR);
    }
    //// **************************** ////
    ofstream ofs_pred_x(outdir + "pred_x_" + output_file_name + ".dat");
    if (!ofs_pred_x.is_open())
    {
        cout << "Exiting: unable to open file to write input predictions" << endl;
        exit(EXIT_FILE_ERROR);
    }
    //// **************************** ////
    ofstream ofs_pred_xis(outdir + "pred_xis_" + output_file_name + ".dat");
    if (!ofs_pred_xis.is_open())
    {
        cout << "Exiting: unable to open file to write xis" << endl;
        exit(EXIT_FILE_ERROR);
    }
    //// **************************** ////
    ofstream ofs_pred_xi_init(outdir + "pred_xi_init_" + output_file_name + ".dat");
    if (!ofs_pred_xi_init.is_open())
    {
        cout << "Exiting: unable to open file to write xis (first point)" << endl;
        exit(EXIT_FILE_ERROR);
    }
    //// **************************** ////
    ofstream ofs_pred_xi_last(outdir + "pred_xi_last_" + output_file_name + ".dat");
    if (!ofs_pred_xi_last.is_open())
    {
        cout << "Exiting: unable to open file to write xis (last realization)" << endl;
        exit(EXIT_FILE_ERROR);
    }
    //// **************************** ////
    ofstream ofs_pred_y(outdir + "pred_y_" + output_file_name + ".dat");
    if (!ofs_pred_y.is_open())
    {
        cout << "Exiting: unable to open file to write output predictions" << endl;
        exit(EXIT_FILE_ERROR);
    }

    //// ********** Preparing indexes of inputs and outputs to be saved ********** ////
    //// ********* (we save only a selection due to memory limitations) ********** ////
    int red_size = 2e4;    // number of realizations to be saved
    red_size = min(red_size, n_samples);

    double steppo = (double)(n_samples) / (double)(red_size);

    vector<size_t> red_ind;  // indexes of realizations to be saved
    for (size_t ix = 1; ix <= red_size; ++ix){
        red_ind.push_back(round(ix*steppo));
    }

    vector<size_t> red_ind_with_0;
    for (size_t ix = 0; ix <= red_size; ++ix){
        red_ind_with_0.push_back(round(ix*steppo));
    }
    //// **************************** ////

    //// ************************************************************************* ////
    //// ************************************************************************* ////
    //// ****************************** HMC loop ********************************* ////
    //// ************************************************************************* ////
    //// ************************************************************************* ////

    cout << endl << " ======================================================== " << endl;
    cout << "*** Starting HMC loops ***" << endl;
    cout << " ======================================================== " << endl << endl;

    int reject_counter = 0;
    int params_lim_counter = 0;
    int u_lim_counter = 0;
    int red_ind_ix = 0;

    clock_t t_init = clock();
    clock_t t_energy_calc;
    vector<double> t_energy_tot(2, 0.0);
    vector<double> t_derivatives_tot(2, 0.0);
    // These are used to estimate the contributions of energy calculations (-> function evaluations) compared to
    // the derivative estimations from napa

    for (int counter = 1; counter <= n_samples; ++counter){

        //// ********************************************************************* ////
        //// Sample momenta
        //// ********************************************************************* ////
        norm_rand(rand_vec_p);
        norm_rand(rand_vec_1);
        pp = vtimes(sqrt_mp, rand_vec_p);
        p1 = vtimes(sqrt_m1, rand_vec_1);

        /*for (auto ix: theta)
            cout << ix << " ";
        cout << endl;*/

        //// ********************************************************************* ////
        //// Calculate energy
        //// ENERGY IS CALCULATED ONLY ONCE, AT THE END OF THE LOOP (except for the first step of the chain)
        //// ********************************************************************* ////

        if (counter==1){
            //// This is done only once, at the first step of the chain.
            /// Afterwards, V_old is stored at the end of this loop
            V_old = V_N(nx1, jx, dt, u1) +
                    V_n(nx1, ny1, jx, jy, dt, x0, y1, theta, u1, non_zeros_event1, zeros_event1) +
                    V_1(N1, nx1, jx, dt, u1)  + V_p(theta, mu_thetas, sigma_thetas);
        }

        H_old = vreduce(vdiv(vsquare(pp), ctimes(2.0, mp))) +
             vreduce(vdiv(vsquare(p1), ctimes(2.0, m1))) + V_old;

        //// ********************************************************************* ////
        //// ********************************************************************* ////
        //// Finite energy check and safe way out
        //// ********************************************************************* ////
        //// ********************************************************************* ////
        if ( isnan(H_old) != 0) {
            cout << "Exiting at sample " << counter << ": energy value -- " << H_old << " -- not acceptable" << endl;
            cout << " *******************" << endl;
            cout << "Current parameter values: " ;
            for (auto ix: theta)
                cout << ix << " ";
            cout << endl;
            cout << " *******************" << endl;

            int red_size_911;
            red_size_911 = min(red_size, counter + 1);

            double steppo = (double)(counter) / (double)(red_size_911 - 1.0);

            vector<size_t> red_ind_911;  // indexes of realizations to be saved
            for (size_t ix = 0; ix < red_size_911 - 1; ++ix)
                // 'red_size_911 - 1' because the current iteration (step = counter) does not count
                red_ind_911.push_back(round(ix*steppo));

            cout << "Saving chains before exiting ... " ;

            //// From u to xi
            vector< vector<double> > xis_911;
            for (size_t ix = 0; ix < u1_sample.size(); ++ix){
                xis_911.push_back(u2xi(nx1, jx, u1_sample[ix]));
            }

            //// From xis to rain
            vector< vector<double> > xs_911(xis_911.size(), vector<double> (xis_911[0].size()));
            for (size_t ic = 0; ic < xis_911[0].size(); ++ic){
                for (size_t ir = 0; ir < xis_911.size(); ++ir) {
                    xs_911[ir][ic] = r(xis_911[ir][ic], theta_sample[ir]);
                }
            }

            //// From rain to discharges
            vector< vector<double> > ys_911(xs_911.size(), vector<double> (xs_911[0].size()));
            vector< vector<double> > Qs_911(xs_911.size(), vector<double> (xs_911[0].size()));
            vector<double> current_theta_911(n_params);
            for (size_t ir = 0; ir < xs_911.size(); ++ir)
                Qs_911[ir][0] = 0.0;
            for (size_t ic = 1; ic < xs_911[0].size(); ++ic) {
                for (size_t ir = 0; ir < xs_911.size(); ++ir) {
                    Qs_911[ir][ic] = (1-(dt/theta_sample[ir][0])) * Qs_911[ir][ic-1] + xs_911[ir][ic-1];
                }
            }
            for (size_t ir = 0; ir < xs_911.size(); ++ir)
                ys_911[ir][0] = (theta_sample[ir][7]/theta_sample[ir][0]);
            for (size_t ic = 1; ic < xs_911[0].size(); ++ic){
                for (size_t ir = 0; ir < xs_911.size(); ++ir) {
                    current_theta_911 = theta_sample[ir];
                    ys_911[ir][ic] = (current_theta_911[7]/current_theta_911[0]) * pow(1.0-(dt/current_theta_911[0]),ic) +
                                 (1-pow(1-(dt/current_theta_911[0]),ic)) * current_theta_911[1] +
                                 A * (dt/current_theta_911[0]) * Qs_911[ir][ic];
                }
            }

            //// ****** Parameters Markov chains ******
            //// **************************************
            for (size_t ir = 0; ir < theta_sample.size(); ++ir) {
                ofs_thetas << theta_sample[ir][0] << "\t" << theta_sample[ir][1] << "\t" << theta_sample[ir][2] << "\t" <<
                           theta_sample[ir][3] << "\t" << theta_sample[ir][4] << "\t" << theta_sample[ir][5] << "\t" <<
                           theta_sample[ir][6] << "\t" << theta_sample[ir][7] << "\t" << endl;
            }
            ofs_thetas.close();

            //// ****** Energy chains ******
            //// ***************************
            for (size_t ir = 0; ir < energies.size(); ++ir) {
                ofs_energies << energies[ir] << endl;
            }
            ofs_energies.close();

            //// ****** Rain (x) ******
            //// ********************************
            for (int row = 0; row < red_ind_911.size(); ++row)
            {
                for (int col = 0; col < N1; ++col)
                {
                    ofs_pred_x << xs_911[red_ind_911[row]][col] << " ";
                }
                ofs_pred_x << endl;
            }
            ofs_pred_x.close();

            //// ****** Rainfall potential (xi) ******
            //// ********************************
            for (int row = 0; row < red_ind_911.size(); ++row)
            {
                ofs_pred_xis << red_ind_911[row] << " ";
                for (int col = 0; col < N1; ++col)
                {
                    ofs_pred_xis << xis_911[red_ind_911[row]][col] << " ";
                }
                ofs_pred_xis << endl;
            }

            ofs_pred_xis.close();

            //// ****** First point of the rainfall potential ******
            //// ********************************
            for (int row = 0; row < xis_911.size(); ++row)
            {
                ofs_pred_xi_init << xis_911[row][0] << endl;
            }
            ofs_pred_xi_init.close();

            /// ****** Last realization of the rainfall potential ******
            //// ********************************
            for (int col = 0; col < N1; ++col)
            {
                ofs_pred_xi_last << xis_911[xis_911.size()-1][col] << endl;
            }
            ofs_pred_xi_last.close();


            //// ****** Discharges (y) ******
            //// ********************************
            for (int row = 0; row < red_ind_911.size(); ++row)
            {
                for (int col = 0; col < N1; ++col)
                {
                    ofs_pred_y << ys_911[red_ind_911[row]][col] << " ";
                }
                ofs_pred_y << endl;
            }
            ofs_pred_y.close();

            cout << " DONE " << endl;

            exit(EXIT_FILE_ERROR);}

        //// ********************************************************************* ////
        //// ********************************************************************* ////
        //// ********************************************************************* ////
        //// ********************************************************************* ////

        //// ********************************************************************* ////
        //// Store current state
        //// ********************************************************************* ////
        theta_save = theta;
        u1_save = u1;

        //// ********************************************************************* ////
        //// MD integration (NAPA scheme)
        //// ********************************************************************* ////
        clock_t t1 = clock();
        for (int counter_napa = 1; counter_napa <= n_napa; ++counter_napa)
        {
            napa(nx1, ny1, jx, jy, N1, n_params, dt, dtau, m_stg, m_bdy, mp, x0, y1, mu_thetas, sigma_thetas,
                 theta, u1, pp, p1, x, stack, non_zeros_event1, zeros_event1, t_derivatives_tot);
        }
        time_napa.push_back((float)(clock()-t1)/CLOCKS_PER_SEC);

        //// ********************************************************************* ////
        //// Calculate energy of proposal state
        //// ********************************************************************* ////
        if (
                theta[0] <= K_min || theta[0] >= K_max ||
                theta[1] <= xgw_min || theta[1] >= xgw_max ||
                theta[2] <= sigma_z_min || theta[2] >= sigma_z_max ||
                theta[3] <= sigma_xi_min || theta[3] >= sigma_xi_max ||
                theta[4] <= lambda_min || theta[4] >= lambda_max ||
                theta[5] <= gamma_min || theta[5] >= gamma_max ||
                theta[6] <= xir_min || theta[6] >= xir_max ||
                theta[7] <= S0_min || theta[7] >= S0_max )
        {
            for (auto ix:theta)
                cout << theta[ix] << " ";
            cout << endl;
            theta = theta_save; u1 = u1_save;
            reject_counter += 1;
            params_lim_counter += 1;
            cout << "WARNING: parameter limits exceeded" << endl;
        }
        else if (
                find_if(u1.begin(), u1.end(), [&](double el){return (el > 50.0);}) != u1.end() ||
                find_if(u1.begin(), u1.end(), [&](double el){return (el < -50.0);}) != u1.end()
                )
        {
            theta = theta_save; u1 = u1_save;
            reject_counter += 1;
            u_lim_counter += 1;
            cout << "WARNING: u limits exceeded" << endl;
        }
        else
        {
            t_energy_calc = clock();
            V_new = V_N(nx1, jx, dt, u1) +
                    V_n(nx1, ny1, jx, jy, dt, x0, y1, theta, u1, non_zeros_event1, zeros_event1) +
                    V_1(N1, nx1, jx, dt, u1)  + V_p(theta, mu_thetas, sigma_thetas);
            t_energy_tot[0] += 1;
            t_energy_tot[1] += ((float)(clock()-t_energy_calc)/CLOCKS_PER_SEC);
            H_new = vreduce(vdiv(vsquare(pp), ctimes(2.0, mp))) +
                    vreduce(vdiv(vsquare(p1), ctimes(2.0, m1))) + V_new;

            accept_prob = min(1.0, exp(H_old-H_new));
            if (u_rand() > accept_prob)
            {
                theta = theta_save; u1 = u1_save;
                reject_counter += 1;
                energies.push_back(H_old);
            } else {
                energies.push_back(H_new);
                V_old = V_new;
            }
        }

        theta_sample.push_back(theta);
        u0_sample.push_back(u1[0]);
        if (counter == red_ind[red_ind_ix]){
            u1_sample.push_back(u1);
            red_ind_ix += 1;
        }

        //// ********************************************************************* ////
        //// Resampling xi0 at zero-rain points
        //// ********************************************************************* ////
        //// COMMENT THIS OUT IF INFERENCE IS DONE WITHOUT DATA
        //// OR IF USING CDF OF NORMAL DISTRIBUTION FOR ZERO-RAIN POINTS
        /*for (size_t ix = 0; ix < zeros_event1.size(); ++ix){
            xi01[zeros_event1[ix]-1] = sample_from_truncated_normal(theta[6], u1[zero_ind_1[ix]-1], theta[3]);
        };*/

        //// ********************************************************************* ////
        //// Checkpointing
        //// ********************************************************************* ////
        if (counter%1000 == 0)
            cout << "== Step " << counter << " of " << n_samples << " == Total time: " <<
            ((float)(clock()-t_init)/CLOCKS_PER_SEC) << " seconds" << endl;
    }

    double time_NAPA = accumulate(time_napa.begin(), time_napa.end(), 0.0);

    cout << endl;
    cout << "*************************************************" << endl;
    cout << "*** HMC loops completed in " << ((float)(clock()-t_init)/CLOCKS_PER_SEC) << " seconds ***" << endl;
    cout << "*** with " << time_NAPA << " seconds in NAPA ***" << endl;
    cout << "*** Rejection rate ==> " << (double)reject_counter/n_samples * 100.0 << " % ***" << endl;
    cout << "*** Total number of energy calculations ==> " << (float)(t_energy_tot[0]) << " in " << t_energy_tot[1]
        << " seconds (==> " << t_energy_tot[1]/(t_energy_tot[0])*1000 << " ms / evaluation)" << endl;
    cout << "*** Total number of derivative calculations (NAPA) ==> " << (float)t_derivatives_tot[0] << " in "
        << t_derivatives_tot[1] << " seconds (==> " << t_derivatives_tot[1]/(t_derivatives_tot[0]) * 1000 << " ms / evaluation)" << endl;
    // cout << "*** Parameter limits exceeded in " << params_lim_counter << " of " << n_samples << " steps ***" << endl;
    // cout << "*** Coordinate u limits exceeded in " << u_lim_counter << " of " << n_samples << " steps ***" << endl;
    cout << "*************************************************" << endl;
    cout << endl;

    //// ************************************************************************* ////
    //// ****************** Coordinate transformation: u -> xi ******************* ////
    //// ******************* (Tuckerman et al., JCP 99, 1993) ******************** ////
    //// ************************************************************************* ////
    clock_t t_transform_u2xi = clock();
    cout << " *** Transforming u -> xis ... ";
    vector< vector<double> > xis;
    for (size_t ix = 0; ix < u1_sample.size(); ++ix){
        xis.push_back(u2xi(nx1, jx, u1_sample[ix]));
    }
    cout << " DONE in " << ((float)(clock()-t_transform_u2xi)/CLOCKS_PER_SEC) << " seconds ***" << endl;

    //// From xis to rain
    clock_t t_transform_xi2x = clock();
    cout << " *** Transforming xis -> rain ... ";
    vector< vector<double> > xs(xis.size(), vector<double> (xis[0].size()));
    for (size_t ic = 0; ic < xis[0].size(); ++ic){
        for (size_t ir = 0; ir < xis.size(); ++ir) {
            xs[ir][ic] = r(xis[ir][ic], theta_sample[red_ind_with_0[ir]]);
        }
    }
    cout << " DONE in " << ((float)(clock()-t_transform_xi2x)/CLOCKS_PER_SEC) << " seconds ***" << endl;

    //// From rain to discharges
    clock_t t_transform_x2y = clock();
    cout << " *** Transforming rain -> discharges ... ";
    vector< vector<double> > ys(xs.size(), vector<double> (xs[0].size()));
    vector< vector<double> > Qs(xs.size(), vector<double> (xs[0].size()));
    vector<double> current_theta(n_params);
    for (size_t ir = 0; ir < xs.size(); ++ir)
        Qs[ir][0] = 0.0;
    for (size_t ic = 1; ic < xs[0].size(); ++ic) {
        for (size_t ir = 0; ir < xs.size(); ++ir) {
            Qs[ir][ic] = (1-(dt/theta_sample[red_ind_with_0[ir]][0])) * Qs[ir][ic-1] + xs[ir][ic-1];
        }
    }
    for (size_t ir = 0; ir < xs.size(); ++ir)
        ys[ir][0] = (theta_sample[red_ind_with_0[ir]][7]/theta_sample[red_ind_with_0[ir]][0]);
    for (size_t ic = 1; ic < xs[0].size(); ++ic){
        for (size_t ir = 0; ir < xs.size(); ++ir) {
            current_theta = theta_sample[red_ind_with_0[ir]];
            ys[ir][ic] = (current_theta[7]/current_theta[0]) * pow(1.0-(dt/current_theta[0]),ic) +
                    (1-pow(1-(dt/current_theta[0]),ic)) * current_theta[1] +
                    A * (dt/current_theta[0]) * Qs[ir][ic];
        }
    }
    cout << " DONE in " << ((float)(clock()-t_transform_x2y)/CLOCKS_PER_SEC) << " seconds ***" << endl;

    //// ************************************************************************* ////
    //// ***************************** Save results ****************************** ////
    //// ************************************************************************* ////

    clock_t t_saving = clock();
    cout << " *** Writing outputs to file ... ";

    //// ****** Parameters Markov chains ******
    //// **************************************
    for (size_t ir = 0; ir < theta_sample.size(); ++ir) {
        ofs_thetas << theta_sample[ir][0] << "\t" << theta_sample[ir][1] << "\t" << theta_sample[ir][2] << "\t" <<
                   theta_sample[ir][3] << "\t" << theta_sample[ir][4] << "\t" << theta_sample[ir][5] << "\t" <<
                   theta_sample[ir][6] << "\t" << theta_sample[ir][7] << "\t" << endl;
    }
    ofs_thetas.close();

    //// ****** Energy chains ******
    //// ***************************
    for (size_t ir = 0; ir < energies.size(); ++ir) {
        ofs_energies << energies[ir] << endl;
    }
    ofs_energies.close();

    //// ****** Rain (x) ******
    //// ********************************
    for (int row = 0; row < xs.size(); ++row)
    {
        for (int col = 0; col < N1; ++col)
        {
            ofs_pred_x << xs[row][col] << " ";
        }
        ofs_pred_x << endl;
    }
    ofs_pred_x.close();

    //// ****** Rainfall potential (xi) ******
    //// ********************************
    for (int row = 0; row < xis.size(); ++row)
    {
        ofs_pred_xis << red_ind_with_0[row] << " ";
        for (int col = 0; col < N1; ++col)
        {
            ofs_pred_xis << xis[row][col] << " ";
        }
        ofs_pred_xis << endl;
    }
    ofs_pred_xis.close();

    //// ****** First point of the rainfall potential ******
    //// ********************************
    for (int row = 0; row < n_samples; ++row)
    {
        ofs_pred_xi_init << u0_sample[row] << endl;
    }
    ofs_pred_xi_init.close();

    /// ****** Last realization of the rainfall potential ******
    //// ********************************
    for (int col = 0; col < N1; ++col)
    {
        ofs_pred_xi_last << xis[xis.size()-1][col] << endl;
    }
    ofs_pred_xi_last.close();

    //// ****** Discharges (y) ******
    //// ********************************
    for (int row = 0; row < ys.size(); ++row)
    {
        for (int col = 0; col < N1; ++col)
        {
            ofs_pred_y << ys[row][col] << " ";
        }
        ofs_pred_y << endl;
    }
    ofs_pred_y.close();

    /*//// ********* xi evolution *********
    ofstream ofs_xi_chains(outdir + "xi_chains_" + output_file_name + ".dat");
    if (!ofs_xi_chains.is_open())
    {
        cout << "Exiting: unable to open file to write xi chains" << endl;
        exit(EXIT_FILE_ERROR);
    }
    int xi_chain_step = (obs_ind_1[1] - obs_ind_1[0])/2;
    for (int row = 0; row < xis.size(); ++row)
    {
        for (int col = 0; col < obs_ind_1.size() - 1; ++col)
        {
            ofs_xi_chains << xis[row][obs_ind_1[col]] << " " << xis[row][obs_ind_1[col] + xi_chain_step] << " ";
        }
        ofs_xi_chains << xis[row][obs_ind_1[obs_ind_1.size() - 1]] << endl;
    }
    ofs_xi_chains.close();*/

    cout << " DONE in " << ((float)(clock()-t_saving)/CLOCKS_PER_SEC) << " seconds ***" << endl;

    //// ************************************************************************* ////

    return 0;
}
