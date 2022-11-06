////
////  Bayesian parameter inference in rainfall-runoff hydrological modelling
////  using a Hamiltonian Monte Carlo approach with a time-scale separation
///   and a stochastic rain model
////
////  Created by Ulzega Simone, October 2022
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
//// The most reliable way (as of June 2021) to get a copy of Boost
//// is to download a distribution from SourceForge:
//// 1. Download boost_1_75_0.tar.bz2.
//// 2. In the directory where you want to put the Boost installation, execute:
//// tar --bzip2 -xf /path/to/boost_1_75_0.tar.bz2
////
//// > cd boost_1_75_0
//// > ./bootstrap.sh
//// > ./b2
//// > ./b2 install
////
//// ********************************************************************* ////
//// ********************************************************************* ////

//// ********************************************************************* ////
//// ********************* Data-less or no-rain tests ******************** ////
//// ********************************************************************* ////

//// Just one thing to do: modify "hmc_functions.cpp" and "adept_functions.cpp"
//// by simply commenting out the data-dependent components
//// (detailed instructions in the functions)

//// ********************************************************************* ////
//// ********************************************************************* ////

//// *************************** Global variables ************************* ////
//// ******************* (declared in global_variables.h) ***************** ////
double A, alpha, beta, tau, pi, inf;
//// A = catchment area
//// alpha, beta = coefficients of the heteroscedasticity transformation
//// tau = correlation time of the OU process (stochastic process describing the rain)
//// pi = 3.14....
//// inf = infinity
//// ********************************************************************** ////

//// ****************************** FUNCTIONS ***************************** ////
//// ********************************************************************** ////
//// To sample from truncated normal
////
//// THIS WAS USED IN AN OLD VERSION OF THE CODE (with Gibbs sampling)
//// I KEEP IT HERE ALTHOUGH WE DON'T NEED IT.
//// ONE NEVER KNOWS WHAT MIGHT BE USEFUL AGAIN IN THE FUTURE!
////
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

//// N.B.: functions are seeded here with constant values.
//// Seeds should be changed to generate different random chains.
int seed1 = 1, seed2 = 2, seed3 = 3;
//// **********************************************************************
//// To fill in a vector with normally distributed random numbers
//// (used to generate momenta in the HMC loop)
void norm_rand(vector<double> & vec, double mu = 0.0, double std = 1.0)
{
    static mt19937_64 engine2(seed1);
    // static mt19937_64 engine(time(NULL));
    static normal_distribution<double> norm_dist_vec(mu, std);
    for (size_t ix = 0; ix < vec.size(); ++ix)
        vec[ix] = norm_dist_vec(engine2);
}
//// **********************************************************************

//// **********************************************************************
//// To generate random real numbers from normal distribution N[0,1]
//// (used to generate initial OU state)
double n_rand(double mu = 0.0, double std = 1.0)
{
    static mt19937_64 engine3(seed2);
    // static mt19937_64 engine(time(NULL));
    static normal_distribution<double> norm_dist(mu, std);
    double res = norm_dist(engine3);
    return res;
}
//// **********************************************************************

//// **********************************************************************
//// To generate random real numbers from uniform distribution U[0,1]
//// (used in the Metropolis accept-reject step)
double u_rand(double a = 0.0, double b = 1.0)
{
    static mt19937_64 engine4(seed3);
    // static mt19937_64 engine(time(NULL));
    static uniform_real_distribution<double> uniform_dist(a,b);
    double res = uniform_dist(engine4);
    return res;
}
//// **********************************************************************
//// *************************** END OF FUNCTIONS ************************* ////

//// ************************* Useful directories ************************* ////
// LOCAL:
const string datadir  = "/Users/ulzg/CLionProjects/HMC_SIP/data/";  // Input data (observations)
const string outdir  = "/Users/ulzg/CLionProjects/HMC_SIP/output/"; // HMC-generated outputs
// CLUSTER:
//const string datadir  = "/cfs/earth/scratch/ulzg/hmc_sip_norain/data/";
//const string outdir  = "/cfs/earth/scratch/ulzg/hmc_sip_norain/output/";
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

    bool save_init_polymer = false;  // Flag to 'save' or 'not save' initial configuration (in data directory)
    bool continue_from_previous = false;  // Flag to continue from previous file or start from scratch
    const string previous_file_name = "type_here_file_name_to_continue_from";  // Used only if continue_from_previous = true
    bool use_sc2 = true;  // Flag to select scenario Sc2 (inaccurate) or Sc1 (accurate)
    string scenario;
    // Set the selected scenario
    if (use_sc2){
        scenario = "Sc2";
    } else {
        scenario = "Sc1";
    }

    //// ********************************************************************* ////
    //// ***************************** LOAD DATA ***************************** ////
    //// ********************************************************************* ////
    //// N.B.: Original work by Del Giudice et al., Water Resources Research 52, 2016,
    //// included 3 rain events, each including 1 discharge and 2 precipitation time-series (Sc1 and Sc2)
    //// for a total of 9 data files.
    //// HERE: INFERENCE IS BASED ONLY ON EVENT 1. The other observations (events 2 and 3)
    //// are still in the data folder, but ARE NOT USED.

    cout << "*** Loading data ***" << endl;

    //// **************************** Rain **************************** ////
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
    x01.erase(x01.begin());  // to delete headers

    // Close file streams
    ifs_rain1.close();

    //// ************************** Discharge ************************** ////
    //// Discharge data is the same for both Sc1 and Sc2
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
    y1.erase(y1.begin());  // to delete headers

    // Close file streams
    ifs_out1.close();

    //// ******************************************************************************** ////
    //// GETTING RID OF EXTRA POINTS
    //// This is done to have rain and discharge observations start and end at the same time point ////
    //// IT IS NOT IDEAL, ONLY A TEMPORARY SOLUTION,
    //// THE CODE SHOULD BE ADAPTED TO ALLOW DIFFERENT INITIAL AND FINAL TIMES
    //// MAYBE ONE DAY ...

    // In Sc1:
    // Rain observation times: {1053., 1054., 1055., 1056., 1057., 1058., 1059., 1060., 1061., ...
    // ... 1292., 1293., 1294., 1295., 1296., 1297., 1298., 1299., 1300., 1301., 1302., 1303.}
    // Discharge observation times: {1056., 1060., 1064., 1068., ... 1284., 1288., 1292., 1296., 1300.}
    // -> we delete the first and the last 3 precipitation data points so that both
    // precipitation and discharge begin at time 1056 and end at time 1300
    if (!use_sc2){
        x01.erase(x01.begin(),x01.begin()+3);
        x01.erase(x01.end() - 3, x01.end());
    }

    // In Sc2:
    // Rain observation times: {1060., 1070., 1080., 1090., ... 1250., 1260., 1270., 1280., 1290., 1300.}
    // Discharge observation times (as in Sc1): {1056., 1060., 1064., 1068., ... 1284., 1288., 1292., 1296., 1300.}
    // -> we delete the first data point of the discharge time-series, so that both
    // precipitation and discharge begin at time 1060 and end at time 1300
    if (use_sc2){
        y1.erase(y1.begin());
    };

    //// A FEW MORE THINGS TO DO BEFORE WE ARE ALL SET...

    //// To have rain inputs in mm/s instead of mm/min
    for (size_t ix = 0; ix < x01.size(); ++ix)
        x01[ix][1] = x01[ix][1]/60.;

    //// Create a vector containing observed rain without times
    vector<double> x0;
    for (size_t ix = 0; ix < x01.size(); ++ix)
        x0.push_back(x01[ix][1]);

    //// ********************************************************************* ////

    //// ************************ Get measurements times ************************ ////
    //// This is actually not really necessary.
    /// However, since it doesn't bother anyone, I'll keep it here.
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

    // Print file dimensions
    cout << "Dimensions of rain event datasets: (" << x01.size() << "," << x01[0].size() << ")" << endl;
    cout << "Dimensions of discharge datasets: (" << y1.size() << "," << y1[0].size() << ")" << endl;
    cout << endl;

    //// ********************************************************************* ////
    //// ************************* END of DATA LOADING *********************** ////
    //// ********************************************************************* ////

    //// ********************************************************************* ////
    //// ***************************** PARAMETERS **************************** ////
    //// ********************************************************************* ////
    const int n_params = 8;

    //// Parameters are (LN = log-normal prior, N = normal prior):
    /// K (LN), xgw (LN), sigma_z (LN), sigma_xi (LN), lambda (LN), gamma (LN), xir (N), S0 (N)

    //// ****************** Model parameters: fixed values ****************** ////
    // Fixed parameters are global variables and are declared in global_variables.h
    A = 11815.8;   // Area [m^2]
    alpha = 25.0;  // Coefficients of the discharge heteroscedasticity transformation H [l/s]
    beta = 50.0;   // Coefficients of the discharge heteroscedasticity transformation H [l/s]
    tau = 636.0;     // Auto correlation coefficient of the OU process [s]
    pi = boost::math::constants::pi<double>();
    inf = std::numeric_limits<double>::infinity();

    //// **************** Model parameters: mu and sigma for priors **************** ////
    vector<double> mu_thetas_temp(n_params), mu_thetas(n_params);
    vector<double> sigma_thetas_temp(n_params), sigma_thetas(n_params);

    //// Parameters mean and sd (they need to be transformed for LN distributions)
    mu_thetas_temp = {284.4, 6.0, 4.5, 0.65, 0.1/60.0, 0.5, 0.5, 0.0};
    sigma_thetas_temp = {57.6, 1.0, 0.45, 0.3, 0.05/60.0, 0.25, 0.1, 5000.0};

    mu_thetas = mu_thetas_temp;
    sigma_thetas = sigma_thetas_temp;

    for (size_t ix = 0; ix < 6; ++ix){
        mu_thetas[ix] = log(pow(mu_thetas_temp[ix],2)/sqrt(pow(mu_thetas_temp[ix],2)+
                                                           pow(sigma_thetas_temp[ix],2)));
        sigma_thetas[ix] = sqrt(log(1+(pow(sigma_thetas_temp[ix],2)/pow(mu_thetas_temp[ix],2))));
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
    //// Priors, mean and sd for the parameters
    //// (numbers in parenthesis are mean and sd of the random variable)
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
    const int nx1= x01.size() - 1; // Number of data points is nx1 + 1
    const int ny1= y1.size() - 1;  // Number of data points is ny1 + 1

    int jx;  // Number of sub-intervals between consecutive data points

    if (use_sc2){
        jx = 60;  // Delta_t between data points = 10 minutes = 600 seconds -> fine grid step dt = 10 seconds
    } else {
        jx = 6;  // Delta_t between data points = 1 minute = 60 seconds -> fine grid step dt = 10 seconds
    }

    const int jy = 24;  // Delta_t between data points = 4 minutes = 240 seconds -> fine grid step dt = 10 seconds


    const int N1 = nx1 * jx + 1;

    cout << "*** Total number of discretization points (Nx): " << N1 << " ***" << endl;

    //// *********************** Time step ********************** ////
    double dt = round((tx01[1]-tx01[0])/jx *3600);

    cout << "*** Time unit dt: " << dt << " seconds ***" << endl;
    cout << endl;

    //// ********************************************************************* ////
    //// ********************** END of PARAMETERS SECTION ******************** ////
    //// ********************************************************************* ////

    //// ********************************************************************* ////
    //// ******************* INDEXES OF MEASUREMENT POINTS ****************** ////
    //// ************************ AND ZERO-RAIN POINTS *********************** ////
    //// ********************************************************************* ////

    //// Indexes of data points in the discretized framework (size N)
    //// Indexes start at 1
    //// (P.S.: this might not be used anywhere, but I would keep it. 
    //// Might be useful for debugging or just curiosity.)
    
    vector<size_t> obs_ind_1(nx1+1);
    for (size_t ix = 0; ix < nx1+1; ++ix)
        obs_ind_1[ix] = (int)ix * jx + 1;

    /*for (auto i: obs_ind_1)
        cout << i << ' ';
    return 0;*/

    //// ********************************************************************* ////

    //// Indices of zero- and non-zero-rain points (indices in the data files of size nx + 1)
    //// Indices start at 1
    vector<size_t> zeros;
    auto it1 = find_if(x01.begin(), x01.end(), [](vector<double> v){return v[1] == 0;});
    while (it1 != x01.end()) {
        zeros.push_back(distance(x01.begin(), it1) + 1);
        it1 = find_if(next(it1), x01.end(), [](vector<double> v){return v[1] == 0;});
    }

    vector<size_t> non_zeros;
    it1 = find_if(x01.begin(), x01.end(), [](vector<double> v){return v[1] != 0;});
    while (it1 != x01.end()) {
        non_zeros.push_back(distance(x01.begin(), it1) + 1);
        it1 = find_if(next(it1), x01.end(), [](vector<double> v){return v[1] != 0;});
    }

    /*for (auto i: zeros)
        cout << i << ' ';
    cout << endl;
    for (auto i: non_zeros)
        cout << i << ' ';
    cout << endl;
    return(0);*/
    //// ********************************************************************* ////

    //// Now zero-rain indices in the discretized framework (size N)
    //// As usual, indices start at 1
    vector<size_t> zeros_N(zeros.size());
    for (time_t ix = 0; ix < zeros.size(); ++ix)
        zeros_N[ix] = ((int)zeros[ix] - 1) * jx + 1;

    cout << setprecision(3) << "*** Number of zero-rain points: "
         << zeros.size() << "(" << (double)zeros.size()/(nx1+1)*100 << "%) " << "***" << endl;
    cout << endl;
    //// ********************************************************************* ////

    //// ************************************************************************* ////
    //// ***************************** INITIAL STATE ***************************** ////
    //// ************************ (linear interpolation) ************************* ////
    //// ************************************************************************* ////

    cout << "*** Generating initial configuration ***" << endl;

    //// Container for rainfall potential in the discretized framework (size N)
    vector<double> xi1(N1);

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

        //// Run OU process - Initial state is just a random realization of an OU process
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
    //// ****************** COORDINATE TRANSFORMATION: xi -> u ******************* ////
    //// ************ (eqs 2.16-17 in Tuckerman et al., JCP 99, 1993) ************ ////
    //// ************************************************************************* ////

    cout << "*** Coordinates transformation xi -> u ***" << endl;

    //// Container for discretized rainfall potential in
    //// the space of transformed coordinates u
    vector<double> u1(N1);

    /// IMPORTANT: the transformation that should be applied to the intermediate points is:
    //// u[s*j+k-1] = xi[s*j+k-1] - ( (k-1)*xi[s*j+k] + xi[s*j] )/k;
    //// BUT: it is always = 0 when the xi[sj+k-1] between data points are
    //// a linear interpolation of the data points xi[sj]
    //// When discretization points are a linear interpolation of data points, therefore one could use
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
    //// ********************** PREPARE CONTAINERS FOR HMC *********************** ////
    //// ************************************************************************* ////

    cout << "*** Preparing HMC ***" << endl;

    //// Containers for Markov chains
    vector< vector<double> > theta_sample;  // store parameter chains
    vector< vector<double> > u1_sample;     // store rain realizations (coordinates u)
    vector<double> energies;                // store energies
    vector<double> u0_sample;               // store first point of each rain realization (coordinates u)

    //// +++++++ STORE INITIAL VALUES ++++++++ ////
    theta_sample.push_back(theta);
    u1_sample.push_back(u1);
    //// +++++++++++++++++++++++++++++++++++++ ////

    //// Containers for temporary values during HMC steps
    vector<double> theta_save;
    vector<double> u1_save;

    //// Containers for masses and momenta
    vector<double> mp(n_params), m1(N1);
    vector<double> sqrt_mp(n_params), sqrt_m1(N1);
    vector<double> pp, p1;

    //// Containers for random numbers (to generate momenta p)
    vector<double> rand_vec_p(n_params), rand_vec_1(N1);

    //// Containers for energies and Metropolis accept/reject probability
    double V_old, V_new, H_old, H_new;
    double accept_prob;

    //// Containers for execution times
    vector<double> time_napa;

    //// Containers for AD
    adept::Stack stack;
    vector<adouble> x(n_params+N1);

    //// Containers for computing times
    vector<double> t_energy_tot(2, 0.0);
    vector<double> t_derivatives_tot(2, 0.0);
    // These are used to estimate the contributions of energy calculations (-> function evaluations) compared to
    // the derivative estimations (in the 'napa' integration loop). We store both number of evaluations and time.

    //// ************************************************************************* ////
    //// *********************** Get ready to save results *********************** ////
    //// ************************************************************************* ////

    //// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ////
    const string output_file_name = "output_files_name";  // name of output files, choose it carefully!
    //// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ////

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

    //// _________________________________________________________________________ ////
    //// _________________________________________________________________________ ////
    //// ************************************************************************* ////
    //// ************************************************************************* ////
    //// ****************************** HMC BLOCK ******************************** ////
    //// ************************************************************************* ////
    //// ************************************************************************* ////
    //// _________________________________________________________________________ ////
    //// _________________________________________________________________________ ////

    //// *********************************************************************** ////
    //// ************************ DEFINITION OF MASSES ************************* ////
    //// *********************************************************************** ////
    //// For a data-less (or rain-less) run, recommended masses are:
    double m_bdy = 1.0;   // m = m_q / dt
    double m_stg = 1.0;   // we assume m_q proportional to dt ==> m = costant
    //// For a normal run with data, recommended masses are:
    //double m_bdy = 1.6;   // m = m_q / dt
    //double m_stg = 0.4;   // we assume m_q proportional to dt ==> m = costant

    //// Masses for parameters (used for data-less, rain-less and normal [with data] runs)
    mp[0] = 1e-5;  // refers to K
    mp[1] = 1.0;  // refers to Qgw
    mp[2] = 1.0;  // refers to sigma_z
    mp[3] = 1.0;  // refers to sigma_xi
    mp[4] = 2e5;  // refers to lambda
    mp[5] = 0.5;  // refers to gamma
    mp[6] = 15.0;  // refers to xir
    mp[7] = 1e-7;  // refers to S0

    //// Assign masses to coordinates
    for (int s = 0; s < nx1; ++s){
        m1[s*jx] = m_bdy;  // data points
        for (int k = 2; k <= jx; ++k)
            m1[s*jx + k - 1] = m_stg;   // discretization points
    }
    m1[N1-1] = m_bdy;

    //// Square roots (needed below in the HMC loop to generate momenta)
    sqrt_mp = vsqrt(mp);
    sqrt_m1 = vsqrt(m1);

    //// *********************************************************************** ////
    //// ************************ SIMULATION PARAMETERS ************************ ////
    //// *********************************************************************** ////
    const int n_samples = 75000;  // Total number of points in the MCMC
    const double dtau = 0.015;  // Integration time step
    const int n_napa = 3;       //

    //// Prepare indices of rain and discharge realizations to be saved
    //// (we save only a selection due to memory limitations)
    int red_size = 2e4;    // number of realizations to be saved
    red_size = min(red_size, n_samples);

    double steppo = (double)(n_samples) / (double)(red_size);

    vector<size_t> red_ind;  // indices of realizations to be saved
    for (size_t ix = 1; ix <= red_size; ++ix){
        red_ind.push_back(round(ix*steppo));
    }

    vector<size_t> red_ind_with_0;
    for (size_t ix = 0; ix <= red_size; ++ix){
        red_ind_with_0.push_back(round(ix*steppo));
    }

    //// ************************************************************************* ////
    //// ************************************************************************* ////
    //// ****************************** HMC LOOP ********************************* ////
    //// ************************************************************************* ////
    //// ************************************************************************* ////

    cout << endl << " ======================================================== " << endl;
    cout << "*** Starting HMC loops ***" << endl;
    cout << " ======================================================== " << endl << endl;

    int reject_counter = 0;
    int params_lim_counter = 0;
    int u_lim_counter = 0;
    int red_ind_ix = 0;

    clock_t t_init = clock();  // To estimate total simulation time
    clock_t t_energy_calc;     // To estimate energy evaluations (or 'functions evaluations') time

    for (int counter = 1; counter <= n_samples; ++counter){

        //// ********************************************************************* ////
        //// Sample momenta
        //// ********************************************************************* ////
        norm_rand(rand_vec_p);  // for model parameters to be inferred
        norm_rand(rand_vec_1);  // for coordinates (= discretization and data points)
        // From Albert et al., PRE 93, 2016:
        // Momenta are drawn from normal distributions:
        // p^2/(2m)
        // So, sigma -> sqrt(m), and thus, p = sqrt_m * random_normal
        //
        pp = vtimes(sqrt_mp, rand_vec_p); // sqrt of mass * random vector
        p1 = vtimes(sqrt_m1, rand_vec_1);

        /*for (auto ix: theta)
            cout << ix << " ";
        cout << endl;*/

        //// ********************************************************************* ////
        //// Calculate energy
        //// THE POTENTIAL ENERGY V IS CALCULATED ONLY ONCE, AT THE END OF THE LOOP
        //// EXCEPT FOR THE FIRST STEP OF THE CHAIN
        //// ********************************************************************* ////
        if (counter==1){
            //// This is done only once, at the first step of the chain.
            /// Afterwards, V_old is stored at the end of this loop
            V_old = V_N(nx1, jx, dt, u1) +
                    V_n(nx1, ny1, jx, jy, dt, x0, y1, theta, u1, non_zeros, zeros) +
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
        //// End of safe way out
        //// ********************************************************************* ////
        //// ********************************************************************* ////

        //// ********************************************************************* ////
        //// Store current state for parameters and coordinates
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
                 theta, u1, pp, p1, x, stack, non_zeros, zeros, t_derivatives_tot);
        }
        time_napa.push_back((float)(clock()-t1)/CLOCKS_PER_SEC);

        //// Side note: if you are wondering why Sc1 is faster than Sc2 although N is the same for both scenarios...
        //// The ANSWER is ... napa depends on jx!
        //// Not only the analytical part, which is way faster and does not affect the total performance, but also
        //// inside V_1 and V_n there are transformations xi <-> u which depend on jx

        //// ********************************************************************* ////
        //// Calculate energy of proposal state, if possible
        //// ********************************************************************* ////
        if (
                //// Parameter limits exceeded! ////
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
            theta = theta_save; u1 = u1_save;  // restore previous values
            reject_counter += 1;
            params_lim_counter += 1;
            cout << "WARNING: parameter limits exceeded" << endl;
        }
        else if (
                //// u-limits exceeded! ////
                find_if(u1.begin(), u1.end(), [&](double el){return (el > 50.0);}) != u1.end() ||
                find_if(u1.begin(), u1.end(), [&](double el){return (el < -50.0);}) != u1.end()
                )
        {
            theta = theta_save; u1 = u1_save; // restore previous values
            reject_counter += 1;
            u_lim_counter += 1;
            cout << "WARNING: u limits exceeded" << endl;
        }
        else
        {
            t_energy_calc = clock();
            V_new = V_N(nx1, jx, dt, u1) +
                    V_n(nx1, ny1, jx, jy, dt, x0, y1, theta, u1, non_zeros, zeros) +
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
        //// THIS WAS AN OLD VERSION OF THE CODE (GIBBS SAMPLING)
        //// WE DO NOT NEED THIS PART ANY MORE
        //// BUT I'D RATHER KEEP IT HERE, ONE NEVER KNOWS...
        //// ********************************************************************* ////
        //// COMMENT THIS OUT IF INFERENCE IS DONE WITHOUT DATA
        /*for (size_t ix = 0; ix < zeros.size(); ++ix){
            xi01[zeros[ix]-1] = sample_from_truncated_normal(theta[6], u1[zeros_N[ix]-1], theta[3]);
        };*/

        //// ********************************************************************* ////
        //// Checkpointing
        //// ********************************************************************* ////
        if (counter%1000 == 0)
            cout << "== Step " << counter << " of " << n_samples << " with " << reject_counter
            << " rejected samples == Total time: " <<
            ((float)(clock()-t_init)/CLOCKS_PER_SEC) << " seconds" << endl;
    }

    //// ************************************************************************* ////
    //// ************************************************************************* ////
    //// *************************** END OF HMC LOOP ***************************** ////
    //// ************************************************************************* ////
    //// ************************************************************************* ////

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

    //// **************************************************************** ////
    //// **************************************************************** ////
    //// ****************** PREPARING TO SAVE RESULTS ******************* ////
    //// **************************************************************** ////
    //// **************************************************************** ////

    //// Coordinate transformation: u -> xi (Tuckerman et al., JCP 99, 1993)
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

    cout << " DONE in " << ((float)(clock()-t_saving)/CLOCKS_PER_SEC) << " seconds ***" << endl;

    //// ************************************************************************* ////

    return 0;
}