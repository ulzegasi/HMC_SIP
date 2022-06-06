cmake_minimum_required(VERSION 3.17)
project(HMC_SIP)

set(CMAKE_CXX_STANDARD 14)

add_executable(HMC_SIP main.cpp hmc_functions.cpp hmc_functions.h heteroscedasticity_function.cpp heteroscedasticity_function.h global_variables.h rainfall_potential_transformation.cpp rainfall_potential_transformation.h xi_u_transformation.cpp xi_u_transformation.h napa.cpp napa.h adept_functions.cpp adept_functions.h)
include_directories("../boost_1_75_0")
include_directories("./adept-1.1/include")
# target_link_libraries(HMC_SIP "/Users/ulzg/CLionProjects/boost_1_75_0/stage/lib")
target_link_libraries(HMC_SIP ${Boost_LIBRARIES})