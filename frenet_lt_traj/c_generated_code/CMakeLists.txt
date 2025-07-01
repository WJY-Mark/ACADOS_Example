cmake_minimum_required(VERSION 3.1)
project(acados_test LANGUAGES C CXX)

# for macOS
set(CMAKE_MACOSX_RPATH 1)

# set compiler standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -DEXT_DEP -fdiagnostics-show-option")

# include Eigen3
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})

# include some more paths
include_directories($ENV{ACADOS_SOURCE_DIR}/include)
include_directories($ENV{ACADOS_SOURCE_DIR}/include/blasfeo/include)
include_directories($ENV{ACADOS_SOURCE_DIR}/include/hpipm/include)
include_directories($ENV{ACADOS_SOURCE_DIR}/include/acados)
# include_directories(../python/c_generated_code)

# load acados libs
link_directories($ENV{ACADOS_SOURCE_DIR}/lib)

file(GLOB ocp_solver
        ./acados_solver_frenet_lt.c
        )

file(GLOB casadi_fun
        ./frenet_lt_model/frenet_lt_expl_ode_fun.c
        ./frenet_lt_model/frenet_lt_expl_vde_forw.c
        ./frenet_lt_model/frenet_lt_expl_vde_adj.c)

file(GLOB sim_solver
        ./acados_sim_solver_frenet_lt.c
        )
# Define the example
## these two are only basic tests
#add_executable(main_sim_frenet_lt ../python/c_generated_code/main_sim_frenet_lt.c
#        ${casadi_fun}
#        ${sim_solver}
#        )
#target_link_libraries(main_sim_frenet_lt acados hpipm blasfeo)
# # only
#add_executable(main_frenet_lt ../python/c_generated_code/main_frenet_lt.c
#        ${casadi_fun}
#        ${sim_solver}
#        ${ocp_solver})
#target_link_libraries(main_frenet_lt acados hpipm blasfeo)
#add_test(main_frenet_lt main_frenet_lt)

# create libraries
add_library(ocp_shared_lib SHARED ${ocp_solver} ${casadi_fun} )
target_link_libraries(ocp_shared_lib acados hpipm blasfeo)

add_library(sim_shared_lib SHARED ${sim_solver} ${casadi_fun})
target_link_libraries(sim_shared_lib acados hpipm blasfeo)

add_executable(main_frenet_lt main_frenet_lt.c )
target_link_libraries(main_frenet_lt ocp_shared_lib sim_shared_lib)