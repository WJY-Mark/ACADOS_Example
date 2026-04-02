cmake_minimum_required(VERSION 3.10)
project(esa_mpc_ocp_main C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2")

set(ACADOS_DIR $ENV{HOME}/acados)
set(ACADOS_INCLUDE ${ACADOS_DIR}/include)
set(ACADOS_LIB ${ACADOS_DIR}/lib)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ACADOS_INCLUDE}
    ${ACADOS_INCLUDE}/acados
    ${ACADOS_INCLUDE}/blasfeo/include
    ${ACADOS_INCLUDE}/hpipm/include
)

link_directories(${ACADOS_LIB})

# Model sources
set(MODEL_SRC
    esa_mpc_ocp_model/esa_mpc_ocp_expl_ode_fun.c
    esa_mpc_ocp_model/esa_mpc_ocp_expl_vde_forw.c
    esa_mpc_ocp_model/esa_mpc_ocp_expl_vde_adj.c
)

# OCP solver source
set(OCP_SRC acados_solver_esa_mpc_ocp.c)

# Sim solver source
set(SIM_SRC acados_sim_solver_esa_mpc_ocp.c)

# Shared library: bundles model + sim + ocp
add_library(acados_solver_esa_mpc_ocp SHARED ${MODEL_SRC} ${SIM_SRC} ${OCP_SRC})
target_link_libraries(acados_solver_esa_mpc_ocp acados hpipm blasfeo m)
set_target_properties(acados_solver_esa_mpc_ocp PROPERTIES
    BUILD_RPATH ${ACADOS_LIB}
    INSTALL_RPATH ${ACADOS_LIB}
)

# C main: fprintf JSON dump (default dump file: esa_mpc_ocp_nlp_dump_c.json)
add_executable(main_esa_mpc_ocp_c main_esa_mpc_ocp.c)
target_link_libraries(main_esa_mpc_ocp_c acados_solver_esa_mpc_ocp acados hpipm blasfeo m)
set_target_properties(main_esa_mpc_ocp_c PROPERTIES
    BUILD_RPATH ${ACADOS_LIB}
    INSTALL_RPATH ${ACADOS_LIB}
)

# C++ main: DumpNlpToJsonFile from esa_mpc_ocp_wrapper.cpp (default: esa_mpc_ocp_nlp_dump.json)
add_executable(main_esa_mpc_ocp main_esa_mpc_ocp.cpp)
target_link_libraries(main_esa_mpc_ocp acados_solver_esa_mpc_ocp acados hpipm blasfeo m stdc++)
set_target_properties(main_esa_mpc_ocp PROPERTIES
    BUILD_RPATH ${ACADOS_LIB}
    INSTALL_RPATH ${ACADOS_LIB}
)
