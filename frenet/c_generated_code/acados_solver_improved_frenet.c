/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "improved_frenet_model/improved_frenet_model.h"





#include "acados_solver_improved_frenet.h"

#define NX     IMPROVED_FRENET_NX
#define NZ     IMPROVED_FRENET_NZ
#define NU     IMPROVED_FRENET_NU
#define NP     IMPROVED_FRENET_NP
#define NP_GLOBAL     IMPROVED_FRENET_NP_GLOBAL
#define NY0    IMPROVED_FRENET_NY0
#define NY     IMPROVED_FRENET_NY
#define NYN    IMPROVED_FRENET_NYN

#define NBX    IMPROVED_FRENET_NBX
#define NBX0   IMPROVED_FRENET_NBX0
#define NBU    IMPROVED_FRENET_NBU
#define NG     IMPROVED_FRENET_NG
#define NBXN   IMPROVED_FRENET_NBXN
#define NGN    IMPROVED_FRENET_NGN

#define NH     IMPROVED_FRENET_NH
#define NHN    IMPROVED_FRENET_NHN
#define NH0    IMPROVED_FRENET_NH0
#define NPHI   IMPROVED_FRENET_NPHI
#define NPHIN  IMPROVED_FRENET_NPHIN
#define NPHI0  IMPROVED_FRENET_NPHI0
#define NR     IMPROVED_FRENET_NR

#define NS     IMPROVED_FRENET_NS
#define NS0    IMPROVED_FRENET_NS0
#define NSN    IMPROVED_FRENET_NSN

#define NSBX   IMPROVED_FRENET_NSBX
#define NSBU   IMPROVED_FRENET_NSBU
#define NSH0   IMPROVED_FRENET_NSH0
#define NSH    IMPROVED_FRENET_NSH
#define NSHN   IMPROVED_FRENET_NSHN
#define NSG    IMPROVED_FRENET_NSG
#define NSPHI0 IMPROVED_FRENET_NSPHI0
#define NSPHI  IMPROVED_FRENET_NSPHI
#define NSPHIN IMPROVED_FRENET_NSPHIN
#define NSGN   IMPROVED_FRENET_NSGN
#define NSBXN  IMPROVED_FRENET_NSBXN



// ** solver data **

improved_frenet_solver_capsule * improved_frenet_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(improved_frenet_solver_capsule));
    improved_frenet_solver_capsule *capsule = (improved_frenet_solver_capsule *) capsule_mem;

    return capsule;
}


int improved_frenet_acados_free_capsule(improved_frenet_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int improved_frenet_acados_create(improved_frenet_solver_capsule* capsule)
{
    int N_shooting_intervals = IMPROVED_FRENET_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return improved_frenet_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int improved_frenet_acados_update_time_steps(improved_frenet_solver_capsule* capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "improved_frenet_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;
}

/**
 * Internal function for improved_frenet_acados_create: step 1
 */
void improved_frenet_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    nlp_solver_plan->nlp_cost[0] = LINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = ERK;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* improved_frenet_acados_create_setup_dimensions(improved_frenet_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    nbxe[0] = 3;
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);

    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for improved_frenet_acados_create: step 3
 */
void improved_frenet_acados_create_setup_functions(improved_frenet_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;




    // explicit ode
    capsule->expl_vde_forw = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(expl_vde_forw[i], improved_frenet_expl_vde_forw);
    }

    capsule->expl_ode_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(expl_ode_fun[i], improved_frenet_expl_ode_fun);
    }

    capsule->expl_vde_adj = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(expl_vde_adj[i], improved_frenet_expl_vde_adj);
    }



#undef MAP_CASADI_FNC
}


/**
 * Internal function for improved_frenet_acados_create: step 4
 */
void improved_frenet_acados_create_set_default_parameters(improved_frenet_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        improved_frenet_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for improved_frenet_acados_create: step 5
 */
void improved_frenet_acados_setup_nlp_in(improved_frenet_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
//    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
//    capsule->nlp_in = nlp_in;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        improved_frenet_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    double time_step = 0.5;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.5;
        cost_scaling[1] = 0.5;
        cost_scaling[2] = 0.5;
        cost_scaling[3] = 0.5;
        cost_scaling[4] = 0.5;
        cost_scaling[5] = 0.5;
        cost_scaling[6] = 0.5;
        cost_scaling[7] = 0.5;
        cost_scaling[8] = 0.5;
        cost_scaling[9] = 0.5;
        cost_scaling[10] = 0.5;
        cost_scaling[11] = 0.5;
        cost_scaling[12] = 0.5;
        cost_scaling[13] = 0.5;
        cost_scaling[14] = 0.5;
        cost_scaling[15] = 0.5;
        cost_scaling[16] = 0.5;
        cost_scaling[17] = 0.5;
        cost_scaling[18] = 0.5;
        cost_scaling[19] = 0.5;
        cost_scaling[20] = 0.5;
        cost_scaling[21] = 0.5;
        cost_scaling[22] = 0.5;
        cost_scaling[23] = 0.5;
        cost_scaling[24] = 0.5;
        cost_scaling[25] = 0.5;
        cost_scaling[26] = 0.5;
        cost_scaling[27] = 0.5;
        cost_scaling[28] = 0.5;
        cost_scaling[29] = 0.5;
        cost_scaling[30] = 0.5;
        cost_scaling[31] = 0.5;
        cost_scaling[32] = 0.5;
        cost_scaling[33] = 0.5;
        cost_scaling[34] = 0.5;
        cost_scaling[35] = 0.5;
        cost_scaling[36] = 0.5;
        cost_scaling[37] = 0.5;
        cost_scaling[38] = 0.5;
        cost_scaling[39] = 0.5;
        cost_scaling[40] = 0.5;
        cost_scaling[41] = 0.5;
        cost_scaling[42] = 0.5;
        cost_scaling[43] = 0.5;
        cost_scaling[44] = 0.5;
        cost_scaling[45] = 0.5;
        cost_scaling[46] = 0.5;
        cost_scaling[47] = 0.5;
        cost_scaling[48] = 0.5;
        cost_scaling[49] = 0.5;
        cost_scaling[50] = 0.5;
        cost_scaling[51] = 0.5;
        cost_scaling[52] = 0.5;
        cost_scaling[53] = 0.5;
        cost_scaling[54] = 0.5;
        cost_scaling[55] = 0.5;
        cost_scaling[56] = 0.5;
        cost_scaling[57] = 0.5;
        cost_scaling[58] = 0.5;
        cost_scaling[59] = 0.5;
        cost_scaling[60] = 0.5;
        cost_scaling[61] = 0.5;
        cost_scaling[62] = 0.5;
        cost_scaling[63] = 0.5;
        cost_scaling[64] = 0.5;
        cost_scaling[65] = 0.5;
        cost_scaling[66] = 0.5;
        cost_scaling[67] = 0.5;
        cost_scaling[68] = 0.5;
        cost_scaling[69] = 0.5;
        cost_scaling[70] = 0.5;
        cost_scaling[71] = 0.5;
        cost_scaling[72] = 0.5;
        cost_scaling[73] = 0.5;
        cost_scaling[74] = 0.5;
        cost_scaling[75] = 0.5;
        cost_scaling[76] = 0.5;
        cost_scaling[77] = 0.5;
        cost_scaling[78] = 0.5;
        cost_scaling[79] = 0.5;
        cost_scaling[80] = 0.5;
        cost_scaling[81] = 0.5;
        cost_scaling[82] = 0.5;
        cost_scaling[83] = 0.5;
        cost_scaling[84] = 0.5;
        cost_scaling[85] = 0.5;
        cost_scaling[86] = 0.5;
        cost_scaling[87] = 0.5;
        cost_scaling[88] = 0.5;
        cost_scaling[89] = 0.5;
        cost_scaling[90] = 0.5;
        cost_scaling[91] = 0.5;
        cost_scaling[92] = 0.5;
        cost_scaling[93] = 0.5;
        cost_scaling[94] = 0.5;
        cost_scaling[95] = 0.5;
        cost_scaling[96] = 0.5;
        cost_scaling[97] = 0.5;
        cost_scaling[98] = 0.5;
        cost_scaling[99] = 0.5;
        cost_scaling[100] = 0.5;
        cost_scaling[101] = 0.5;
        cost_scaling[102] = 0.5;
        cost_scaling[103] = 0.5;
        cost_scaling[104] = 0.5;
        cost_scaling[105] = 0.5;
        cost_scaling[106] = 0.5;
        cost_scaling[107] = 0.5;
        cost_scaling[108] = 0.5;
        cost_scaling[109] = 0.5;
        cost_scaling[110] = 0.5;
        cost_scaling[111] = 0.5;
        cost_scaling[112] = 0.5;
        cost_scaling[113] = 0.5;
        cost_scaling[114] = 0.5;
        cost_scaling[115] = 0.5;
        cost_scaling[116] = 0.5;
        cost_scaling[117] = 0.5;
        cost_scaling[118] = 0.5;
        cost_scaling[119] = 0.5;
        cost_scaling[120] = 0.5;
        cost_scaling[121] = 0.5;
        cost_scaling[122] = 0.5;
        cost_scaling[123] = 0.5;
        cost_scaling[124] = 0.5;
        cost_scaling[125] = 0.5;
        cost_scaling[126] = 0.5;
        cost_scaling[127] = 0.5;
        cost_scaling[128] = 0.5;
        cost_scaling[129] = 0.5;
        cost_scaling[130] = 0.5;
        cost_scaling[131] = 0.5;
        cost_scaling[132] = 0.5;
        cost_scaling[133] = 0.5;
        cost_scaling[134] = 0.5;
        cost_scaling[135] = 0.5;
        cost_scaling[136] = 0.5;
        cost_scaling[137] = 0.5;
        cost_scaling[138] = 0.5;
        cost_scaling[139] = 0.5;
        cost_scaling[140] = 0.5;
        cost_scaling[141] = 0.5;
        cost_scaling[142] = 0.5;
        cost_scaling[143] = 0.5;
        cost_scaling[144] = 0.5;
        cost_scaling[145] = 0.5;
        cost_scaling[146] = 0.5;
        cost_scaling[147] = 0.5;
        cost_scaling[148] = 0.5;
        cost_scaling[149] = 0.5;
        cost_scaling[150] = 0.5;
        cost_scaling[151] = 0.5;
        cost_scaling[152] = 0.5;
        cost_scaling[153] = 0.5;
        cost_scaling[154] = 0.5;
        cost_scaling[155] = 0.5;
        cost_scaling[156] = 0.5;
        cost_scaling[157] = 0.5;
        cost_scaling[158] = 0.5;
        cost_scaling[159] = 0.5;
        cost_scaling[160] = 0.5;
        cost_scaling[161] = 0.5;
        cost_scaling[162] = 0.5;
        cost_scaling[163] = 0.5;
        cost_scaling[164] = 0.5;
        cost_scaling[165] = 0.5;
        cost_scaling[166] = 0.5;
        cost_scaling[167] = 0.5;
        cost_scaling[168] = 0.5;
        cost_scaling[169] = 0.5;
        cost_scaling[170] = 0.5;
        cost_scaling[171] = 0.5;
        cost_scaling[172] = 0.5;
        cost_scaling[173] = 0.5;
        cost_scaling[174] = 0.5;
        cost_scaling[175] = 0.5;
        cost_scaling[176] = 0.5;
        cost_scaling[177] = 0.5;
        cost_scaling[178] = 0.5;
        cost_scaling[179] = 0.5;
        cost_scaling[180] = 0.5;
        cost_scaling[181] = 0.5;
        cost_scaling[182] = 0.5;
        cost_scaling[183] = 0.5;
        cost_scaling[184] = 0.5;
        cost_scaling[185] = 0.5;
        cost_scaling[186] = 0.5;
        cost_scaling[187] = 0.5;
        cost_scaling[188] = 0.5;
        cost_scaling[189] = 0.5;
        cost_scaling[190] = 0.5;
        cost_scaling[191] = 0.5;
        cost_scaling[192] = 0.5;
        cost_scaling[193] = 0.5;
        cost_scaling[194] = 0.5;
        cost_scaling[195] = 0.5;
        cost_scaling[196] = 0.5;
        cost_scaling[197] = 0.5;
        cost_scaling[198] = 0.5;
        cost_scaling[199] = 0.5;
        cost_scaling[200] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }


    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_forw", &capsule->expl_vde_forw[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_ode_fun", &capsule->expl_ode_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_adj", &capsule->expl_vde_adj[i]);
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 10;
    W_0[2+(NY0) * 2] = 1000;
    W_0[3+(NY0) * 3] = 100;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* Vx_0 = calloc(NY0*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_0[0+(NY0) * 0] = 1;
    Vx_0[1+(NY0) * 1] = 1;
    Vx_0[2+(NY0) * 2] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);
    double* Vu_0 = calloc(NY0*NU, sizeof(double));
    // change only the non-zero elements:
    Vu_0[3+(NY0) * 0] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 10;
    W[2+(NY) * 2] = 1000;
    W[3+(NY) * 3] = 100;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* Vx = calloc(NY*NX, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(NY) * 0] = 1;
    Vx[1+(NY) * 1] = 1;
    Vx[2+(NY) * 2] = 1;
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    double* Vu = calloc(NY*NU, sizeof(double));
    // change only the non-zero elements:
    Vu[3+(NY) * 0] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);



    // slacks initial
    double* zlu0_mem = calloc(4*NS0, sizeof(double));
    double* Zl_0 = zlu0_mem+NS0*0;
    double* Zu_0 = zlu0_mem+NS0*1;
    double* zl_0 = zlu0_mem+NS0*2;
    double* zu_0 = zlu0_mem+NS0*3;

    // change only the non-zero elements:
    zl_0[0] = 100;
    zu_0[0] = 100;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Zl", Zl_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Zu", Zu_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "zl", zl_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "zu", zu_0);
    free(zlu0_mem);
    // slacks
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    zl[0] = 100;
    zu[0] = 100;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);



    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(3 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);








    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -0.5;
    ubu[0] = 0.5;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);


    // set up soft bounds for general linear constraints
    int* idxsg = malloc(NSG * sizeof(int));
    idxsg[0] = 0;
    double* lusg = calloc(2*NSG, sizeof(double));
    double* lsg = lusg;
    double* usg = lusg + NSG;
    lsg[0] = 1;
    usg[0] = 1;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsg", idxsg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsg", lsg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usg", usg);
    }
    free(idxsg);
    free(lusg);






    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 0;
    idxbx[1] = 1;
    idxbx[2] = 2;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    lbx[0] = -10;
    ubx[0] = 10;
    lbx[1] = -1.5;
    ubx[1] = 1.5;
    lbx[2] = -0.5;
    ubx[2] = 0.5;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);


    // set up general constraints for stage 0 to N-1
    double* D = calloc(NG*NU, sizeof(double));
    double* C = calloc(NG*NX, sizeof(double));
    double* lug = calloc(2*NG, sizeof(double));
    double* lg = lug;
    double* ug = lug + NG;
    D[3+NG * 0] = 1;
    C[0+NG * 0] = 1;
    C[1+NG * 1] = 1;
    C[2+NG * 2] = 1;
    lg[0] = -10;
    lg[1] = -1.5;
    lg[2] = -0.5;
    lg[3] = -0.5;
    ug[0] = 10;
    ug[1] = 1.5;
    ug[2] = 0.5;
    ug[3] = 0.5;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "D", D);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "C", C);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lg", lg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ug", ug);
    }
    free(D);
    free(C);
    free(lug);





    /* terminal constraints */













}


static void improved_frenet_acados_create_set_opts(improved_frenet_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    double newton_tol_val = 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_tol", &newton_tol_val);

    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 200;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for improved_frenet_acados_create: step 7
 */
void improved_frenet_acados_set_nlp_out(improved_frenet_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for improved_frenet_acados_create: step 9
 */
int improved_frenet_acados_create_precompute(improved_frenet_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int improved_frenet_acados_create_with_discretization(improved_frenet_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != IMPROVED_FRENET_N && !new_time_steps) {
        fprintf(stderr, "improved_frenet_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, IMPROVED_FRENET_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    improved_frenet_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = improved_frenet_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    improved_frenet_acados_create_set_opts(capsule);

    // 4) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 5) setup functions, nlp_in and default parameters
    improved_frenet_acados_create_setup_functions(capsule);
    improved_frenet_acados_setup_nlp_in(capsule, N, new_time_steps);
    improved_frenet_acados_create_set_default_parameters(capsule);

    // 6) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // 7) create and set nlp_out
    // 7.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 7.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    improved_frenet_acados_set_nlp_out(capsule);

    // 8) do precomputations
    int status = improved_frenet_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int improved_frenet_acados_update_qp_solver_cond_N(improved_frenet_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from improved_frenet_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = improved_frenet_acados_create_precompute(capsule);
    return status;
}


int improved_frenet_acados_reset(improved_frenet_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int improved_frenet_acados_update_params(improved_frenet_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 2;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int improved_frenet_acados_update_params_sparse(improved_frenet_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int improved_frenet_acados_set_p_global_and_precompute_dependencies(improved_frenet_solver_capsule* capsule, double* data, int data_len)
{

    printf("No global_data, improved_frenet_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int improved_frenet_acados_solve(improved_frenet_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int improved_frenet_acados_setup_qp_matrices_and_factorize(improved_frenet_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



void improved_frenet_acados_batch_solve(improved_frenet_solver_capsule ** capsules, int * status_out, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_solve(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}


void improved_frenet_acados_batch_setup_qp_matrices_and_factorize(improved_frenet_solver_capsule ** capsules, int * status_out, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_setup_qp_matrices_and_factorize(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}


void improved_frenet_acados_batch_eval_params_jac(improved_frenet_solver_capsule ** capsules, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_params_jac(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}



void improved_frenet_acados_batch_eval_solution_sens_adj_p(improved_frenet_solver_capsule ** capsules, const char *field, int stage, double *out, int offset, int N_batch)
{


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_solution_sens_adj_p(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->sens_out, field, stage, out + i*offset);
    }


    return;
}


void improved_frenet_acados_batch_set_flat(improved_frenet_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_set_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_set_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }


    return;
}



void improved_frenet_acados_batch_get_flat(improved_frenet_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_get_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_get_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }


    return;
}


int improved_frenet_acados_free(improved_frenet_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->expl_vde_forw[i]);
        external_function_external_param_casadi_free(&capsule->expl_ode_fun[i]);
        external_function_external_param_casadi_free(&capsule->expl_vde_adj[i]);
    }
    free(capsule->expl_vde_adj);
    free(capsule->expl_vde_forw);
    free(capsule->expl_ode_fun);

    // cost

    // constraints



    return 0;
}


void improved_frenet_acados_print_stats(improved_frenet_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    double stat[1200];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int improved_frenet_acados_custom_update(improved_frenet_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *improved_frenet_acados_get_nlp_in(improved_frenet_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *improved_frenet_acados_get_nlp_out(improved_frenet_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *improved_frenet_acados_get_sens_out(improved_frenet_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *improved_frenet_acados_get_nlp_solver(improved_frenet_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *improved_frenet_acados_get_nlp_config(improved_frenet_solver_capsule* capsule) { return capsule->nlp_config; }
void *improved_frenet_acados_get_nlp_opts(improved_frenet_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *improved_frenet_acados_get_nlp_dims(improved_frenet_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *improved_frenet_acados_get_nlp_plan(improved_frenet_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
