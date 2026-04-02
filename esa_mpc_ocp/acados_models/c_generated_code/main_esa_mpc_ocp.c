/*
 * C 版 deterministic OCP main（与 main_esa_mpc_ocp.cpp 同题，JSON 用 fprintf 写出）。
 * C++ 版见 main_esa_mpc_ocp.cpp（DumpNlpToJsonFile 与车载 wrapper 一致）。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_esa_mpc_ocp.h"

#include "blasfeo_d_aux_ext_dep.h"

#define NX     ESA_MPC_OCP_NX
#define NP     ESA_MPC_OCP_NP
#define NU     ESA_MPC_OCP_NU
#define NBX0   ESA_MPC_OCP_NBX0
#define NY0    ESA_MPC_OCP_NY0
#define NY     ESA_MPC_OCP_NY
#define NYN    ESA_MPC_OCP_NYN
#define NG     ESA_MPC_OCP_NG
#define NGN    ESA_MPC_OCP_NGN
#define NBX    ESA_MPC_OCP_NBX
#define NBU    ESA_MPC_OCP_NBU
#define NBXN   ESA_MPC_OCP_NBXN
#define NS     ESA_MPC_OCP_NS
#define NSN    ESA_MPC_OCP_NSN

static void write_json_arr_double(FILE *f, const double *v, int n)
{
    fputc('[', f);
    for (int i = 0; i < n; ++i) {
        if (i)
            fprintf(f, ", ");
        fprintf(f, "%.15g", v[i]);
    }
    fputc(']', f);
}

static int dump_nlp_inputs_to_json(
    ocp_nlp_config *nlp_config,
    ocp_nlp_dims *nlp_dims,
    ocp_nlp_in *nlp_in,
    const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "dump_nlp_inputs_to_json: cannot open %s\n", path);
        return -1;
    }

    const int N = nlp_dims->N;

    fprintf(f, "{\n  \"N\": %d,\n", N);

    fprintf(f, "  \"time_steps\": [");
    for (int i = 0; i < N; ++i) {
        double Ts = 0.0;
        ocp_nlp_in_get(nlp_config, nlp_dims, nlp_in, i, "Ts", &Ts);
        if (i)
            fprintf(f, ", ");
        fprintf(f, "%.15g", Ts);
    }
    fprintf(f, "],\n");

    fprintf(f, "  \"stages\": [\n");
    for (int i = 0; i <= N; ++i) {
        const int nx = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, NULL, i, "nx");
        const int nu = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, NULL, i, "nu");
        const int ny = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, NULL, i, "ny");
        const int np = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, NULL, i, "np");
        const int nbx = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, NULL, i, "nbx");
        const int nbu = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, NULL, i, "nbu");
        const int ng = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, NULL, i, "ng");
        const int ns = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, NULL, i, "sl");

        fprintf(f, "    {");
        fprintf(f,
            "\"nx\":%d,\"nu\":%d,\"ny\":%d,\"np\":%d,\"nbx\":%d,\"nbu\":%d,\"ng\":%d,\"ns\":%d,\n",
            nx, nu, ny, np, nbx, nbu, ng, ns);

        fprintf(f, "     \"W\":");
        if (ny > 0) {
            double *W = (double *)calloc((size_t)(ny * ny), sizeof(double));
            ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "W", W);
            write_json_arr_double(f, W, ny * ny);
            free(W);
        } else {
            fprintf(f, "[]");
        }
        fprintf(f, ",\n");

        fprintf(f, "     \"yref\":");
        if (ny > 0) {
            double *yref = (double *)calloc((size_t)ny, sizeof(double));
            ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
            write_json_arr_double(f, yref, ny);
            free(yref);
        } else {
            fprintf(f, "[]");
        }
        fprintf(f, ",\n");

        fprintf(f, "     \"p\":");
        if (np > 0) {
            double *p = (double *)calloc((size_t)np, sizeof(double));
            ocp_nlp_in_get(nlp_config, nlp_dims, nlp_in, i, "p", p);
            write_json_arr_double(f, p, np);
            free(p);
        } else {
            fprintf(f, "[]");
        }
        fprintf(f, ",\n");

        fprintf(f, "     \"lbx\":");
        if (nbx > 0) {
            double *lbx = (double *)calloc((size_t)nbx, sizeof(double));
            double *ubx = (double *)calloc((size_t)nbx, sizeof(double));
            ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
            ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
            write_json_arr_double(f, lbx, nbx);
            fprintf(f, ",\"ubx\":");
            write_json_arr_double(f, ubx, nbx);
            free(lbx);
            free(ubx);
        } else {
            fprintf(f, "[],\"ubx\":[]");
        }
        fprintf(f, ",\n");

        fprintf(f, "     \"lbu\":");
        if (nbu > 0) {
            double *lbu = (double *)calloc((size_t)nbu, sizeof(double));
            double *ubu = (double *)calloc((size_t)nbu, sizeof(double));
            ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
            ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
            write_json_arr_double(f, lbu, nbu);
            fprintf(f, ",\"ubu\":");
            write_json_arr_double(f, ubu, nbu);
            free(lbu);
            free(ubu);
        } else {
            fprintf(f, "[],\"ubu\":[]");
        }
        fprintf(f, ",\n");

        fprintf(f, "     \"lg\":");
        if (ng > 0) {
            double *lg = (double *)calloc((size_t)ng, sizeof(double));
            double *ug = (double *)calloc((size_t)ng, sizeof(double));
            ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "lg", lg);
            ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "ug", ug);
            write_json_arr_double(f, lg, ng);
            fprintf(f, ",\"ug\":");
            write_json_arr_double(f, ug, ng);
            fprintf(f, ",\n");
            fprintf(f, "     \"C\":");
            if (nx > 0) {
                double *Cmat = (double *)calloc((size_t)(ng * nx), sizeof(double));
                ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "C", Cmat);
                write_json_arr_double(f, Cmat, ng * nx);
                free(Cmat);
            } else {
                fprintf(f, "[]");
            }
            fprintf(f, ",\"D\":");
            if (nu > 0 && ng > 0) {
                double *Dmat = (double *)calloc((size_t)(ng * nu), sizeof(double));
                ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "D", Dmat);
                write_json_arr_double(f, Dmat, ng * nu);
                free(Dmat);
            } else {
                fprintf(f, "[]");
            }
            free(lg);
            free(ug);
        } else {
            fprintf(f, "[],\"ug\":[],\n     \"C\":[],\"D\":[]");
        }
        fprintf(f, ",\n");

        fprintf(f, "     \"Zl\":");
        if (ns > 0) {
            double *Zl = (double *)calloc((size_t)ns, sizeof(double));
            double *Zu = (double *)calloc((size_t)ns, sizeof(double));
            double *zl = (double *)calloc((size_t)ns, sizeof(double));
            double *zu = (double *)calloc((size_t)ns, sizeof(double));
            ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
            ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
            ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
            ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
            write_json_arr_double(f, Zl, ns);
            fprintf(f, ",\"Zu\":");
            write_json_arr_double(f, Zu, ns);
            fprintf(f, ",\"zl\":");
            write_json_arr_double(f, zl, ns);
            fprintf(f, ",\"zu\":");
            write_json_arr_double(f, zu, ns);
            free(Zl);
            free(Zu);
            free(zl);
            free(zu);
        } else {
            fprintf(f, "[],\"Zu\":[],\"zl\":[],\"zu\":[]");
        }
        fprintf(f, "\n    }");
        if (i < N)
            fprintf(f, ",");
        fprintf(f, "\n");
    }

    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("[ESA_OCP_DEBUG] dumped NLP inputs (C) to %s\n", path);
    return 0;
}

int main(void)
{
    int N = ESA_MPC_OCP_N;
    int status;

    double time_steps[ESA_MPC_OCP_N];
    for (int i = 0; i < 20; i++) time_steps[i] = 0.02;
    for (int i = 20; i < 40; i++) time_steps[i] = 0.2;

    esa_mpc_ocp_solver_capsule *capsule = esa_mpc_ocp_acados_create_capsule();
    status = esa_mpc_ocp_acados_create_with_discretization(capsule, N, time_steps);
    if (status) {
        printf("acados_create failed with status %d\n", status);
        return 1;
    }

    ocp_nlp_config  *nlp_config  = esa_mpc_ocp_acados_get_nlp_config(capsule);
    ocp_nlp_dims    *nlp_dims    = esa_mpc_ocp_acados_get_nlp_dims(capsule);
    ocp_nlp_in      *nlp_in      = esa_mpc_ocp_acados_get_nlp_in(capsule);
    ocp_nlp_out     *nlp_out     = esa_mpc_ocp_acados_get_nlp_out(capsule);
    ocp_nlp_solver  *nlp_solver  = esa_mpc_ocp_acados_get_nlp_solver(capsule);

    double p[NP] = {33.39, 0.0, 149738.0, 0.0, 0.0, 2594.0, 1.588, 1.451, 2500.0};
    for (int i = 0; i <= N; i++)
        esa_mpc_ocp_acados_update_params(capsule, i, p, NP);

    double lbx0[NBX0] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double ubx0[NBX0] = {0.0, 0.0, 0.0, 0.0, 0.0};
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);

    double yref[NY] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    for (int i = 0; i < N; i++)
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    double yref_e[NYN] = {0.0, 0.0, 0.0, 3.0, 0.0};
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);

    double lbu_val = -10.0, ubu_val = 10.0;
    for (int i = 0; i < N; i++) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", &lbu_val);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", &ubu_val);
    }

    double lg_val = -10.0, ug_val = 10.0;
    for (int i = 0; i < N; i++) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lg", &lg_val);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ug", &ug_val);
    }
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lg", &lg_val);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "ug", &ug_val);

    double lat_lb[41], lat_ub[41];
    for (int k = 0; k <= 22; k++) { lat_lb[k] = 0.0; lat_ub[k] = 3.5; }
    for (int k = 23; k <= 28; k++) { lat_lb[k] = 2.5; lat_ub[k] = 3.5; }
    for (int k = 29; k <= 32; k++) { lat_lb[k] = 2.5; lat_ub[k] = 4.0; }
    for (int k = 33; k <= 40; k++) { lat_lb[k] = 2.5; lat_ub[k] = 5.0; }

    for (int i = 1; i < N; i++) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbx", &lat_lb[i]);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubx", &lat_ub[i]);
    }
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lbx", &lat_lb[N]);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "ubx", &lat_ub[N]);

    {
        double Zl = 100.0, Zu = 100.0, zl = 0.0, zu = 0.0;
        for (int i = 1; i < N; i++) {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", &Zl);
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", &Zu);
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", &zl);
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", &zu);
        }
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zl", &Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zu", &Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zl", &zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zu", &zu);
    }

    esa_mpc_ocp_acados_reset(capsule, 1);

    double x_init[NX] = {0};
    double u_init[NU] = {0};
    for (int i = 0; i < N; i++) {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x_init);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u_init);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x_init);

    {
        const char *dump_path = getenv("ESA_MPC_OCP_NLP_DUMP_JSON");
        if (dump_path == NULL || dump_path[0] == '\0')
            dump_path = "esa_mpc_ocp_nlp_dump_c.json";
        dump_nlp_inputs_to_json(nlp_config, nlp_dims, nlp_in, dump_path);
    }

    status = esa_mpc_ocp_acados_solve(capsule);

    double elapsed_time;
    int sqp_iter;
    double kkt_norm_inf;
    ocp_nlp_get(nlp_solver, "time_tot", &elapsed_time);
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);

    printf("status = %d  nlp_iter = %d  time_tot = %.3f ms  KKT = %.6e\n",
           status, sqp_iter, elapsed_time * 1000.0, kkt_norm_inf);

    double xtraj[NX * (N + 1)];
    double utraj[NU * N];
    for (int i = 0; i <= N; i++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "x", &xtraj[i * NX]);
    for (int i = 0; i < N; i++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "u", &utraj[i * NU]);

    printf("\n--- xtraj ---\n");
    for (int k = 0; k <= N; k++) {
        printf("  [%2d]", k);
        for (int j = 0; j < NX; j++)
            printf(" % .15e", xtraj[k * NX + j]);
        printf("\n");
    }
    printf("\n--- utraj ---\n");
    for (int k = 0; k < N; k++) {
        printf("  [%2d]", k);
        for (int j = 0; j < NU; j++)
            printf(" % .15e", utraj[k * NU + j]);
        printf("\n");
    }

    if (status == ACADOS_SUCCESS)
        printf("\nesa_mpc_ocp_acados_solve(): SUCCESS!\n");
    else
        printf("\nesa_mpc_ocp_acados_solve() failed with status %d.\n", status);

    esa_mpc_ocp_acados_print_stats(capsule);

    esa_mpc_ocp_acados_free(capsule);
    esa_mpc_ocp_acados_free_capsule(capsule);

    return status;
}
