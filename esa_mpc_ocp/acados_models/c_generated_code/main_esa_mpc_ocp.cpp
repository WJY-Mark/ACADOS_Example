/*
 * Deterministic OCP solve — mirrors Python run_deterministic_solve() exactly.
 * NLP JSON dump: same logic as aeb/src/esa_handler/esa_mpc/esa_mpc_ocp_wrapper.cpp
 * (WriteJsonArr + DumpNlpToJsonFile), as a free function below.
 */

#include <cstdio>
#include <cstdlib>  // std::getenv
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

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

namespace {

// Copied from esa_mpc_ocp_wrapper.cpp (anonymous namespace)
void WriteJsonArr(std::ofstream &f, const double *v, int n) {
  f << '[';
  for (int i = 0; i < n; ++i) {
    if (i) {
      f << ", ";
    }
    f << v[i];
  }
  f << ']';
}

// Copied from EsaMpcOCPWrapper::DumpNlpToJsonFile (esa_mpc_ocp_wrapper.cpp)
void DumpNlpToJsonFile(ocp_nlp_config *nlp_config,
                       ocp_nlp_dims *nlp_dims,
                       ocp_nlp_in *nlp_in,
                       const std::string &path) {
  if (nlp_config == nullptr || nlp_dims == nullptr || nlp_in == nullptr) {
    printf("[ESA_OCP_DEBUG] DumpNlpToJsonFile: NLP not initialized\n");
    return;
  }
  std::ofstream f(path);
  if (!f.is_open()) {
    printf("[ESA_OCP_DEBUG] cannot open %s\n", path.c_str());
    return;
  }
  f << std::setprecision(15);

  const int N = nlp_dims->N;
  f << "{\n  \"N\": " << N << ",\n";

  f << "  \"time_steps\": [";
  for (int i = 0; i < N; ++i) {
    double Ts = 0.0;
    ocp_nlp_in_get(nlp_config, nlp_dims, nlp_in, i, "Ts", &Ts);
    if (i) {
      f << ", ";
    }
    f << Ts;
  }
  f << "],\n";

  f << "  \"stages\": [\n";
  for (int i = 0; i <= N; ++i) {
    const int nx =
        ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nullptr, i, "nx");
    const int nu =
        ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nullptr, i, "nu");
    const int ny =
        ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nullptr, i, "ny");
    const int np =
        ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nullptr, i, "np");
    const int nbx =
        ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nullptr, i, "nbx");
    const int nbu =
        ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nullptr, i, "nbu");
    const int ng =
        ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nullptr, i, "ng");
    const int ns =
        ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nullptr, i, "sl");

    f << "    {";
    f << "\"nx\":" << nx << ",\"nu\":" << nu << ",\"ny\":" << ny
      << ",\"np\":" << np << ",\"nbx\":" << nbx << ",\"nbu\":" << nbu
      << ",\"ng\":" << ng << ",\"ns\":" << ns << ",\n";

    // cost W (column-major ny*ny)
    std::vector<double> W(static_cast<size_t>(ny * ny));
    ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "W", W.data());
    f << "     \"W\":";
    WriteJsonArr(f, W.data(), ny * ny);
    f << ",\n";

    // yref
    std::vector<double> yref(static_cast<size_t>(ny));
    ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "yref",
                           yref.data());
    f << "     \"yref\":";
    WriteJsonArr(f, yref.data(), ny);
    f << ",\n";

    // parameters
    f << "     \"p\":";
    if (np > 0) {
      std::vector<double> p(static_cast<size_t>(np));
      ocp_nlp_in_get(nlp_config, nlp_dims, nlp_in, i, "p", p.data());
      WriteJsonArr(f, p.data(), np);
    } else {
      f << "[]";
    }
    f << ",\n";

    // state bounds
    f << "     \"lbx\":";
    if (nbx > 0) {
      std::vector<double> lbx(static_cast<size_t>(nbx));
      std::vector<double> ubx(static_cast<size_t>(nbx));
      ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "lbx",
                                    lbx.data());
      ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "ubx",
                                    ubx.data());
      WriteJsonArr(f, lbx.data(), nbx);
      f << ",\"ubx\":";
      WriteJsonArr(f, ubx.data(), nbx);
    } else {
      f << "[],\"ubx\":[]";
    }
    f << ",\n";

    // control bounds
    f << "     \"lbu\":";
    if (nbu > 0) {
      std::vector<double> lbu(static_cast<size_t>(nbu));
      std::vector<double> ubu(static_cast<size_t>(nbu));
      ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "lbu",
                                    lbu.data());
      ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "ubu",
                                    ubu.data());
      WriteJsonArr(f, lbu.data(), nbu);
      f << ",\"ubu\":";
      WriteJsonArr(f, ubu.data(), nbu);
    } else {
      f << "[],\"ubu\":[]";
    }
    f << ",\n";

    // general constraints
    f << "     \"lg\":";
    if (ng > 0) {
      std::vector<double> lg(static_cast<size_t>(ng));
      std::vector<double> ug(static_cast<size_t>(ng));
      ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "lg",
                                    lg.data());
      ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "ug",
                                    ug.data());
      WriteJsonArr(f, lg.data(), ng);
      f << ",\"ug\":";
      WriteJsonArr(f, ug.data(), ng);
      f << ",\n";
      // C matrix (column-major ng*nx from acados)
      f << "     \"C\":";
      if (nx > 0) {
        std::vector<double> C(static_cast<size_t>(ng * nx));
        ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "C",
                                      C.data());
        WriteJsonArr(f, C.data(), ng * nx);
      } else {
        f << "[]";
      }
      f << ",\"D\":";
      if (nu > 0 && ng > 0) {
        std::vector<double> D(static_cast<size_t>(ng * nu));
        ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, i, "D",
                                      D.data());
        WriteJsonArr(f, D.data(), ng * nu);
      } else {
        f << "[]";
      }
    } else {
      f << "[],\"ug\":[],\n     \"C\":[],\"D\":[]";
    }
    f << ",\n";

    // slack weights
    f << "     \"Zl\":";
    if (ns > 0) {
      std::vector<double> Zl(static_cast<size_t>(ns));
      std::vector<double> Zu(static_cast<size_t>(ns));
      std::vector<double> zl(static_cast<size_t>(ns));
      std::vector<double> zu(static_cast<size_t>(ns));
      ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "Zl",
                             Zl.data());
      ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "Zu",
                             Zu.data());
      ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "zl",
                             zl.data());
      ocp_nlp_cost_model_get(nlp_config, nlp_dims, nlp_in, i, "zu",
                             zu.data());
      WriteJsonArr(f, Zl.data(), ns);
      f << ",\"Zu\":";
      WriteJsonArr(f, Zu.data(), ns);
      f << ",\"zl\":";
      WriteJsonArr(f, zl.data(), ns);
      f << ",\"zu\":";
      WriteJsonArr(f, zu.data(), ns);
    } else {
      f << "[],\"Zu\":[],\"zl\":[],\"zu\":[]";
    }
    f << "\n    }";
    if (i < N) {
      f << ',';
    }
    f << '\n';
  }
  f << "  ]\n}\n";
  f.close();
  printf("[ESA_OCP_DEBUG] dumped NLP inputs to %s\n", path.c_str());
}

}  // namespace

int main()
{
    int N = ESA_MPC_OCP_N;  /* 40 */
    int status;

    /* ---- time_steps: 20×0.02 + 20×0.2 ---- */
    double time_steps[ESA_MPC_OCP_N];
    for (int i = 0; i < 20; i++) time_steps[i] = 0.02;
    for (int i = 20; i < 40; i++) time_steps[i] = 0.2;

    /* ---- create solver with explicit time_steps ---- */
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

    /* ---- model parameters (same for all stages) ---- */
    double p[NP] = {33.39, 0.0, 149738.0, 0.0, 0.0, 2594.0, 1.588, 1.451, 2500.0};
    for (int i = 0; i <= N; i++)
        esa_mpc_ocp_acados_update_params(capsule, i, p, NP);

    /* ---- x0 ---- */
    double lbx0[NBX0] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double ubx0[NBX0] = {0.0, 0.0, 0.0, 0.0, 0.0};
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);

    /* ---- yref ---- */
    double yref[NY] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    for (int i = 0; i < N; i++)
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    double yref_e[NYN] = {0.0, 0.0, 0.0, 3.0, 0.0};
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);

    /* ---- control box ---- */
    double lbu_val = -10.0, ubu_val = 10.0;
    for (int i = 0; i < N; i++) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", &lbu_val);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", &ubu_val);
    }

    /* ---- delta linear constraint ---- */
    double lg_val = -10.0, ug_val = 10.0;
    for (int i = 0; i < N; i++) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lg", &lg_val);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ug", &ug_val);
    }
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lg", &lg_val);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "ug", &ug_val);

    /* ---- time-varying lat_error bounds ---- */
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

    /* ---- slack costs ---- */
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

    /* ---- dump NLP (same as EsaMpcOCPWrapper::DumpNlpToJsonFile) ---- */
    {
        std::string dump_path = std::string("/tmp/test_esa_mpc_ocp_nlp_dump.json");
        DumpNlpToJsonFile(nlp_config, nlp_dims, nlp_in, dump_path);
    }

    /* ---- solve ---- */
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
