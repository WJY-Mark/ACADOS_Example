"""
ESA MPC 力缩放对 QP 数值条件影响的定量分析脚本。

对比有无缩放（Sf=Sd=1 vs Sf=Sd=1e4）时：
  1. 连续动力学矩阵 A_c 的奇异值
  2. 离散动力学矩阵 A_d 的条件数（近端 dt=0.02 / 远端 dt=0.2）
  3. 单阶段 KKT 矩阵的条件数
  4. 完整 N=40 阶段 KKT 矩阵的条件数
"""

import numpy as np
from scipy.linalg import expm

# ── 车辆参数（与 esa_mpc_ocp.py 一致）──
m = 2594.0        # kg
lf = 1.588        # m
lr = 1.451        # m
Iz = 2500.0       # kg·m²
vx = 33.39        # m/s
Cr = 149738.0     # N/rad

Sf = 1e4          # FYF_SCALE
Sd = 1e4          # DF_SCALE

nx, nu = 5, 1
N = 40

# 代价权重（物理值）
W_BETA = 1e-2
W_YAW_RATE = 15.0
W_HEADING_ERROR = 20.0
W_LATERAL_ERROR = 1.5
W_FYF_PHYSICAL = 1e-10
W_DFYF_PHYSICAL = 1e-8


def build_Ac(sf):
    """构造连续动力学矩阵 A_c，sf 为 Fyf 缩放因子。"""
    return np.array([
        [-Cr / (m * vx), Cr * lr / (m * vx**2) - 1, 0, 0, sf / (m * vx)],
        [Cr * lr / Iz, -Cr * lr**2 / (Iz * vx), 0, 0, sf * lf / Iz],
        [0, 1, 0, 0, 0],
        [vx, 0, vx, 0, 0],
        [0, 0, 0, 0, 0],
    ])


def build_Bu(sf, sd):
    """构造控制输入矩阵 Bu_c。"""
    return np.array([[0], [0], [0], [0], [sd / sf]])


def build_H_diag(sf, sd):
    """构造 Hessian 对角元素 [Q_diag, R_diag]，带缩放修正。"""
    w_fyf_n = W_FYF_PHYSICAL * sf**2
    w_dfyf_n = W_DFYF_PHYSICAL * sd**2
    stage = np.array([W_BETA, W_YAW_RATE, W_HEADING_ERROR,
                      W_LATERAL_ERROR, w_fyf_n, w_dfyf_n])
    terminal = stage[:nx]
    return stage, terminal


def print_matrix(label, M, fmt="{:12.4e}"):
    print(f"  {label}:")
    for r in range(M.shape[0]):
        row_str = ", ".join(fmt.format(M[r, c]) for c in range(M.shape[1]))
        print(f"    [{row_str}]")


def analyze_Ac(sf, label):
    """分析 A_c 的奇异值。"""
    Ac = build_Ac(sf)
    svs = np.linalg.svd(Ac, compute_uv=False)
    print(f"\n{'=' * 60}")
    print(f"A_c ({label}, Sf={sf:.0e})")
    print(f"{'=' * 60}")
    print_matrix("A_c", Ac)
    print(f"  奇异值: {svs}")
    nonzero = svs[svs > 1e-15]
    if len(nonzero) > 0:
        print(f"  σ_max = {nonzero[0]:.4e},  σ_min⁺ = {nonzero[-1]:.4e}")
        print(f"  cond(A_c, 非零) = {nonzero[0] / nonzero[-1]:.4e}")
    return Ac


def analyze_Ad(Ac, dt, label):
    """分析离散 A_d 的条件数。"""
    Ad = expm(Ac * dt)
    cond = np.linalg.cond(Ad)
    print(f"\n  A_d (dt={dt}, {label}):")
    print_matrix("A_d", Ad)
    print(f"    cond(A_d) = {cond:.4e}")
    print(f"    第5列最大元素 = {np.max(np.abs(Ad[:, 4])):.4e}")
    return Ad


def build_single_stage_kkt(Ad, Bu, H_stage_diag, H_e_diag):
    """构造单阶段 KKT 矩阵: z=[x_k, u_k, x_{k+1}]。"""
    nz = nx + nu + nx  # 11
    nc = nx             # 5
    dim = nz + nc       # 16
    H = np.diag(np.concatenate([H_stage_diag, H_e_diag]))
    G = np.hstack([Ad, Bu, -np.eye(nx)])
    KKT = np.zeros((dim, dim))
    KKT[:nz, :nz] = H
    KKT[nz:, :nz] = G
    KKT[:nz, nz:] = G.T
    svs = np.linalg.svd(KKT, compute_uv=False)
    cond = svs[0] / svs[svs > 1e-15][-1]
    return cond, svs


def build_full_kkt(Ac, Bu, H_stage_diag, H_e_diag, time_steps):
    """构造完整 N 阶段 KKT 矩阵并计算条件数。"""
    nz = (N + 1) * nx + N * nu
    nc = N * nx
    dim = nz + nc

    Ad_list = [expm(Ac * dt) for dt in time_steps]

    KKT = np.zeros((dim, dim))

    def x_idx(k):
        return k * (nx + nu)

    def u_idx(k):
        return k * (nx + nu) + nx

    for k in range(N):
        ix, iu = x_idx(k), u_idx(k)
        KKT[ix:ix + nx, ix:ix + nx] = np.diag(H_stage_diag[:nx])
        KKT[iu:iu + nu, iu:iu + nu] = np.diag(H_stage_diag[nx:])

    ix_N = x_idx(N)
    KKT[ix_N:ix_N + nx, ix_N:ix_N + nx] = np.diag(H_e_diag)

    for k in range(N):
        row = nz + k * nx
        ix_k, iu_k, ix_k1 = x_idx(k), u_idx(k), x_idx(k + 1)
        KKT[row:row + nx, ix_k:ix_k + nx] = Ad_list[k]
        KKT[ix_k:ix_k + nx, row:row + nx] = Ad_list[k].T
        KKT[row:row + nx, iu_k:iu_k + nu] = Bu
        KKT[iu_k:iu_k + nu, row:row + nx] = Bu.T
        KKT[row:row + nx, ix_k1:ix_k1 + nx] = -np.eye(nx)
        KKT[ix_k1:ix_k1 + nx, row:row + nx] = -np.eye(nx)

    svs = np.linalg.svd(KKT, compute_uv=False)
    cond = svs[0] / svs[svs > 1e-15][-1]
    return cond, svs, KKT.shape[0]


def main():
    time_steps = np.concatenate([np.full(20, 0.02), np.full(20, 0.2)])

    cases = [
        ("不缩放", 1.0, 1.0),
        ("缩放后", Sf, Sd),
    ]

    results = {}

    for label, sf, sd in cases:
        print(f"\n{'#' * 70}")
        print(f"#  {label} (Sf={sf:.0e}, Sd={sd:.0e})")
        print(f"{'#' * 70}")

        Ac = analyze_Ac(sf, label)
        Bu = build_Bu(sf, sd)
        H_stage, H_e = build_H_diag(sf, sd)

        print(f"\n  Hessian 对角 (stage): {H_stage}")
        print(f"  Hessian 对角 (terminal): {H_e}")
        print(f"  κ(H) ≥ {np.max(H_stage) / np.min(H_stage):.4e}")

        for dt in [0.02, 0.2]:
            Ad = analyze_Ad(Ac, dt, label)

            cond_1, svs_1 = build_single_stage_kkt(Ad, Bu, H_stage, H_e)
            print(f"\n    单阶段 KKT (dt={dt}):")
            print(f"      κ(KKT) = {cond_1:.4e}")
            print(f"      σ_max  = {svs_1[0]:.4e}")
            print(f"      σ_min⁺ = {svs_1[svs_1 > 1e-15][-1]:.4e}")

        cond_full, svs_full, dim = build_full_kkt(Ac, Bu, H_stage, H_e, time_steps)
        results[label] = {
            "cond": cond_full,
            "sv_max": svs_full[0],
            "sv_min_pos": svs_full[svs_full > 1e-15][-1],
        }

        print(f"\n  {'─' * 50}")
        print(f"  完整 N={N} 阶段 KKT (维度 {dim}×{dim}):")
        print(f"    κ(KKT) = {cond_full:.4e}")
        print(f"    σ_max  = {svs_full[0]:.4e}")
        print(f"    σ_min⁺ = {svs_full[svs_full > 1e-15][-1]:.4e}")
        print(f"    log₁₀(κ) = {np.log10(cond_full):.1f}")
        print(f"    有效精度损失: ~{np.log10(cond_full):.0f} / 16 位 (double)")

    # ── 对比汇总 ──
    print(f"\n{'=' * 70}")
    print("对比汇总")
    print(f"{'=' * 70}")
    print(f"{'指标':<25} {'不缩放':>18} {'缩放后':>18}")
    print(f"{'─' * 61}")

    r_ns, r_sc = results["不缩放"], results["缩放后"]
    print(f"{'σ_max(KKT)':<25} {r_ns['sv_max']:>18.4e} {r_sc['sv_max']:>18.4e}")
    print(f"{'σ_min⁺(KKT)':<25} {r_ns['sv_min_pos']:>18.4e} {r_sc['sv_min_pos']:>18.4e}")
    print(f"{'κ(KKT)':<25} {r_ns['cond']:>18.4e} {r_sc['cond']:>18.4e}")
    print(f"{'log₁₀(κ)':<25} {np.log10(r_ns['cond']):>18.1f} {np.log10(r_sc['cond']):>18.1f}")
    print(f"{'有效精度损失':<25} {'~' + str(int(np.log10(r_ns['cond']))) + ' / 16 位':>18} "
          f"{'~' + str(int(np.log10(r_sc['cond']))) + ' / 16 位':>18}")

    ratio = r_ns["cond"] / r_sc["cond"]
    print(f"\n  改善倍数: {ratio:.1f}x  ({np.log10(ratio):.1f} 个数量级)")


if __name__ == "__main__":
    main()
