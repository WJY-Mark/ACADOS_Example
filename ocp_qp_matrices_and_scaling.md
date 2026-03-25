# acados OCP-QP 矩阵字段说明与 QP Scaling

本文汇总 Python 接口 `AcadosOcpSolver.get_from_qp_in(stage, field)` 中常见字段的含义，并说明 `get_qp_scaling_constraints` / `get_qp_scaling_objective` 中的 **QP scaling**。

底层数据格式与 HPIPM 的 `d_ocp_qp` 一致；C 层可通过 `ocp_nlp_get_at_stage` / `ocp_nlp_qp_dims_get_from_attr` 读取相同字段（见 `interfaces/acados_c/ocp_nlp_interface.h`）。

---

## 1. HPIPM 阶段 QP 的标准形式（便于对照符号）

每个阶段 \(k\) 的代价（控制在前、状态在后）：

\[
\frac{1}{2} \begin{pmatrix} u_k \\ x_k \end{pmatrix}^T\begin{pmatrix} R_k & S_k \\ S_k^T & Q_k \end{pmatrix}\begin{pmatrix} u_k \\ x_k \end{pmatrix}+ \begin{pmatrix} r_k \\ q_k \end{pmatrix}^T\begin{pmatrix} u_k \\ x_k \end{pmatrix}
\]

动力学（等式约束）：

\[
x_{k+1} = A_k x_k + B_k u_k + b_k
\]

一般线性不等式：

\[
\ell_k \le C_k x_k + D_k u_k \le u_k^{\mathrm{g}}
\quad\text{（在 acados 中对应 `lg`, `ug`）}
\]

状态/控制盒约束通过 `lbx`/`ubx`、`lbu`/`ubu` 与索引 `idxb` 等描述。

---

## 2. 原始 QP 数据字段（`get_from_qp_in`）

| 字段 | 含义 | 典型维度 | 适用阶段 |
|------|------|----------|----------|
| `Q` | 状态代价 Hessian \(Q_k\) | \(n_x \times n_x\) | \(0,\ldots,N\) |
| `R` | 控制代价 Hessian \(R_k\) | \(n_u \times n_u\) | \(0,\ldots,N-1\) |
| `S` | 交叉项 \(S_k\)（\(u\)-\(x\)） | \(n_u \times n_x\) | \(0,\ldots,N-1\) |
| `q` | 状态线性项梯度 | \(n_x \times 1\) | \(0,\ldots,N\) |
| `r` | 控制线性项梯度 | \(n_u \times 1\) | \(0,\ldots,N-1\) |
| `A` | 动力学中 \(\partial x_{k+1}/\partial x_k\) | \(n_{x,+}\times n_x\) | \(0,\ldots,N-1\) |
| `B` | 动力学中 \(\partial x_{k+1}/\partial u_k\) | \(n_{x,+}\times n_u\) | \(0,\ldots,N-1\) |
| `b` | 动力学仿射项（含线性化残差等） | \(n_{x,+}\times 1\) | \(0,\ldots,N-1\) |
| `C` | 一般约束对 \(x\) 的系数 | \(n_g \times n_x\) | \(0,\ldots,N\) |
| `D` | 一般约束对 \(u\) 的系数 | \(n_g \times n_u\) | \(0,\ldots,N-1\) |
| `lg` | 一般约束下界 | \(n_g \times 1\) | \(0,\ldots,N\) |
| `ug` | 一般约束上界 | \(n_g \times 1\) | \(0,\ldots,N\) |
| `lbx` | 状态盒约束下界（按 `idxb` 选中的分量） | \(n_{bx}\times 1\) | \(0,\ldots,N\) |
| `ubx` | 状态盒约束上界 | \(n_{bx}\times 1\) | \(0,\ldots,N\) |
| `lbu` | 控制盒约束下界 | \(n_{bu}\times 1\) | \(0,\ldots,N-1\) |
| `ubu` | 控制盒约束上界 | \(n_{bu}\times 1\) | \(0,\ldots,N-1\) |

**说明：**

- HPIPM 中 Hessian 分块顺序为 **\([u;x]\)**，即左上角为 `R`、右下角为 `Q`；`get_hessian_block(stage)` 会拼成 \(\begin{smallmatrix}R&S\\S^\top&Q\end{smallmatrix}\)。
- SQP 中 `A,B,b` 来自当前迭代点动力学线性化；`Q,R,S,q,r` 来自代价的二次近似（如 Gauss-Newton）。
- 整数索引字段（如 `idxs`, `idxb`, `idxs_rev`）用于松弛与盒约束映射，含义见 HPIPM 文档。

---

## 3. Riccati 相关字段（`P`, `K`, `Lr`, `p`）

仅当 QP 求解器为 **`PARTIAL_CONDENSING_HPIPM`** 且 **`qp_solver_cond_N == N_horizon`** 时，Python 侧允许读取（与 `acados_ocp_solver.py` 校验一致）。这些量来自 **HPIPM 内点法工作区中的 Riccati 因子分解**，不是 QP 问题数据里的原始 `Q,R,S`。

| 字段 | 含义 | 典型维度 |
|------|------|----------|
| `P` | Riccati 价值矩阵：消去未来控制后，关于当前 \(x_k\) 的“代价-to-go”二次项系数（对称） | \(n_x \times n_x\) |
| `K` | 反馈增益：最优控制齐次部分 \(u_k \approx -K_k x_k + \cdots\) | \(n_u \times n_x\) |
| `Lr` | 对 \(u_k\) 的有效 Hessian 的 **下三角 Cholesky 因子**，满足 \(R_{\mathrm{ric},k} \approx L_r L_r^\top\)（\(R_{\mathrm{ric}}\) 由 Riccati 递推得到，**不等于**数据矩阵 `R`） | \(n_u \times n_u\) |
| `p` | Riccati 中与状态线性项相关的向量（仿射控制律的一部分） | \(n_x \times 1\) |

**注意：** 有不等式约束或 IPM 迭代未结束时，这些矩阵对应当前迭代下的内部因子；`qp_diagnostics(..., PROJECTED_HESSIAN)` 中会用 `Lr @ Lr.T` 与 `P`, `B` 构造投影 Hessian 块。

---

## 4. Partial condensing 后的字段（`pcond_*`）

当 QP 求解器为 **`PARTIAL_CONDENSING_*`** 时，HPIPM 先在内部构造 **阶段数更短**（\(N_2 =\) `qp_solver_cond_N`）的等价 OCP-QP。`pcond_*` 字段与该 **压缩后 QP** 的矩阵一一对应，含义与第 2 节相同，仅 **阶段索引与维度** 不同：

| 字段 | 对应原始字段 | 说明 |
|------|--------------|------|
| `pcond_Q`, `pcond_R`, `pcond_S`, `pcond_q`, `pcond_r` | `Q`, `R`, `S`, `q`, `r` | 块往往更大、更稠密（多步控制拼在一起） |
| `pcond_A`, `pcond_B`, `pcond_b` | `A`, `B`, `b` | 块内多步动力学凝聚后的等效线性模型 |
| `pcond_C`, `pcond_D`, `pcond_lg`, `pcond_ug` | `C`, `D`, `lg`, `ug` | 一般约束在 condensed 阶段的表示 |
| `pcond_lbx`, `pcond_ubx`, `pcond_lbu`, `pcond_ubu` | `lbx`, `ubx`, `lbu`, `ubu` | 边界约束在 condensed 阶段的表示 |

**阶段范围：** `stage` 满足 \(0 \le \mathrm{stage} \le N_2\)；动力学类字段在最后一阶段不可用（与 Python 校验一致）。

---

## 5. QP Scaling 是什么？（`get_qp_scaling_*`）

### 5.1 动机

数值上，若 Hessian 特征值很大、约束行尺度差异很大，内点法或主动集法在求解 QP 时容易出现：

- KKT 矩阵病态，迭代不稳定或收敛变慢；
- 停机准则（残差范数）与真实物理尺度脱节。

**QP scaling** 在送入 HPIPM 之前对目标与（部分）约束做 **等价变形**，使系数与界限的量级更接近 1，从而改善数值性。求解器在缩放空间里求解；解可通过逆变换回到原空间（对用户透明）。

### 5.2 acados 中的选项（`AcadosOcpOptions`）

- **`qpscaling_scale_objective`**（默认 `NO_OBJECTIVE_SCALING`）  
  - `OBJECTIVE_GERSHGORIN`：用 Gershgorin 估计 Hessian 最大绝对特征值上界 `max_abs_eig`，再设  
    \(\texttt{obj\_factor} = \min(1,\; \texttt{qpscaling\_ub\_max\_abs\_eig} / \texttt{max\_abs\_eig})\)。  
  - 整体效果：**缩小过大的二次代价尺度**，避免 Hessian 特征值过大。

- **`qpscaling_scale_constraints`**（默认 `NO_CONSTRAINT_SCALING`）  
  - `INF_NORM`：对每条一般约束（**不含**简单盒界）按系数与界的 \(\infty\)-范数缩放，使缩放后系数与界大致 \(\le 1\)；松弛惩罚会同步调整以保持 **数学上等价**。

顺序：**先缩放目标，再缩放约束**（见 `acados_ocp_options.py` 文档字符串）。

### 5.3 Python 读取缩放因子

- **`get_qp_scaling_constraints(stage)`**  
  - 返回长度 **`ng + nh + nphi`** 的向量（盒界不缩放）。  
  - 仅当 `qpscaling_scale_constraints != "NO_CONSTRAINT_SCALING"` 时可用。

- **`get_qp_scaling_objective()`**  
  - 返回标量 **目标缩放因子**（上一次 QP 求解对应的值）。  
  - 仅当 `qpscaling_scale_objective != "NO_OBJECTIVE_SCALING"` 时可用。

### 5.4 简单例子

设某阶段有一条一般约束 \(1000\, x_1 + 0.001\, u_1 \le 5000\)。左侧系数范数与右端都很大，行尺度极差。启用 **`INF_NORM`** 后，可能等价于在缩放变量下写成约 \(1\cdot \tilde x_1 + \cdots \le 1\) 的形式，HPIPM 在更好条件下求同一 QP。  
若 Hessian 某块特征值约 \(10^8\)，启用 **`OBJECTIVE_GERSHGORIN`** 且 `qpscaling_ub_max_abs_eig = 1e5` 时，`obj_factor` 可能为 \(10^5/10^8 = 10^{-3}\)，相当于在缩放空间里把二次项整体缩小，**最优解不变**，但数值更稳。

---

## 6. 相关代码位置

| 内容 | 路径 |
|------|------|
| Python `get_from_qp_in` / `get_hessian_block` | `interfaces/acados_template/acados_template/acados_ocp_solver.py` |
| Python scaling 读取 | 同上，`get_qp_scaling_constraints`, `get_qp_scaling_objective` |
| QP scaling 选项说明 | `interfaces/acados_template/acados_template/acados_ocp_options.py`（`qpscaling_*`） |
| C API | `interfaces/acados_c/ocp_nlp_interface.h`：`ocp_nlp_get_at_stage`, `ocp_nlp_qp_dims_get_from_attr` |
| HPIPM QP 结构 | `external/hpipm/include/hpipm_d_ocp_qp.h` |
| Riccati 量提取 | `external/hpipm/ocp_qp/x_ocp_qp_ipm.c`（`OCP_QP_IPM_GET_RIC_*`） |

---

## 7. 与 condensing 推导文档的关系

多阶段 QP 如何经 full/partial condensing 变换的数学推导见同目录下的 [condensing_explanation.md](./condensing_explanation.md)。
