# HPIPM 内点法 KKT 矩阵推导

## 1. OCP QP 问题定义

MPC 在每个控制周期求解如下 QP 子问题：

$$
\min_{x_0, u_0, \ldots, x_N} \;\sum_{k=0}^{N-1} \frac{1}{2}
\begin{bmatrix} x_k \\ u_k \end{bmatrix}^T
\begin{bmatrix} Q_k & S_k \\ S_k^T & R_k \end{bmatrix}
\begin{bmatrix} x_k \\ u_k \end{bmatrix} + \frac{1}{2}\, x_N^T Q_N\, x_N
$$

subject to:

- **动力学约束（等式）**：$x_{k+1} = A_k\, x_k + B_k\, u_k + b_k, \quad k = 0, \ldots, N-1$
- **状态约束（不等式）**：$\underline{x}_k \le x_k \le \bar{x}_k$
- **控制约束（不等式）**：$\underline{u}_k \le u_k \le \bar{u}_k$
- **初始条件**：$x_0 = \bar{x}_0$

在我们的 ESA MPC 模型中（LINEAR_LS 代价 + Gauss-Newton）：
- $S_k = 0$（无交叉项）
- $Q_k = \text{diag}(W_\beta,\; W_{\dot\psi},\; W_{\Delta\theta},\; W_{e_y},\; W_{F_{yf}})$
- $R_k = \text{diag}(W_{\dot{F}_{yf}})$
- $Q_N = Q_k$（终端代价与过程代价相同）

## 2. 内点法（IPM）处理不等式约束

HPIPM 使用 primal-dual 内点法。对每个不等式约束引入松弛变量 $s$ 和对偶乘子 $\mu$，将不等式转化为等式 + 非负约束：

$$
x_k - s_k^{lb} = \underline{x}_k, \quad s_k^{lb} \ge 0
$$
$$
\bar{x}_k - x_k - s_k^{ub} = 0, \quad s_k^{ub} \ge 0
$$

并在目标函数中加入对数障碍项：$-\tau \sum_i \ln(s_i)$，其中 $\tau > 0$ 是障碍参数。

## 3. KKT 条件

对修改后的拉格朗日函数求一阶最优性条件，得到 KKT 系统。

### 3.1 定义变量

**原始变量向量**：

$$
z = \begin{bmatrix} x_0 \\ u_0 \\ x_1 \\ u_1 \\ \vdots \\ x_{N-1} \\ u_{N-1} \\ x_N \end{bmatrix}
\in \mathbb{R}^{n_z}, \quad n_z = (N+1)\,n_x + N\,n_u
$$

**等式约束（动力学）对偶变量**：

$$
\lambda = \begin{bmatrix} \lambda_0 \\ \lambda_1 \\ \vdots \\ \lambda_{N-1} \end{bmatrix}
\in \mathbb{R}^{n_c}, \quad n_c = N \cdot n_x
$$

### 3.2 原始 KKT 条件（对原始变量求导）

**对 $x_k$ 求导**（$k = 1, \ldots, N-1$）：

$$
Q_k\, x_k + A_k^T \lambda_k - \lambda_{k-1} + \mu_k^{ub} - \mu_k^{lb} = 0
$$

**对 $u_k$ 求导**（$k = 0, \ldots, N-1$）：

$$
R_k\, u_k + B_k^T \lambda_k + \mu_{u,k}^{ub} - \mu_{u,k}^{lb} = 0
$$

**对 $x_N$ 求导**：

$$
Q_N\, x_N - \lambda_{N-1} + \mu_N^{ub} - \mu_N^{lb} = 0
$$

**可行性（对 $\lambda_k$ 求导）**：

$$
A_k\, x_k + B_k\, u_k - x_{k+1} + b_k = 0
$$

**互补松弛条件**：

$$
\mu_i^{lb} \cdot s_i^{lb} = \tau, \quad \mu_i^{ub} \cdot s_i^{ub} = \tau
$$

### 3.3 消去松弛变量和不等式乘子

通过互补松弛条件和松弛定义，消去 $s$ 和 $\mu$，不等式约束的贡献被浓缩为一个**正定对角矩阵** $\Sigma$：

$$
\Sigma_{ii} = \frac{\mu_i^{lb}}{s_i^{lb}} + \frac{\mu_i^{ub}}{s_i^{ub}}
$$

$\Sigma$ 加到 Hessian 的对角上。

## 4. HPIPM 每次 IPM 迭代求解的线性系统

消去后，每次 Newton 迭代求解：

$$
\boxed{
\begin{bmatrix} H + \Sigma & G^T \\ G & 0 \end{bmatrix}
\begin{bmatrix} \Delta z \\ \Delta \lambda \end{bmatrix}
= \begin{bmatrix} -r_z \\ -r_\lambda \end{bmatrix}
}
$$

这就是 **HPIPM 的 KKT 矩阵**。

### 4.1 各块定义

**$H$（代价 Hessian，块对角）**：

$$
H = \text{blkdiag}\!\big(\underbrace{Q_0,\, R_0,\, Q_1,\, R_1,\, \ldots,\, Q_{N-1},\, R_{N-1}}_{N \text{ pairs}},\; Q_N\big)
$$

维度：$n_z \times n_z$

**$\Sigma$（IPM 障碍对角项）**：

$$
\Sigma = \text{diag}\!\left(\frac{\mu_i^{lb}}{s_i^{lb}} + \frac{\mu_i^{ub}}{s_i^{ub}}\right)_{i=1}^{n_z}
$$

维度：$n_z \times n_z$（正定对角矩阵，随 IPM 迭代变化）

**$G$（动力学约束 Jacobian）**：

$$
G =
\begin{bmatrix}
A_0 & B_0 & -I &     &     &        &     &    \\
    &     & A_1 & B_1 & -I  &        &     &    \\
    &     &     &     & \ddots &     &     &    \\
    &     &     &     &     & A_{N-1} & B_{N-1} & -I
\end{bmatrix}
$$

维度：$n_c \times n_z$

### 4.2 展开示例（N=3, $n_x$=5, $n_u$=1）

$$
\text{KKT} =
\left[\begin{array}{cc|cc|cc|c|ccc}
Q_0{+}\Sigma_0 & 0 & & & & & & A_0^T & & \\
0 & R_0{+}\Sigma_0' & & & & & & B_0^T & & \\
\hline
& & Q_1{+}\Sigma_1 & 0 & & & & -I & A_1^T & \\
& & 0 & R_1{+}\Sigma_1' & & & & & B_1^T & \\
\hline
& & & & Q_2{+}\Sigma_2 & 0 & & & -I & A_2^T \\
& & & & 0 & R_2{+}\Sigma_2' & & & & B_2^T \\
\hline
& & & & & & Q_3{+}\Sigma_3 & & & -I \\
\hline
A_0 & B_0 & -I & & & & & & & \\
& & A_1 & B_1 & -I & & & & & \\
& & & & A_2 & B_2 & -I & & &
\end{array}\right]
$$

## 5. HPIPM 的求解方法：Riccati 递推

HPIPM **不直接分解**上面的大矩阵，而是利用 OCP 的级联结构，用等价的 **Riccati 递推**高效求解。

从第 $N$ 步反向递推 Riccati 矩阵 $P_k$：

$$
P_N = Q_N + \Sigma_{x,N}
$$

$$
P_k = \tilde{Q}_k + A_k^T P_{k+1} A_k - (A_k^T P_{k+1} B_k)\,
\underbrace{(\tilde{R}_k + B_k^T P_{k+1} B_k)^{-1}}_{\text{每步需要求逆}}
\,(B_k^T P_{k+1} A_k)
$$

其中：
- $\tilde{Q}_k = Q_k + \Sigma_{x,k}$
- $\tilde{R}_k = R_k + \Sigma_{u,k}$

### Riccati 递推的数值稳定性取决于

1. $\tilde{R}_k + B_k^T P_{k+1} B_k$ 的条件数（每步需求逆）
2. $P_k$ 在递推过程中的增长/衰减
3. $(A, B)$ 的可稳定性和 $(A, Q^{1/2})$ 的可检测性

## 6. 条件数与求解失败的关系

| $\text{cond}(\text{KKT})$ | 含义 | HPIPM 表现 |
|---|---|---|
| $< 10^6$ | 良好 | 正常求解 |
| $10^6 \sim 10^{10}$ | 一般 | 可能需要 LM 正则化 |
| $10^{10} \sim 10^{14}$ | 差 | 大概率出现 MINSTEP |
| $> 10^{14}$ | 病态 | 基本会失败 |

## 7. 导致 KKT 病态的常见原因

1. **代价权重包含零**：$W_{F_{yf}} = 0$ 使 $H$ 的某些对角元素为零，导致 $(A, Q^{1/2})$ 不可检测
2. **状态量级差异大**：未归一化时 $F_{yf} \sim O(10^4)$ vs 其他状态 $\sim O(10^{-1})$
3. **$A_d$ 有多个特征值恰好在 1**：积分器模态 + 零权重 → Riccati $P_k$ 发散
4. **$B_k$ 元素太小**：$\tilde{R}_k + B_k^T P_{k+1} B_k \approx \tilde{R}_k$，矩阵可能奇异

## 8. ESA MPC 模型中的具体数值

对于当前模型（$N=40$, $n_x=5$, $n_u=1$）：

| 量 | 值 |
|---|---|
| 原始变量 $\dim(z)$ | $(40+1) \times 5 + 40 \times 1 = 245$ |
| 对偶变量 $\dim(\lambda)$ | $40 \times 5 = 200$ |
| KKT 矩阵维度 | $445 \times 445$ |
