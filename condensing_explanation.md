# Condensing 详解：从稀疏 OCP-QP 到稠密 QP

## 1. 问题设定

取 $N=3$ 为例（3 个控制阶段），状态 $x \in \mathbb{R}^{n_x}$，控制 $u \in \mathbb{R}^{n_u}$，$x_0$ 已知。

$$
\min_{x_1,x_2,x_3,\, u_0,u_1,u_2} \;\; \sum_{k=0}^{2}\left(x_k^T Q\, x_k + u_k^T R\, u_k\right) + x_3^T Q\, x_3
$$

$$
\text{s.t.} \quad x_{k+1} = A\,x_k + B\,u_k, \quad k=0,1,2
$$

---

## 2. Condensing 前：稀疏结构化 QP

将所有变量堆叠为一个大向量：

$$
z = \begin{bmatrix} x_0 \\ u_0 \\ x_1 \\ u_1 \\ x_2 \\ u_2 \\ x_3 \end{bmatrix}
$$

此时 QP 的结构为：

$$
\min_z \;\frac{1}{2}\, z^T \underbrace{\begin{bmatrix} Q & & & & & & \\ & R & & & & & \\ & & Q & & & & \\ & & & R & & & \\ & & & & Q & & \\ & & & & & R & \\ & & & & & & Q \end{bmatrix}}_{\mathcal{H}_{\text{sparse}}} z
$$

$$
\text{s.t.} \quad \underbrace{\begin{bmatrix} A & B & -I & & & & \\ & & A & B & -I & & \\ & & & & A & B & -I \end{bmatrix}}_{\mathcal{A}_{\text{eq}}} z = 0
$$

**特点**：

- Hessian $\mathcal{H}_{\text{sparse}}$ 是 **块对角** 的，非常稀疏
- 等式约束矩阵 $\mathcal{A}_{\text{eq}}$ 是 **带状** 的
- 决策变量维度：$4n_x + 3n_u$（含 $x_0$）
- 等式约束数量：$3n_x$

---

## 3. Condensing 过程：递推消元

利用动力学方程，把所有 $x_k$ 用 $x_0$（已知常量）和 $u_0, u_1, u_2$ 表达：

$$
\begin{aligned}
x_1 &= Ax_0 + Bu_0 \\
x_2 &= Ax_1 + Bu_1 = A^2 x_0 + AB\,u_0 + B\,u_1 \\
x_3 &= Ax_2 + Bu_2 = A^3 x_0 + A^2B\,u_0 + AB\,u_1 + B\,u_2
\end{aligned}
$$

写成矩阵形式：

$$
\underbrace{\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}}_{\mathbf{x}} = \underbrace{\begin{bmatrix} A \\ A^2 \\ A^3 \end{bmatrix}}_{\bar{A}} x_0 + \underbrace{\begin{bmatrix} B & 0 & 0 \\ AB & B & 0 \\ A^2B & AB & B \end{bmatrix}}_{\bar{B}} \underbrace{\begin{bmatrix} u_0 \\ u_1 \\ u_2 \end{bmatrix}}_{\mathbf{u}}
$$

**关键**：$\bar{B}$ 是一个 **下三角稠密矩阵**！这就是 condensing 名字的来源——稀疏结构被"凝聚"成了稠密结构。

---

## 4. Condensing 后：稠密 QP

将 $\mathbf{x} = \bar{A}x_0 + \bar{B}\mathbf{u}$ 代入目标函数（$x_0$ 已知，其贡献为常数项，忽略）：

$$
J = \mathbf{x}^T \bar{Q}\, \mathbf{x} + \mathbf{u}^T \bar{R}\, \mathbf{u}
$$

其中 $\bar{Q} = \text{blkdiag}(Q, Q, Q)$，$\bar{R} = \text{blkdiag}(R, R, R)$。

展开：

$$
J = (\bar{A}x_0 + \bar{B}\mathbf{u})^T \bar{Q}\, (\bar{A}x_0 + \bar{B}\mathbf{u}) + \mathbf{u}^T \bar{R}\, \mathbf{u}
$$

$$
= \underbrace{x_0^T \bar{A}^T \bar{Q}\, \bar{A}\, x_0}_{\text{常数}} + 2\,x_0^T \bar{A}^T \bar{Q}\, \bar{B}\, \mathbf{u} + \mathbf{u}^T \bar{B}^T \bar{Q}\, \bar{B}\, \mathbf{u} + \mathbf{u}^T \bar{R}\, \mathbf{u}
$$

因此 condensing 后的 QP 为：

$$
\boxed{\min_{\mathbf{u}} \;\frac{1}{2}\, \mathbf{u}^T H\, \mathbf{u} + g^T \mathbf{u}}
$$

其中：

$$
\boxed{H = \bar{B}^T \bar{Q}\, \bar{B} + \bar{R}} \quad \in \mathbb{R}^{Nn_u \times Nn_u}
$$

$$
\boxed{g = \bar{B}^T \bar{Q}\, \bar{A}\, x_0} \quad \in \mathbb{R}^{Nn_u}
$$

---

## 5. 对比总结（N=3 为例）

|  | Condensing 前（稀疏 QP） | Condensing 后（稠密 QP） |
|--|--------------------------|--------------------------|
| **决策变量** | $z = (x_0,u_0,x_1,u_1,x_2,u_2,x_3)$ | $\mathbf{u} = (u_0, u_1, u_2)$ |
| **变量维度** | $4n_x + 3n_u$ | $3n_u$ |
| **等式约束** | 3 条动力学约束 ($3n_x$ 维) | **无** |
| **Hessian 结构** | 块对角，稀疏 | **全稠密** |
| **Hessian 大小** | $(4n_x+3n_u) \times (4n_x+3n_u)$ | $3n_u \times 3n_u$ |

---

## 6. Hessian 结构直观对比

**Condensing 前**（$\mathcal{H}$ 是块对角的，空白=零）：

```
┌─────────────────────────┐
│ Q │   │   │   │   │   │   │
│   │ R │   │   │   │   │   │
│   │   │ Q │   │   │   │   │
│   │   │   │ R │   │   │   │
│   │   │   │   │ Q │   │   │
│   │   │   │   │   │ R │   │
│   │   │   │   │   │   │ Q │
└─────────────────────────┘
  稀疏！大量零块，可以高效利用结构
```

**Condensing 后**（$H$ 是全稠密的）：

```
┌─────────────────┐
│ ██████████████│
│ ██████████████│
│ ██████████████│
└─────────────────┘
  稠密！维度小（3n_u × 3n_u），但每个元素都非零
```

$H$ 的具体结构（展开 $\bar{B}^T \bar{Q} \bar{B} + \bar{R}$）：

$$
H = \begin{bmatrix}
B^TQB + B^TA^TQAB + B^T(A^2)^TQA^2B + R & B^TA^TQB + B^T(A^2)^TQAB & B^T(A^2)^TQB \\
\text{sym} & B^TQB + B^TA^TQAB + R & B^TA^TQB \\
\text{sym} & \text{sym} & B^TQB + R
\end{bmatrix}
$$

可以看到 $H$ 的每个块都耦合了多个阶段的信息——原来分散在各阶段的稀疏信息，被"凝聚"进了一个稠密矩阵。

---

## 7. Partial Condensing 的直觉

如果原问题有 $N=6$ 个阶段，设 $N_2 = 2$，`block_size = [3, 3]`：

- **Block 1**：把阶段 0,1,2 凝聚为 1 个阶段（内部的 $x_1, x_2$ 被消去）
- **Block 2**：把阶段 3,4,5 凝聚为 1 个阶段（内部的 $x_4, x_5$ 被消去）

结果是一个 **$N_2=2$ 阶段的 OCP-QP**，每个阶段的控制维度变大（因为把多个阶段的 $u$ 合并了），但整体仍保持 OCP 的稀疏带状结构，只是阶段更少。

---

## 8. Condensing 的好处

| 好处 | 说明 |
|------|------|
| **减少决策变量** | 消除状态变量后，QP 的决策变量数从 $\sum(n_x + n_u)$ 降为 $\sum n_u$（full）或介于两者之间（partial） |
| **消除等式约束** | 动力学等式约束被完全消除（full）或部分消除（partial），简化了约束结构 |
| **适配不同求解器** | Full condensing 后可使用 Dense QP 求解器（如 QPOASES、DAQP）；Partial condensing 后可使用阶段数更少的结构化 QP 求解器（如 HPIPM） |
| **Hessian 复用** | 支持 LHS/RHS 分离，在 Hessian 不变的场景下避免重复凝聚 Hessian，加速求解 |

## 9. 哪些问题适合做 Condensing

### 适合 Full Condensing 的场景

- 控制输入维度 $n_u$ **远小于** 状态维度 $n_x$ 的问题（凝聚后稠密 QP 很小）
- 预测步长 $N$ **较短**（否则稠密 QP 会非常大，$O(N \cdot n_u)$ 维度）
- 需要使用 Dense QP 求解器（如 QPOASES）的场景
- 典型：线性 MPC，Gauss-Newton 类问题

### 适合 Partial Condensing 的场景

- $N$ 较大但想减少求解器的阶段数，在稀疏结构和稠密结构之间取得平衡
- $n_u$ 和 $n_x$ 都较大，完全凝聚会导致稠密矩阵过大
- 想要利用结构化求解器（如 HPIPM）的稀疏性优势，同时减少阶段数以降低求解时间
- 通过调节 $N_2$ 和 `block_size` 在求解速度和内存之间灵活权衡

### 不适合 Condensing 的场景

- 状态维度 $n_x$ 很小而 $N$ 很大的问题——直接用稀疏求解器可能更高效
- 约束结构非常稀疏且求解器本身能高效利用这种稀疏性时，condensing 反而引入了额外的稠密化开销

---

**一句话总结**：Condensing = 用动力学递推关系做变量替换，把"大而稀疏"的结构化 QP 变成"小而稠密"的 QP，用空间结构换取问题规模的缩减。
$$
[ \frac{1}{2} \begin{pmatrix} u_k \ x_k \end{pmatrix}^T \underbrace{\begin{pmatrix} R_k & S_k \ S_k^T & Q_k \end{pmatrix}}_{\text{Hessian block}} \begin{pmatrix} u_k \ x_k \end{pmatrix} ]
$$

$$
[ \begin{pmatrix} R & \mathbf{0} \ \mathbf{0} & Q \end{pmatrix} ]
$$

$$
[ \begin{pmatrix} R & S \ S^T & Q \end{pmatrix} \in \mathbb{R}^{(n_u+n_x) \times (n_u+n_x)} ]
$$

$$
[ \min_{x_0,u_0,\ldots,x_N} \sum_{k=0}^{N-1} \left[ \frac{1}{2} \begin{pmatrix} u_k \ x_k \end{pmatrix}^T \begin{pmatrix} R_k & S_k \ S_k^T & Q_k \end{pmatrix} \begin{pmatrix} u_k \ x_k \end{pmatrix} + \begin{pmatrix} r_k \ q_k \end{pmatrix}^T \begin{pmatrix} u_k \ x_k \end{pmatrix} \right] + \frac{1}{2} x_N^T Q_N x_N + q_N^T x_N ]
$$

$$
[ \text{s.t.} \quad x_{k+1} = A_k x_k + B_k u_k + b_k \quad (\text{动力学等式约束}) ]
$$
$$
[ \text{lbx}_k \le x_k \le \text{ubx}_k \quad (\text{状态界约束}) ]
$$
$$
[ \text{lbu}_k \le u_k \le \text{ubu}_k \quad (\text{控制界约束}) ]
$$
$$
[ \text{lg}_k \le C_k x_k + D_k u_k \le \text{ug}_k \quad (\text{一般线性不等式约束}) ]
$$