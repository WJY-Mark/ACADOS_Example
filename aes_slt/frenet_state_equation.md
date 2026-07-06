# Frenet 坐标系下非线性状态方程推导

## 1. 状态量与控制量定义

### 状态量 $\mathbf{x} = [l,\ \delta\theta,\ \kappa,\ s,\ v_x,\ a_x]^T$

| 符号 | 含义 | 典型范围 |
|------|------|----------|
| $l$ | 横向偏移（车辆到参考线的有符号距离） | $[-8, 8]$ m |
| $\delta\theta$ | 车辆航向与参考线切线的夹角 | $[-10°, 10°]$ |
| $\kappa$ | 车辆路径曲率 | — |
| $s$ | 沿参考线的弧长坐标 | — |
| $v_x$ | 车辆纵向速度 | — |
| $a_x$ | 车辆纵向加速度 | — |

### 控制量 $\mathbf{u} = [d\kappa,\ j_x]^T$

| 符号 | 含义 |
|------|------|
| $d\kappa$ | 曲率变化率 |
| $j_x$ | 纵向加加速度（jerk） |

### 参考线参数

| 符号 | 含义 |
|------|------|
| $\kappa_r(s)$ | 参考线在 $s$ 处的曲率（时变参数，从参考线插值获得） |

---

## 2. 几何关系推导

### 2.1 速度在 Frenet 坐标系中的分解

车辆以速度 $v_x$ 沿自身航向运动，将其分解到 Frenet 坐标系下：

- **沿参考线切向分量**：$v_x \cos(\delta\theta)$
- **沿参考线法向分量**：$v_x \sin(\delta\theta)$

### 2.2 弧长缩放因子

在参考线曲率为 $\kappa_r$ 处，距参考线横向偏移 $l$ 的位置，沿切向方向的实际弧长与参考线弧长之比为：

$$\frac{ds_{actual}}{ds_{ref}} = 1 - \kappa_r(s) \cdot l$$

物理含义：在弯道内侧（$l > 0$, $\kappa_r > 0$），偏移位置走的弧长比参考线短；外侧则更长。

---

## 3. 完整非线性状态方程

### 3.1 横向偏移 $\dot{l}$

横向偏移的变化率等于车速在法向上的投影：

$$\dot{l} = v_x \sin(\delta\theta)$$

### 3.2 参考线弧长 $\dot{s}$

车辆沿切向的速度分量 $v_x \cos(\delta\theta)$ 对应的参考线弧长变化率，需除以缩放因子：

$$\dot{s} = \frac{v_x \cos(\delta\theta)}{1 - \kappa_r(s) \cdot l}$$

### 3.3 航向偏差 $\dot{\delta\theta}$

车辆的绝对航向角变化率（yaw rate）为曲率乘以速度：

$$\dot{\theta}_{vehicle} = \kappa \cdot v_x$$

参考线航向角变化率为参考线曲率乘以参考线弧长变化率：

$$\dot{\theta}_{ref} = \kappa_r(s) \cdot \dot{s} = \frac{\kappa_r(s) \cdot v_x \cos(\delta\theta)}{1 - \kappa_r(s) \cdot l}$$

两者之差即为航向偏差变化率：

$$\dot{\delta\theta} = \kappa \cdot v_x - \frac{\kappa_r(s) \cdot v_x \cos(\delta\theta)}{1 - \kappa_r(s) \cdot l}$$

### 3.4 曲率 $\dot{\kappa}$

$$\dot{\kappa} = d\kappa$$

### 3.5 车速 $\dot{v}_x$

$$\dot{v}_x = a_x$$

### 3.6 纵向加速度 $\dot{a}_x$

$$\dot{a}_x = j_x$$

### 3.7 汇总

$$\boxed{
\begin{cases}
\dot{l} = v_x \sin(\delta\theta) \\[6pt]
\dot{\delta\theta} = \kappa \cdot v_x - \dfrac{\kappa_r(s) \cdot v_x \cos(\delta\theta)}{1 - \kappa_r(s) \cdot l} \\[6pt]
\dot{\kappa} = d\kappa \\[6pt]
\dot{s} = \dfrac{v_x \cos(\delta\theta)}{1 - \kappa_r(s) \cdot l} \\[6pt]
\dot{v}_x = a_x \\[6pt]
\dot{a}_x = j_x
\end{cases}
}$$

---

## 4. 近似分析

模型中有两个主要非线性来源，可以分别考虑是否简化。

### 4.1 简化候选 1：$1 - \kappa_r(s) \cdot l \approx 1$

取值范围：$\kappa_r \in [-0.01, 0.01]$，$l \in [-8, 8]$

| $\kappa_r$ | $l$ | $\kappa_r \cdot l$ | $\frac{1}{1 - \kappa_r l}$ | 相对误差 |
|---|---|---|---|---|
| 0.005 | 4 | 0.02 | 1.020 | **2.0%** |
| 0.01 | 4 | 0.04 | 1.042 | **4.2%** |
| 0.005 | 8 | 0.04 | 1.042 | **4.2%** |
| 0.01 | 8 | 0.08 | 1.087 | **8.7%** |

**结论**：在大曲率 + 大横向偏移场景下，最大误差达 **~8.7%**，直接影响 $\dot{s}$ 和 $\dot{\delta\theta}$ 的精度。

### 4.2 简化候选 2：$\sin(\delta\theta) \approx \delta\theta$，$\cos(\delta\theta) \approx 1$

取值范围：$\delta\theta \in [-10°, 10°] = [-0.1745, 0.1745]$ rad

| $\delta\theta$ | $\sin$ 真值 | 线性近似 | sin 相对误差 | $\cos$ 真值 | cos 近似=1 误差 |
|---|---|---|---|---|---|
| 5° (0.0873 rad) | 0.0872 | 0.0873 | **0.13%** | 0.9962 | **0.38%** |
| 10° (0.1745 rad) | 0.1736 | 0.1745 | **0.51%** | 0.9848 | **1.52%** |

**结论**：在 ±10° 范围内，最大误差仅 **~1.5%**，工程上可忽略。

### 4.3 对比总结

| 简化方式 | 最大相对误差 | 影响的方程 |
|----------|-------------|-----------|
| $1 - \kappa_r l \approx 1$ | **~8.7%** | $\dot{s}$, $\dot{\delta\theta}$ |
| $\sin \approx \delta\theta$, $\cos \approx 1$ | **~1.5%** | $\dot{l}$, $\dot{s}$, $\dot{\delta\theta}$ |

**小角度近似的误差远小于曲率-偏移近似**。

---

## 5. 推荐：半简化模型（代码实际使用）

保留 $1 - \kappa_r l$ 项，仅做小角度近似：

$$\boxed{
\begin{cases}
\dot{l} = v_x \cdot \delta\theta \\[6pt]
\dot{\delta\theta} = \kappa \cdot v_x - \dfrac{\kappa_r(s) \cdot v_x}{1 - \kappa_r(s) \cdot l} \\[6pt]
\dot{\kappa} = d\kappa \\[6pt]
\dot{s} = \dfrac{v_x}{1 - \kappa_r(s) \cdot l} \\[6pt]
\dot{v}_x = a_x \\[6pt]
\dot{a}_x = j_x
\end{cases}
}$$

此模型特点：

- 非线性仅来自 $\frac{1}{1 - \kappa_r l}$ 和状态量之间的乘积项，结构简洁
- 在大横向偏移（换道场景 $l$ 可达 ±3.5m 甚至更多）+ 弯道时，保留了关键的几何耦合
- 小角度近似带来的误差在工程上可忽略

---

## 6. OCP 问题定义

### 6.1 约束

| 约束 | 表达式 | 类型 | 作用范围 |
|------|--------|------|----------|
| 起点等式 | $x(0) = x_0$ | 硬约束 | stage 0 |
| 横向偏移 | $l_{\min} \leq l \leq l_{\max}$ | 软约束 | stages 1..N |
| 终点弧长 | $s(T) \leq S_{\max}$ | 软约束 | stage N |
| 速度非负 | $v_x \geq 0$ | 硬约束 | stages 1..N |
| 摩擦椭圆 | $\dfrac{a_x^2}{a_{x,\max}^2} + \dfrac{(v_x^2 \kappa)^2}{a_{y,\max}^2} \leq 1$ | 硬约束 | stages 1..N |

其中侧向加速度 $a_y = v_x^2 \cdot \kappa$。

### 6.2 代价函数

$$J = \sum_{k=0}^{N-1} \| y_k - y_{ref,k} \|^2_W + \| y_N - y_{ref,N} \|^2_{W_e}$$

其中：

$$y_k = [l,\ \delta\theta,\ \kappa,\ v_x,\ a_x,\ d\kappa,\ j_x]^T$$

$$y_N = [l,\ \delta\theta,\ \kappa,\ v_x,\ a_x]^T$$

参考值：$l_{ref}$ 和 $v_{x,ref}$ 由输入设置，其余参考值取 $0$。

### 补充：全线性化条件

若场景满足 $|\kappa_r| \leq 0.003$（曲率半径 > 333m）且 $|l| \leq 4$ m，则 $\kappa_r l \leq 0.012$，两个简化同时使用也仅有 ~1.2% 误差，此时可进一步全线性化为：

$$\begin{cases}
\dot{l} = v_x \cdot \delta\theta \\
\dot{\delta\theta} = (\kappa - \kappa_r) \cdot v_x \\
\dot{\kappa} = d\kappa \\
\dot{s} = v_x \\
\dot{v}_x = a_x \\
\dot{a}_x = j_x
\end{cases}$$
