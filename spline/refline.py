"""
约 150 m 平面三次样条参考线：沿几何弧长每 1.5 m 采样，
计算位置、对参数 u 的一/二阶导数、朝向角与曲率，并可视化。

参数 u 为路点累积弦长（米），与 ``CubicSpline2d.from_waypoints`` 一致；
弧长沿样条几何长度用密化折线近似。
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np

from spline2d import CubicSpline2d


def _show_nonblocking() -> None:
    if plt.get_backend().lower() != "agg":
        plt.show(block=False)


def _dense_samples(
    spl: CubicSpline2d, n: int = 8000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, spl.t_knots[-1], n)
    xy = np.array([spl.evaluate(float(t)) for t in u])
    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    return u, xy, s


def total_arc_length(spl: CubicSpline2d, n: int = 8000) -> float:
    _, _, s = _dense_samples(spl, n)
    return float(s[-1])


def parameter_at_arc_length(
    spl: CubicSpline2d, s_query: np.ndarray, n_dense: int = 8000
) -> np.ndarray:
    """给定弧长 s（米），反求参数 u（与路点累积弦长同量纲）。"""
    u, _, s = _dense_samples(spl, n_dense)
    s_query = np.asarray(s_query, dtype=float)
    s_max = s[-1]
    s_clipped = np.clip(s_query, 0.0, s_max)
    return np.interp(s_clipped, s, u)


def scale_waypoints_to_target_arc_length(
    pts: np.ndarray, target_m: float = 150.0
) -> np.ndarray:
    """均匀缩放路点，使拟合三次样条的几何弧长约为 target_m（米）。"""
    p = np.asarray(pts, dtype=float).copy()
    spl0 = CubicSpline2d.from_waypoints(p)
    L0 = total_arc_length(spl0)
    if L0 < 1e-9:
        raise ValueError("样条弧长过小")
    return p * (target_m / L0)


def build_demo_waypoints() -> np.ndarray:
    """生成一条 S 形平面折线（缩放后总弦长约 150 m）。"""
    t = np.linspace(0.0, 1.0, 12)
    xs = 140.0 * t
    ys = 18.0 * np.sin(2.2 * np.pi * t) + 6.0 * t
    return np.column_stack([xs, ys])


def sample_centerline(
    spl: CubicSpline2d, ds: float = 1.5, n_dense: int = 8000
) -> dict[str, np.ndarray]:
    L = total_arc_length(spl, n_dense)
    s = np.arange(0.0, L + 0.5 * ds, ds)
    if s[-1] < L - 1e-6:
        s = np.append(s, L)
    u = parameter_at_arc_length(spl, s, n_dense)

    n = len(s)
    x = np.zeros(n)
    y = np.zeros(n)
    dx = np.zeros(n)
    dy = np.zeros(n)
    ddx = np.zeros(n)
    ddy = np.zeros(n)
    psi = np.zeros(n)
    kappa = np.zeros(n)

    for i in range(n):
        ui = float(u[i])
        x[i], y[i] = spl.evaluate(ui)
        dx[i], dy[i] = spl.evaluate_derivative(ui, 1)
        ddx[i], ddy[i] = spl.evaluate_derivative(ui, 2)
        psi[i] = spl.get_tangent_angle(ui)
        kappa[i] = spl.get_curvature(ui)

    return {
        "s": s,
        "u": u,
        "x": x,
        "y": y,
        "dx_du": dx,
        "dy_du": dy,
        "d2x_du2": ddx,
        "d2y_du2": ddy,
        "heading_rad": psi,
        "curvature": kappa,
        "arc_length_m": L,
    }


def plot_centerline(data: dict[str, np.ndarray], out_path: str | None = None) -> None:
    s = data["s"]
    x, y = data["x"], data["y"]
    psi = data["heading_rad"]
    kappa = data["curvature"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax0 = axes[0, 0]
    ax0.plot(x, y, "-", color="0.2", lw=2.0, label="centerline")
    step = max(1, len(x) // 12)
    for i in range(0, len(x), step):
        c, si = np.cos(psi[i]), np.sin(psi[i])
        ax0.arrow(
            x[i],
            y[i],
            2.5 * c,
            2.5 * si,
            head_width=1.2,
            head_length=1.5,
            fc="C0",
            ec="C0",
            length_includes_head=True,
        )
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("x / m")
    ax0.set_ylabel("y / m")
    ax0.set_title("Trajectory and heading (arrows)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    axes[0, 1].plot(s, np.rad2deg(psi), color="C1", lw=1.5)
    axes[0, 1].set_xlabel("arc length s / m")
    axes[0, 1].set_ylabel("heading / deg")
    axes[0, 1].set_title("Heading (from +x)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(s, kappa * 1e3, color="C2", lw=1.5)
    axes[1, 0].set_xlabel("arc length s / m")
    axes[1, 0].set_ylabel("curvature * 1e3 (1/m)")
    axes[1, 0].set_title("kappa = (x'y''-y'x'') / (x'^2+y'^2)^(3/2)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(s, data["dx_du"], label="dx/du", color="C3", lw=1.2)
    axes[1, 1].plot(s, data["dy_du"], label="dy/du", color="C4", lw=1.2)
    axes[1, 1].plot(s, data["d2x_du2"], "--", label="d²x/du²", color="C3", alpha=0.7)
    axes[1, 1].plot(s, data["d2y_du2"], "--", label="d²y/du²", color="C4", alpha=0.7)
    axes[1, 1].set_xlabel("arc length s / m")
    axes[1, 1].set_ylabel("derivative w.r.t. chord param u")
    axes[1, 1].set_title("1st / 2nd derivatives (u = cum. chord length)")
    axes[1, 1].legend(loc="best", fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Cubic spline refline | arc length ~{data['arc_length_m']:.1f} m | ds ~ 1.5 m",
        fontsize=12,
    )
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _show_nonblocking()
    plt.close(fig)


def convert_to_motion_frame(
    data: dict[str, np.ndarray], v: float = 1.0
) -> dict[str, np.ndarray]:
    """将样条原始数据转换为运动学坐标系下的数据结构。

    定义（假设匀速 v，切向加速度 at=0）：
        dx  = v * cos(heading)
        dy  = v * sin(heading)
        at  = 0
        an  = curvature * v^2
        ddx = at*dx/v - an*dy/v
        ddy = at*dy/v + an*dx/v
    """
    psi = data["heading_rad"]
    kappa = data["curvature"]

    dx = v * np.cos(psi)
    dy = v * np.sin(psi)
    at = np.zeros_like(psi)
    an = kappa * v ** 2
    ddx = at * dx / v - an * dy / v
    ddy = at * dy / v + an * dx / v

    return {
        "s":         data["s"],
        "x":         data["x"],
        "y":         data["y"],
        "heading":   psi,
        "dx":        dx,
        "dy":        dy,
        "ddx":       ddx,
        "ddy":       ddy,
        "curvature": kappa,
        "at":        at,
        "an":        an,
    }


def _print_table(
    columns: list[tuple[str, str, str]],
    data: dict[str, np.ndarray],
    title: str = "",
) -> None:
    """通用对齐打印函数。columns: [(header, key, fmt), ...]"""
    col_w = [max(len(hdr), 11) for hdr, _, _ in columns]
    if title:
        print(f"\n{'=' * (sum(col_w) + 2 * (len(col_w) - 1))}")
        print(title)
        print('=' * (sum(col_w) + 2 * (len(col_w) - 1)))
    print("  ".join(f"{hdr:>{w}}" for (hdr, _, _), w in zip(columns, col_w)))
    print("  ".join("-" * w for w in col_w))
    n = len(data[columns[0][1]])
    for i in range(n):
        cells = [
            f"{fmt.format(float(data[key][i])):>{w}}"
            for (_, key, fmt), w in zip(columns, col_w)
        ]
        print("  ".join(cells))


def print_raw_data(data: dict[str, np.ndarray]) -> None:
    """打印 sample_centerline 返回的原始数据（对参数 u 的导数）。"""
    columns = [
        ("arc_s/m",    "s",          "{:8.3f}"),
        ("x/m",        "x",          "{:8.3f}"),
        ("y/m",        "y",          "{:8.3f}"),
        ("dx_du",      "dx_du",      "{:9.5f}"),
        ("dy_du",      "dy_du",      "{:9.5f}"),
        ("d2x_du2",    "d2x_du2",    "{:10.6f}"),
        ("d2y_du2",    "d2y_du2",    "{:10.6f}"),
        ("psi/rad",    "heading_rad","{:9.5f}"),
        ("kappa/1/m",  "curvature",  "{:11.7f}"),
    ]
    _print_table(columns, data, title="[RAW] sample_centerline output  (derivatives w.r.t. chord param u)")


def print_motion_data(mdata: dict[str, np.ndarray]) -> None:
    """打印 convert_to_motion_frame 转换后的运动学数据。"""
    columns = [
        ("arc_s/m",   "s",         "{:8.3f}"),
        ("x/m",       "x",         "{:8.3f}"),
        ("y/m",       "y",         "{:8.3f}"),
        ("heading/rad","heading",  "{:9.5f}"),
        ("dx m/s",    "dx",        "{:9.5f}"),
        ("dy m/s",    "dy",        "{:9.5f}"),
        ("ddx m/s2",  "ddx",       "{:10.6f}"),
        ("ddy m/s2",  "ddy",       "{:10.6f}"),
        ("kappa/1/m", "curvature", "{:11.7f}"),
        ("at m/s2",   "at",        "{:10.6f}"),
        ("an m/s2",   "an",        "{:11.7f}"),
    ]
    _print_table(columns, mdata, title="[MOTION FRAME]  v=1 m/s, dx=v*cos(psi), dy=v*sin(psi), at=0, an=kappa*v^2, ddx=at*dx/v-an*dy/v, ddy=at*dy/v+an*dx/v")


def _interp_motion_at_s(
    mdata: dict[str, np.ndarray],
    s: float,
) -> tuple[float, float, float, float, float, float]:
    """在弧长 s 处用线性插值从 mdata 取运动学量 (x, y, dx, dy, ddx, ddy)。"""
    sv = mdata["s"]
    x   = float(np.interp(s, sv, mdata["x"]))
    y   = float(np.interp(s, sv, mdata["y"]))
    dx  = float(np.interp(s, sv, mdata["dx"]))
    dy  = float(np.interp(s, sv, mdata["dy"]))
    ddx = float(np.interp(s, sv, mdata["ddx"]))
    ddy = float(np.interp(s, sv, mdata["ddy"]))
    return x, y, dx, dy, ddx, ddy


def eval_f_projection(
    mdata: dict[str, np.ndarray],
    xp: float,
    yp: float,
    s: float,
) -> float:
    """与牛顿循环中相同：f(s) = nx*dx + ny*dy，nx=(x-xp)/r，ny=(y-yp)/r。"""
    x, y, dx, dy, _, _ = _interp_motion_at_s(mdata, s)
    rx, ry = x - xp, y - yp
    r = math.sqrt(rx * rx + ry * ry)
    if r < 1e-30:
        return 0.0
    nx, ny = rx / r, ry / r
    return nx * dx + ny * dy


def project_point_newton(
    mdata: dict[str, np.ndarray],
    p: tuple[float, float],
    s_init: float | None = None,
    max_iter: int = 20,
    tol: float = 1e-2,
) -> dict:
    """将点 p=(xp,yp) 用牛顿法投影到样条曲线上（通过对 mdata 插值求值）。

    令 r = |(x-xp, y-yp)|，方向 n = ((x-xp)/r, (y-yp)/r)。正交条件写为
        f(s) = n·(dx, dy) = ((x-xp)*dx + (y-yp)*dy) / r

    求 f' 时把 **r 视为常数**（对 s 不显式求导），则
        f'(s) = (dx² + (x-xp)*ddx + dy² + (y-yp)*ddy) / r

    牛顿步 s ← s - f/f' 与未归一化的 g=(x-xp)dx+(y-yp)dy 情形等价（f/f' = g/g'）。

    初值：若 s_init=None，则在 mdata 采样点中取距 p 最近点的弧长。

    Returns:
        dict 含 s（投影弧长）、x、y（投影坐标）、dist（距离）、
        iters（迭代次数）、history（每步 (iter, s, f, f', proj_x, proj_y)）。
    """
    xp, yp = float(p[0]), float(p[1])
    sv = mdata["s"]
    s_max = float(sv[-1])

    if s_init is None:
        dist2 = (mdata["x"] - xp) ** 2 + (mdata["y"] - yp) ** 2
        s = float(sv[int(np.argmin(dist2))])
    else:
        s = float(s_init)

    s = max(float(sv[0]), min(s, s_max))

    history: list[tuple[int, float, float, float, float, float]] = []

    for i in range(max_iter):
        x, y, dx, dy, ddx, ddy = _interp_motion_at_s(mdata, s)
        rx, ry = x - xp, y - yp
        r = math.sqrt(rx * rx + ry * ry)
        if r < 1e-30:
            history.append((i, s, 0.0, 1.0, x, y))
            break
        nx, ny = rx / r, ry / r
        gprime = dx * dx + rx * ddx + dy * dy + ry * ddy
        f = nx * dx + ny * dy
        fp = gprime / r
        history.append((i, s, f, fp, x, y))

        if abs(fp) < 1e-30:
            break

        ds = f / fp
        s_new = max(float(sv[0]), min(s - ds, s_max))

        converged = abs(s_new - s) < tol
        s = s_new
        if converged:
            break

    x, y, dx, dy, ddx, ddy = _interp_motion_at_s(mdata, s)
    dist = math.sqrt((x - xp) ** 2 + (y - yp) ** 2)

    return {
        "s":       s,
        "x":       x,
        "y":       y,
        "dist":    dist,
        "iters":   len(history),
        "history": history,
    }


def plot_newton_projection(
    mdata: dict[str, np.ndarray],
    p: tuple[float, float],
    result: dict,
    out_path: str | None = None,
) -> None:
    """画出曲线、测试点与牛顿迭代各步投影点。"""
    xp, yp = p
    cx, cy = mdata["x"], mdata["y"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 曲线
    ax.plot(cx, cy, "-", color="0.3", lw=2.0, label="centerline")

    # 测试点
    ax.scatter([xp], [yp], s=100, color="C3", zorder=5, label=f"query p=({xp},{yp})")

    # 每次迭代的投影点
    n_hist = len(result["history"])
    cmap = plt.cm.plasma                          # 颜色随迭代变化
    for idx, (it, s_val, f_val, fp_val, px, py) in enumerate(result["history"]):
        color = cmap(idx / max(n_hist - 1, 1))
        ax.scatter([px], [py], s=80, color=color, zorder=6,
                   label=f"iter {it}: s={s_val:.4f}, f={f_val:.2e}")
        # 连线：测试点 → 迭代投影点（虚线）
        ax.plot([xp, px], [yp, py], "--", color=color, lw=0.8, alpha=0.7)

    # 最终投影点连到曲线（法线段）
    ax.plot([xp, result["x"]], [yp, result["y"]], "-", color="C2", lw=1.5,
            label=f"final proj s={result['s']:.4f} m, dist={result['dist']:.4f} m")
    ax.scatter([result["x"]], [result["y"]], s=120, marker="*",
               color="C2", zorder=7)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("Newton projection onto cubic spline\n"
                 "f=((x-xp)dx+(y-yp)dy)/r,  "
                 "f'=(dx²+(x-xp)ddx+dy²+(y-yp)ddy)/r  (r const in d/ds)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _show_nonblocking()
    plt.close(fig)


def plot_newton_f_curve(
    mdata: dict[str, np.ndarray],
    p: tuple[float, float],
    result: dict,
    n_curve: int = 4000,
    out_path: str | None = None,
) -> None:
    """画出整条 f(s)=nx*dx+ny*dy，并把牛顿迭代各步 (s, f(s)) 标在同一图上。"""
    xp, yp = float(p[0]), float(p[1])
    sv = mdata["s"]
    s0, s1 = float(sv[0]), float(sv[-1])
    s_dense = np.linspace(s0, s1, n_curve)
    f_dense = np.array([eval_f_projection(mdata, xp, yp, float(t)) for t in s_dense])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s_dense, f_dense, "-", color="0.3", lw=1.6, label=r"$f(s)=n_x d_x + n_y d_y$", zorder=1)
    ax.axhline(0.0, color="0.5", ls=":", lw=1.0, label="$f=0$ (foot)")

    s_hist = [h[1] for h in result["history"]]
    f_hist = [h[2] for h in result["history"]]
    n = len(s_hist)
    cmap = plt.cm.plasma
    for k in range(n):
        c = cmap(k / max(n - 1, 1))
        ax.scatter(
            [s_hist[k]], [f_hist[k]], s=85, color=c, zorder=5,
            edgecolors="k", linewidths=0.35,
            label=f"iter {k}" if k < 12 else None,
        )
    if n >= 2:
        ax.plot(s_hist, f_hist, "-", color="C1", lw=1.2, alpha=0.85, zorder=3, label="Newton polyline")

    f_final = eval_f_projection(mdata, xp, yp, float(result["s"]))
    ax.scatter(
        [result["s"]], [f_final], s=140, marker="*", color="C2", zorder=6,
        edgecolors="k", linewidths=0.4, label=f"final $s$={result['s']:.4f} m",
    )

    ax.set_xlabel("arc length $s$ / m")
    ax.set_ylabel("$f(s)$")
    ax.set_title(r"Residual $f(s)=\frac{(x-x_p)d_x+(y-y_p)d_y}{r}$  with Newton iterates")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _show_nonblocking()
    plt.close(fig)


def main() -> None:
    pts = scale_waypoints_to_target_arc_length(build_demo_waypoints(), 150.0)
    spl = CubicSpline2d.from_waypoints(pts)
    data = sample_centerline(spl, ds=1.5)
    out = os.path.join(os.path.dirname(__file__), "refline_curve.png")
    plot_centerline(data, out_path=out)
    print(f"\narc length: {data['arc_length_m']:.2f} m, samples: {len(data['s'])}, saved: {out}")

    print_raw_data(data)

    mdata = convert_to_motion_frame(data, v=1.0)
    print_motion_data(mdata)

    # ── 牛顿法投影演示 ──────────────────────────────────────────────────────────
    test_point = (50.0, 0.0)

    result = project_point_newton(mdata, test_point, 50.0)

    header = f"\n{'iter':>4}  {'s':>10}  {'f(s)':>14}  {'f_prime(s)':>14}  {'proj_x':>10}  {'proj_y':>10}"
    sep    = "-" * (len(header) - 1)

    print("\n" + "=" * 76)
    print("Newton projection:  f=((x-xp)dx+(y-yp)dy)/r,  "
          "f'=(dx²+(x-xp)ddx+dy²+(y-yp)ddy)/r  (r 对 s 视为常数)")
    print("=" * 76)
    print(f"\nQuery  p = ({test_point[0]:.2f}, {test_point[1]:.2f})")
    print(f"  → proj  s={result['s']:.5f} m, "
          f"(x,y)=({result['x']:.4f}, {result['y']:.4f}), "
          f"dist={result['dist']:.6f} m, "
          f"iters={result['iters']}")
    print(header)
    print(sep)
    for it, s_val, f_val, fp_val, px, py in result["history"]:
        print(f"{it:>4}  {s_val:>10.5f}  {f_val:>14.8f}  {fp_val:>14.8f}  {px:>10.4f}  {py:>10.4f}")

    proj_out = os.path.join(os.path.dirname(__file__), "newton_projection.png")
    plot_newton_projection(mdata, test_point, result, out_path=proj_out)
    print(f"\nNewton projection plot saved: {proj_out}")

    f_curve_out = os.path.join(os.path.dirname(__file__), "newton_f_vs_s.png")
    plot_newton_f_curve(mdata, test_point, result, out_path=f_curve_out)
    print(f"Newton f(s) full curve + iterates saved: {f_curve_out}")


if __name__ == "__main__":
    main()
