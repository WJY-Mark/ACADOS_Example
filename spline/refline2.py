"""
refline2：三次样条参考线生成 + 后续「圆弧法」投影的骨架脚本。

当前 main 仅负责：路点缩放 → 拟合样条 → 沿弧长采样 → 可视化与打印；
圆弧投影逻辑在「── 圆弧法投影」注释处自行扩展。
"""

from __future__ import annotations

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
    """生成一条 S 形平面折线（缩放后几何弧长约 150 m）。"""
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


def _print_table(
    columns: list[tuple[str, str, str]],
    data: dict[str, np.ndarray],
    title: str = "",
) -> None:
    col_w = [max(len(hdr), 11) for hdr, _, _ in columns]
    if title:
        wsum = sum(col_w) + 2 * (len(col_w) - 1)
        print(f"\n{'=' * wsum}")
        print(title)
        print("=" * wsum)
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
    """打印 sample_centerline 返回的原始数据。"""
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
    _print_table(
        columns,
        data,
        title="[RAW] sample_centerline (derivatives w.r.t. chord param u)",
    )


def main() -> None:
    pts = scale_waypoints_to_target_arc_length(build_demo_waypoints(), 150.0)
    spl = CubicSpline2d.from_waypoints(pts)
    data = sample_centerline(spl, ds=1.5)
    out = os.path.join(os.path.dirname(__file__), "refline2_curve.png")
    plot_centerline(data, out_path=out)
    print(f"\narc length: {data['arc_length_m']:.2f} m, samples: {len(data['s'])}, saved: {out}")

    print_raw_data(data)

    # ── 圆弧法投影（在此使用 spl、data、test_point 实现）──────────────────────
    test_point = (50.0, 0.0)


if __name__ == "__main__":
    main()
