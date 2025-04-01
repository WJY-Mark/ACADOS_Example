from typing import List, Tuple, Optional
from typing import List, Tuple
import bisect
import math


class Spline2dSeg:
    def __init__(self, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5):
        """二维五次多项式样条段（直接使用多项式系数构造）
        参数:
            a0-a5: x方向多项式系数 (a0 + a1*t + a2*t² + ... + a5*t⁵)
            b0-b5: y方向多项式系数 (b0 + b1*t + b2*t² + ... + b5*t⁵)
        """
        # x方向系数
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

        # y方向系数
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5

    def evaluate(self, t):
        """计算位置 (x, y)"""
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        x = self.a0 + self.a1*t + self.a2*t2 + self.a3*t3 + self.a4*t4 + self.a5*t5
        y = self.b0 + self.b1*t + self.b2*t2 + self.b3*t3 + self.b4*t4 + self.b5*t5
        return x, y

    def evaluate_derivative(self, t):
        """计算一阶导数 (dx/dt, dy/dt)"""
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        dx = self.a1 + 2*self.a2*t + 3*self.a3*t2 + 4*self.a4*t3 + 5*self.a5*t4
        dy = self.b1 + 2*self.b2*t + 3*self.b3*t2 + 4*self.b4*t3 + 5*self.b5*t4
        return dx, dy

    def evaluate_second_derivative(self, t):
        """计算二阶导数 (d²x/dt², d²y/dt²)"""
        t2 = t * t
        t3 = t2 * t
        d2x = 2*self.a2 + 6*self.a3*t + 12*self.a4*t2 + 20*self.a5*t3
        d2y = 2*self.b2 + 6*self.b3*t + 12*self.b4*t2 + 20*self.b5*t3
        return d2x, d2y

    def evaluate_third_derivative(self, t):
        """计算三阶导数 (d³x/dt³, d³y/dt³)"""
        t2 = t * t
        d3x = 6*self.a3 + 24*self.a4*t + 60*self.a5*t2
        d3y = 6*self.b3 + 24*self.b4*t + 60*self.b5*t2
        return d3x, d3y

    def evaluate_fourth_derivative(self, t):
        """计算四阶导数 (d⁴x/dt⁴, d⁴y/dt⁴)"""
        d4x = 24*self.a4 + 120*self.a5*t
        d4y = 24*self.b4 + 120*self.b5*t
        return d4x, d4y

    def evaluate_fifth_derivative(self, t):
        """计算五阶导数 (d⁵x/dt⁵, d⁵y/dt⁵)"""
        return (120*self.a5, 120*self.b5)  # 常数

    def get_coefficients(self):
        """获取多项式系数"""
        return {
            'x': [self.a0, self.a1, self.a2, self.a3, self.a4, self.a5],
            'y': [self.b0, self.b1, self.b2, self.b3, self.b4, self.b5]
        }

    def get_derivatives_at(self, t=0):
        """获取指定时刻的各阶导数"""
        return {
            'x': self.evaluate(t)[0],
            'dx': self.evaluate_derivative(t)[0],
            'd2x': self.evaluate_second_derivative(t)[0],
            'd3x': self.evaluate_third_derivative(t)[0],
            'd4x': self.evaluate_fourth_derivative(t)[0],
            'd5x': self.evaluate_fifth_derivative(t)[0],
            'y': self.evaluate(t)[1],
            'dy': self.evaluate_derivative(t)[1],
            'd2y': self.evaluate_second_derivative(t)[1],
            'd3y': self.evaluate_third_derivative(t)[1],
            'd4y': self.evaluate_fourth_derivative(t)[1],
            'd5y': self.evaluate_fifth_derivative(t)[1],
        }
    @classmethod
    def spline_seg_from_derivatives(cls, x, dx, d2x, d3x, d4x, d5x, y, dy, d2y, d3y, d4y, d5y):
        """从各阶导数的初始值构造样条段
            参数命名遵循物理意义：
                x, y: 位置
                dx, dy: 一阶导数（速度）
                d2x, d2y: 二阶导数（加速度）
                d3x, d3y: 三阶导数（jerk）
                d4x, d4y: 四阶导数（snap）
                d5x, d5y: 五阶导数（crackle）
            """
        # 将导数转换为多项式系数
        return Spline2dSeg(
            a0=x,
            a1=dx,
            a2=d2x/2.0,
            a3=d3x/6.0,
            a4=d4x/24.0,
            a5=d5x/120.0,
            b0=y,
            b1=dy,
            b2=d2y/2.0,
            b3=d3y/6.0,
            b4=d4y/24.0,
            b5=d5y/120.0
        )


class Spline2d:
    def __init__(self):
        """二维分段五次样条曲线"""
        self.segs: List[Spline2dSeg] = []    # 样条段列表
        self.t_knots: List[float] = []       # 节点参数（分段点）
        self.t_span: List[float] = []        # 各段时间长度

    def add_segment(self, seg: Spline2dSeg, duration: float):
        """添加样条段并更新时间节点"""
        if duration <= 0:
            raise ValueError("Duration must be positive")

        if not self.segs:
            self.t_knots = [0.0, duration]
        else:
            self.t_knots.append(self.t_knots[-1] + duration)
        self.t_span.append(duration)
        self.segs.append(seg)

    def find_segment_index(self, t: float) -> Tuple[int, float]:
        """定位时间t对应的段索引和段内时间"""
        if not self.segs:
            raise ValueError("No segments in spline")
        if t < 0 or t > self.t_knots[-1]:
            raise ValueError(f"Time {t} out of range [0, {self.t_knots[-1]}]")

        seg_idx = bisect.bisect_right(self.t_knots, t) - 1
        seg_idx = max(0, min(seg_idx, len(self.segs)-1))
        return seg_idx, t - self.t_knots[seg_idx]

    def evaluate_derivative(self, t: float, order: int = 0) -> Tuple[float, float]:
        """通用导数求值函数
        
        Args:
            t: 时间参数
            order: 导数阶数 (0-位置, 1-速度, 2-加速度,...)
            
        Returns:
            (dx, dy) 元组
        """
        seg_idx, t_local = self.find_segment_index(t)
        seg = self.segs[seg_idx]

        if order == 0:
            return seg.evaluate(t_local)
        elif order == 1:
            return seg.evaluate_derivative(t_local)
        elif order == 2:
            return seg.evaluate_second_derivative(t_local)
        elif order == 3:
            return seg.evaluate_third_derivative(t_local)
        elif order == 4:
            return seg.evaluate_fourth_derivative(t_local)
        elif order == 5:
            return seg.evaluate_fifth_derivative(t_local)
        else:
            raise ValueError(
                f"Unsupported derivative order: {order} (0-5 supported)")

    # 保持原有方法作为快捷方式
    def evaluate(self, t: float) -> Tuple[float, float]:
        return self.evaluate_derivative(t, 0)

    def evaluate_first_derivative(self, t: float) -> Tuple[float, float]:
        return self.evaluate_derivative(t, 1)

    def evaluate_second_derivative(self, t: float) -> Tuple[float, float]:
        return self.evaluate_derivative(t, 2)

    # ...其他高阶导数快捷方法...

    def get_curvature(self, t: float) -> float:
        """计算曲率 (基于速度和加速度)"""
        vel = self.evaluate_derivative(t, 1)
        acc = self.evaluate_derivative(t, 2)
        return (vel[0]*acc[1] - vel[1]*acc[0]) / (vel[0]**2 + vel[1]**2)**1.5

    def get_tangent_angle(self, t: float) -> float:
        """计算切线角度 (弧度)"""
        dx, dy = self.evaluate_derivative(t, 1)
        return math.atan2(dy, dx)

    def __repr__(self) -> str:
        return f"Spline2d(n_segments={len(self.segs)}, duration={self.t_knots[-1] if self.t_knots else 0})"


# 使用示例
if __name__ == "__main__":
    spline = Spline2d()

    # 添加三段样条
    spline.add_segment(Spline2dSeg(
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1.0)  # 0-1秒
    spline.add_segment(Spline2dSeg(
        1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0), 2.0)  # 1-3秒
    spline.add_segment(Spline2dSeg(3, 0, 0, 0, 0, 0, 2,
                       0, 0, 0, 0, 0), 1.5)  # 3-4.5秒

    # 测试find_segment_index
    test_times = [0, 0.5, 1.0, 2.5, 3.0, 4.0, 4.5]
    for t in test_times:
        seg_idx, t_local = spline.find_segment_index(t)
        t_start, t_end = spline.get_segment_time_range(seg_idx)
        print(
            f"t={t:.1f} → seg_{seg_idx} (t_local={t_local:.1f}), range=[{t_start:.1f}, {t_end:.1f})")
