import math


class Vec2d:
    def __init__(self, x=0.0, y=0.0):
        """初始化二维向量 (x, y)"""
        self.x = float(x)
        self.y = float(y)

    @classmethod
    def from_angle(cls, angle_rad):
        """从角度（弧度）创建单位向量"""
        return cls(math.cos(angle_rad), math.sin(angle_rad))

    def __add__(self, other):
        """向量加法 v1 + v2"""
        return Vec2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """向量减法 v1 - v2"""
        return Vec2d(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """向量乘以标量 v * k"""
        return Vec2d(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        """标量乘以向量 k * v"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """向量除以标量 v / k"""
        return Vec2d(self.x / scalar, self.y / scalar)

    def __neg__(self):
        """向量取负 -v"""
        return Vec2d(-self.x, -self.y)

    def dot(self, other):
        """点积（内积）v1·v2"""
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """叉积（返回标量）v1×v2"""
        return self.x * other.y - self.y * other.x

    def rotate(self, angle_rad):
        """旋转向量（弧度）"""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return Vec2d(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def rotate_90(self, counterclockwise=True):
        """旋转90度（默认逆时针）"""
        return Vec2d(-self.y, self.x) if counterclockwise else Vec2d(self.y, -self.x)

    def rotate_180(self):
        """旋转180度"""
        return Vec2d(-self.x, -self.y)

    def rotate_270(self, counterclockwise=True):
        """旋转270度（默认逆时针）"""
        return Vec2d(self.y, -self.x) if counterclockwise else Vec2d(-self.y, self.x)

    def length(self):
        """向量长度（模）"""
        return math.hypot(self.x, self.y)

    def length_squared(self):
        """向量长度的平方"""
        return self.x**2 + self.y**2

    def normalized(self):
        """返回单位向量"""
        length = self.length()
        return Vec2d(0, 0) if length == 0 else self / length

    def angle(self):
        """返回向量的角度（弧度，从x轴正方向逆时针计算）"""
        return math.atan2(self.y, self.x)

    def distance_to(self, other):
        """计算到另一个向量的距离"""
        return (self - other).length()

    def is_approx(self, other, tol=1e-6):
        """近似相等比较（考虑浮点误差）"""
        return (abs(self.x - other.x) < tol and
                abs(self.y - other.y) < tol)

    def __repr__(self):
        return f"Vec2d({self.x:.4f}, {self.y:.4f})"

    def __iter__(self):
        """支持解包操作，如 x, y = vec"""
        yield self.x
        yield self.y


# 使用示例
if __name__ == "__main__":
    # 基本运算
    v1 = Vec2d(1, 2)
    v2 = Vec2d(3, 4)
    print(f"加法: {v1 + v2}")
    print(f"减法: {v1 - v2}")
    print(f"数乘: {v1 * 3}")
    print(f"数除: {v2 / 2}")

    # 向量运算
    print(f"点积: {v1.dot(v2)}")
    print(f"叉积: {v1.cross(v2)}")

    # 旋转演示
    v = Vec2d(1, 0)
    print(f"原始向量: {v}")
    print(f"旋转90度: {v.rotate_90()}")
    print(f"旋转45度: {v.rotate(math.pi/4)}")

    # 单位向量
    angle = math.pi/3  # 60度
    unit_v = Vec2d.from_angle(angle)
    print(f"60度单位向量: {unit_v}")
    print(f"向量角度: {math.degrees(unit_v.angle()):.1f}°")
    print(f"归一化检查: {unit_v.length():.6f}")
