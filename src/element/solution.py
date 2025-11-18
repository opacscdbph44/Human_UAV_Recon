import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from src.element.instance_class import InstanceClass


@dataclass
class Solution:
    """
    精简的Solution类,只存储核心数据
    使用dataclass和__slots__减少内存占用,提高copy效率
    """

    Solver_name: str
    status: str
    solve_time: float
    mip_gap: float
    instance_name: str
    instance_scale: int
    num_vehicles: int
    num_nodes: int
    routes: List[List[int]] = field(default_factory=list)
    coverage: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)
    objectives: List[float] = field(
        default_factory=lambda: [float("inf"), float("inf")]
    )
    rank: int = -1
    crowding_distance: float = 0.0

    def __post_init__(self):
        """初始化后处理,设置默认的routes"""
        if not self.routes:
            self.routes = [[] for _ in range(self.num_vehicles)]

    def __str__(self):
        str_info = (
            "Solution("
            f"instance={self.instance_name}(N={self.instance_scale}), "
            f"status={self.status}, "
            f"objectives={self.objectives})"
        )
        return str_info

    __repr__ = __str__

    @classmethod
    def new_solution(cls, instance: InstanceClass, solver_name: str = "") -> "Solution":
        """根据算例初始化空解对象"""
        return cls(
            Solver_name=solver_name,
            status="Not Solved",
            solve_time=0.0,
            mip_gap=float("inf"),
            instance_name=instance.name,
            instance_scale=instance.demand_num,
            num_vehicles=instance.total_veh_num,
            num_nodes=instance.total_node_num,
        )

    def set_coverage(self, vehicle_id: int, visit_node: int, covered_nodes: Set[int]):
        """设置覆盖关系"""
        if covered_nodes:
            self.coverage[(vehicle_id, visit_node)] = covered_nodes.copy()

    def get_coverage(self, vehicle_id: int, visit_node: int) -> Set[int]:
        """获取覆盖的节点"""
        return self.coverage.get((vehicle_id, visit_node), set())

    def copy(self) -> "Solution":
        """拷贝解，保留目标函数值但重置种群相关属性"""
        new_sol = Solution(
            self.Solver_name,
            self.status,
            self.solve_time,
            self.mip_gap,
            self.instance_name,
            self.instance_scale,
            self.num_vehicles,
            self.num_nodes,
        )
        new_sol.routes = [route.copy() for route in self.routes]
        new_sol.coverage = {k: v.copy() for k, v in self.coverage.items()}

        # 保留已计算的目标函数值
        new_sol.objectives = self.objectives.copy()

        # 重置种群相关属性（会随种群变化而变化）
        new_sol.rank = -1
        new_sol.crowding_distance = 0.0

        return new_sol

    def to_dict(self) -> Dict:
        """将Solution对象转换为字典形式，便于存储和传输"""

        mip_gap = "inf" if math.isinf(self.mip_gap) else self.mip_gap
        objectives = ["inf" if math.isinf(obj) else obj for obj in self.objectives]
        crowding_distance = (
            "inf" if math.isinf(self.crowding_distance) else self.crowding_distance
        )
        return {
            "Solver_name": self.Solver_name,
            "status": self.status,
            "solve_time": self.solve_time,
            "mip_gap": mip_gap,
            "instance_name": self.instance_name,
            "instance_scale": self.instance_scale,
            "num_vehicles": self.num_vehicles,
            "num_nodes": self.num_nodes,
            "routes": self.routes,
            "coverage": {
                f"{k[0]}_{k[1]}": list(v) for k, v in self.coverage.items()
            },  # 转换为可序列化形式
            "objectives": objectives,
            "rank": self.rank,
            "crowding_distance": crowding_distance,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Solution":
        """从字典形式创建Solution对象"""
        sol = cls(
            Solver_name=str(data["Solver_name"]),
            status=str(data["status"]),
            solve_time=float(data["solve_time"]),
            mip_gap=float(data["mip_gap"]),
            instance_name=str(data["instance_name"]),
            instance_scale=int(data["instance_scale"]),
            num_vehicles=int(data["num_vehicles"]),
            num_nodes=int(data["num_nodes"]),
        )
        # 确保 routes 是 List[List[int]]
        sol.routes = [[int(node) for node in route] for route in data["routes"]]

        # 严格匹配形如"int_int"的键名
        coverage_dict = {}
        for k, v in data["coverage"].items():
            match = re.fullmatch(r"(\d+)_(\d+)", k)
            if match:
                key_tuple = (int(match.group(1)), int(match.group(2)))
                coverage_dict[key_tuple] = set(int(node) for node in v)
            else:
                raise ValueError(f"Invalid coverage key format: {k}")
        sol.coverage = coverage_dict

        # 确保 objectives 是 List[float]
        sol.objectives = [float(obj) for obj in data["objectives"]]
        sol.rank = int(data["rank"])
        sol.crowding_distance = float(data["crowding_distance"])
        return sol

    def _init_attrs(
        self,
        status="初始化属性",
    ):
        """初始化解的属性"""
        self.status = status
        self.objectives = [float("inf"), float("inf")]
        self.rank = -1
        self.crowding_distance = 0.0

    def init_all_routes(
        self,
        status="初始化路径",
    ):
        """初始化所有车辆的路径为起点和终点相同的空路径"""
        self._init_attrs(status=status)
        self.routes = [[0, 0] for _ in range(self.num_vehicles)]
        self.coverage = {}  # 清空覆盖信息

    def get_included_nodes(self) -> Set[int]:
        """获取解中包含的所有节点"""
        included = set()
        for route in self.routes:
            included.update(route)
        for vehicle_id, visit_node in self.coverage.keys():
            covered_nodes = self.coverage[(vehicle_id, visit_node)]
            included.update(covered_nodes)
        return included
