import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import List, Tuple

# ==================== 枚举定义 ====================


class BaseSelectMode(str, Enum):
    RANDOM = "random"
    ASSIGN = "assign"


class DemandSelection(str, Enum):
    TOP_K = "Top_k"
    RANDOM = "Random"


class DistribType(str, Enum):
    RANDOM = "Random"
    LINEAR = "Linear"
    RADIAL = "Radial"
    UNDEFINED = "Undefined"


class OptimizeSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class CoolingStrategy(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class GroupingMethod(str, Enum):
    """需求点分组方式"""

    NONE = "none"  # 不分组
    KMEANS = "kmeans"  # K-means聚类
    GRID = "grid"  # 网格分组
    DISTANCE = "distance"  # 基于距离阈值分组


# ==================== 基础类 ====================


@dataclass
class BaseConfig:
    """配置基类"""

    def to_dict(self) -> dict:
        """转换为字典"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


# ==================== 配置类 ====================


@dataclass
class VehicleConfig(BaseConfig):
    """车辆参数配置"""

    ground_veh_speed: float = 50.0
    drone_speed: float = 20.0
    ground_veh_travel_power_rate: float = 0.1
    ground_veh_comm_power_rate: float = 0.01
    drone_travel_power_rate: float = 0.05
    drone_comm_power_rate: float = 0.005
    ground_veh_range: float = 5000.0
    drone_range: float = 1000.0
    ground_veh_comm_radius: float = 0.0
    drone_comm_radius: float = 50.0

    @classmethod
    def from_dict(cls, data: dict) -> "VehicleConfig":
        """从字典创建，只使用提供的字段，其余使用默认值"""
        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)

    def __post_init__(self) -> None:
        if self.ground_veh_speed <= 0 or self.drone_speed <= 0:
            raise ValueError("Vehicle speed must be positive")
        if any(
            x <= 0
            for x in [
                self.ground_veh_travel_power_rate,
                self.ground_veh_comm_power_rate,
                self.drone_travel_power_rate,
                self.drone_comm_power_rate,
                self.ground_veh_range,
                self.drone_range,
            ]
        ):
            raise ValueError("Power rates and energy capacity must be positive")
        if self.ground_veh_comm_radius < 0 or self.drone_comm_radius < 0:
            raise ValueError("Communication radius must be non-negative")


@dataclass
class InstanceParam(BaseConfig):
    """问题实例参数"""

    name: str
    base_num: int = 1
    base_select_mode: BaseSelectMode = BaseSelectMode.ASSIGN
    base_select_param: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.5, 0.5)]
    )
    demand_num: int = 100
    demand_selection: DemandSelection = DemandSelection.RANDOM
    distrib_type: DistribType = DistribType.RANDOM
    isolate_ratio: float = 0.5
    distrib_center: Tuple[float, float] = (0, 0)
    ground_veh_num: int = 2
    drone_num: int = 2
    vehicle_config: VehicleConfig = field(default_factory=VehicleConfig)
    min_success_task_rate: float = 0.0
    min_visit_time: float = 19.0
    max_visit_time: float = 100.0  # 默认统一的最晚访问时间
    max_visit_time_range: Tuple[float, float] | None = (
        None  # 如果设置,则按此区间随机生成
    )
    # 分组相关参数
    grouping_method: GroupingMethod = GroupingMethod.KMEANS
    max_group_demand: int = 4  # 每个组最大的个体数
    num_groups: int = 0  # 分组数量（用于kmeans和grid）
    distance_threshold: float = 100.0  # 距离阈值（用于distance方法）
    grid_size: Tuple[int, int] = (3, 3)  # 网格尺寸（用于grid方法）

    @classmethod
    def from_dict(cls, data: dict) -> "InstanceParam":
        """从字典创建"""
        kwargs = {"name": data["name"]}  # name 是必需的

        # 处理简单字段
        simple_fields = [
            "base_num",
            "demand_num",
            "isolate_ratio",
            "ground_veh_num",
            "drone_num",
            "min_success_task_rate",
            "min_visit_time",
            "max_visit_time",
            "num_groups",
            "distance_threshold",
        ]
        for field_name in simple_fields:
            if field_name in data:
                kwargs[field_name] = data[field_name]

        # 处理枚举字段
        if "base_select_mode" in data:
            kwargs["base_select_mode"] = BaseSelectMode(data["base_select_mode"])
        if "demand_selection" in data:
            kwargs["demand_selection"] = DemandSelection(data["demand_selection"])
        if "distrib_type" in data:
            kwargs["distrib_type"] = DistribType(data["distrib_type"])
        if "grouping_method" in data:
            kwargs["grouping_method"] = GroupingMethod(data["grouping_method"])

        # 处理 Tuple 字段
        if "distrib_center" in data:
            kwargs["distrib_center"] = tuple(data["distrib_center"])
        if "grid_size" in data:
            kwargs["grid_size"] = tuple(data["grid_size"])
        if "max_visit_time_range" in data and data["max_visit_time_range"] is not None:
            kwargs["max_visit_time_range"] = tuple(data["max_visit_time_range"])

        # 处理 List[Tuple] 字段
        if "base_select_param" in data:
            kwargs["base_select_param"] = [tuple(x) for x in data["base_select_param"]]

        # 处理嵌套对象
        if "vehicle_config" in data:
            kwargs["vehicle_config"] = VehicleConfig.from_dict(data["vehicle_config"])

        return cls(**kwargs)

    def __post_init__(self) -> None:
        if self.base_num <= 0 or self.demand_num <= 0:
            raise ValueError("base_num and demand_num must be positive")
        if not 0 <= self.isolate_ratio <= 1 or not 0 <= self.min_success_task_rate <= 1:
            raise ValueError("Ratios must be in [0, 1]")
        if self.min_visit_time < 0:
            raise ValueError("min_comm_time must be non-negative")
        if self.max_visit_time < 0:
            raise ValueError("max_visit_time must be non-negative")
        if self.max_visit_time_range is not None:
            if len(self.max_visit_time_range) != 2:
                raise ValueError("max_visit_time_range must be a tuple of (min, max)")
            if self.max_visit_time_range[0] < 0 or self.max_visit_time_range[1] < 0:
                raise ValueError("max_visit_time_range values must be non-negative")
            if self.max_visit_time_range[0] > self.max_visit_time_range[1]:
                raise ValueError("max_visit_time_range min must be <= max")
        if not self.name.endswith(".tsp"):
            raise ValueError("name must end with .tsp")
        if (
            self.base_select_mode == BaseSelectMode.ASSIGN
            and not self.base_select_param
        ):
            raise ValueError("base_select_param required for assign mode")
        if self.ground_veh_num == 0 and self.drone_num == 0:
            raise ValueError("At least one vehicle type required")


@dataclass
class ObjConfig(BaseConfig):
    """目标函数配置"""

    name: List[str] = field(default_factory=lambda: ["Makespan", "Priority_Score"])
    sense: List[OptimizeSense] = field(
        default_factory=lambda: [OptimizeSense.MINIMIZE, OptimizeSense.MAXIMIZE]
    )

    @classmethod
    def from_dict(cls, data: dict) -> "ObjConfig":
        """从字典创建"""
        kwargs = {}
        if "name" in data:
            kwargs["name"] = data["name"]
        if "sense" in data:
            kwargs["sense"] = [OptimizeSense(s) for s in data["sense"]]
        return cls(**kwargs)

    def __post_init__(self) -> None:
        if len(self.name) != len(self.sense):
            raise ValueError("Objective names and senses must have same length")


@dataclass
class AlgorithmConfig(BaseConfig):
    """算法参数配置"""

    # 目标函数参数类
    obj_config: ObjConfig = field(default_factory=ObjConfig)

    # epsilon点数
    time_limit: int = 3600
    big_m: float = 1e6
    max_iter: int = 1000
    enable_early_stop: bool = True
    max_not_improve_iter: int = 1000
    early_stop_threshold: float = 1e-5
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    alns_max_iter: int = 1000
    alns_segment_size: int = 1
    destroy_degree_min: float = 0.1
    destroy_degree_max: float = 0.4
    decay_rate: float = 0.9
    multi_start_num: int = 4
    pheromone_evaporation_rate: float = 0.05
    tau_max: float = 10.0
    tau_min: float = 0.01

    @classmethod
    def from_dict(cls, data: dict) -> "AlgorithmConfig":
        """从字典创建"""
        kwargs = {}

        # 处理简单字段
        simple_fields = [
            "time_limit",
            "big_m",
            "max_iter",
            "enable_early_stop",
            "max_not_improve_iter",
            "early_stop_threshold",
            "crossover_prob",
            "mutation_prob",
            "alns_max_iter",
            "alns_segment_size",
            "destroy_degree_min",
            "destroy_degree_max",
            "decay_rate",
        ]
        for field_name in simple_fields:
            if field_name in data:
                kwargs[field_name] = data[field_name]

        # 处理嵌套对象
        if "obj_config" in data:
            kwargs["obj_config"] = ObjConfig.from_dict(data["obj_config"])

        return cls(**kwargs)

    def __post_init__(self) -> None:
        if any(
            x <= 0
            for x in [
                self.max_iter,
                self.time_limit,
            ]
        ):
            raise ValueError("Iteration counts and sizes must be positive")
        if not 0 <= self.crossover_prob <= 1 or not 0 <= self.mutation_prob <= 1:
            raise ValueError("Probabilities must be in [0, 1]")
        if not 0 < self.destroy_degree_min < self.destroy_degree_max <= 1:
            raise ValueError("Invalid destroy degree range")


@dataclass
class Config(BaseConfig):
    """综合配置类"""

    instance_param: InstanceParam
    algorithm_config: AlgorithmConfig
    random_seed: int = 42
    save_file: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """从字典创建"""
        return cls(
            instance_param=InstanceParam.from_dict(data["instance_param"]),
            algorithm_config=AlgorithmConfig.from_dict(data["algorithm_config"]),
            random_seed=data.get("random_seed", 42),
        )

    def __str__(self) -> str:
        str_info = (
            f"算例名称: {self.instance_param.name} "
            f"(需求点 = {self.instance_param.demand_num}个, "
            f"地面车 = {self.instance_param.ground_veh_num}个, "
            f"无人机 = {self.instance_param.drone_num}个), "
            f"算法配置: 最大迭代 = {self.algorithm_config.max_iter}, "
            f"交叉概率 = {self.algorithm_config.crossover_prob}, "
            f"变异概率 = {self.algorithm_config.mutation_prob}"
        )
        return str_info

    __repr__ = __str__


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建配置
    config = Config(
        instance_param=InstanceParam(name="test.tsp"),
        algorithm_config=AlgorithmConfig(),
    )

    # 序列化
    config_dict = config.to_dict()
    print("Serialized to dict")

    # 写入JSON文件
    json.dump(
        config_dict, open("temp/json/config.json", "w"), indent=2, ensure_ascii=False
    )

    # 反序列化
    config_restored = Config.from_dict(config_dict)
    print("Restored from dict")

    # 访问配置
    print(f"Demand: {config_restored.instance_param.demand_num}")
    print(f"Mode: {config_restored.instance_param.base_select_mode.value}")
    print(f"Max iter: {config_restored.algorithm_config.max_iter}")
