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


class SteinerGenerationMode(str, Enum):
    KMEANS = "kmeans"
    GRID = "grid"
    ASSIGN = "assign"


class OptimizeSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class CoolingStrategy(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


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

    ground_veh_speed: float = 10.0
    drone_speed: float = 20.0
    ground_veh_travel_power_rate: float = 0.1
    ground_veh_comm_power_rate: float = 0.01
    drone_travel_power_rate: float = 0.05
    drone_comm_power_rate: float = 0.005
    ground_veh_energy_capacity: float = 1000.0
    drone_energy_capacity: float = 200.0
    ground_veh_comm_radius: float = 50.0
    drone_comm_radius: float = 30.0

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
                self.ground_veh_energy_capacity,
                self.drone_energy_capacity,
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
    steiner_num: int = 4
    steiner_generation_mode: SteinerGenerationMode = SteinerGenerationMode.GRID
    steiner_grid_size: Tuple[int, int] = (2, 2)
    steiner_coords: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.5, 0.8)]
    )
    min_success_task_rate: float = 0.0
    min_comm_time: float = 19.0

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
            "steiner_num",
            "min_success_task_rate",
            "min_comm_time",
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
        if "steiner_generation_mode" in data:
            kwargs["steiner_generation_mode"] = SteinerGenerationMode(
                data["steiner_generation_mode"]
            )

        # 处理 Tuple 字段
        if "distrib_center" in data:
            kwargs["distrib_center"] = tuple(data["distrib_center"])
        if "steiner_grid_size" in data:
            kwargs["steiner_grid_size"] = tuple(data["steiner_grid_size"])

        # 处理 List[Tuple] 字段
        if "base_select_param" in data:
            kwargs["base_select_param"] = [tuple(x) for x in data["base_select_param"]]
        if "steiner_coords" in data:
            kwargs["steiner_coords"] = [tuple(x) for x in data["steiner_coords"]]

        # 处理嵌套对象
        if "vehicle_config" in data:
            kwargs["vehicle_config"] = VehicleConfig.from_dict(data["vehicle_config"])

        return cls(**kwargs)

    def __post_init__(self) -> None:
        if self.base_num <= 0 or self.demand_num <= 0:
            raise ValueError("base_num and demand_num must be positive")
        if not 0 <= self.isolate_ratio <= 1 or not 0 <= self.min_success_task_rate <= 1:
            raise ValueError("Ratios must be in [0, 1]")
        if self.min_comm_time < 0:
            raise ValueError("min_comm_time must be non-negative")
        if not self.name.endswith(".tsp"):
            raise ValueError("name must end with .tsp")
        if (
            self.base_select_mode == BaseSelectMode.ASSIGN
            and not self.base_select_param
        ):
            raise ValueError("base_select_param required for assign mode")
        if (
            self.steiner_generation_mode == SteinerGenerationMode.ASSIGN
            and len(self.steiner_coords) < self.steiner_num
        ):
            raise ValueError("Insufficient steiner_coords for assign mode")
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

    # 目标函数优先级
    makespan_priority: int = 0
    score_priority: int = 1
    # 目标函数参数类
    obj_config: ObjConfig = field(default_factory=ObjConfig)

    # epsilon点数
    num_epsilon_points: int = 10
    time_limit: int = 1000
    big_m: float = 1e6
    max_iter: int = 200
    pop_size: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    alns_max_iter: int = 1000
    alns_segment_size: int = 1
    destroy_degree_min: float = 0.1
    destroy_degree_max: float = 0.4
    alns_initial_temp: float = 100.0
    alns_cooling_rate: float = 0.95
    alns_final_temp: float = 0.1
    alns_tabu_tenure: int = 0
    alns_operator_score_decay_rate: float = 0.8
    cooling_strategy: CoolingStrategy = CoolingStrategy.EXPONENTIAL
    alns_enable_tabu: bool = True
    alns_iters_per_offspring: int = 1
    enable_alns_improvement: bool = True
    enable_hv_early_stopping: bool = True
    hv_improvement_threshold: float = 1e-5
    hv_stagnation_limit: int = 50

    @classmethod
    def from_dict(cls, data: dict) -> "AlgorithmConfig":
        """从字典创建"""
        kwargs = {}

        # 处理简单字段
        simple_fields = [
            "makespan_priority",
            "score_priority",
            "num_epsilon_points",
            "time_limit",
            "big_m",
            "max_iter",
            "pop_size",
            "crossover_prob",
            "mutation_prob",
            "alns_max_iter",
            "alns_segment_size",
            "destroy_degree_min",
            "destroy_degree_max",
            "alns_initial_temp",
            "alns_cooling_rate",
            "alns_final_temp",
            "alns_tabu_tenure",
            "alns_enable_tabu",
            "alns_iters_per_offspring",
            "enable_alns_improvement",
            "enable_hv_early_stopping",
            "hv_improvement_threshold",
            "hv_stagnation_limit",
        ]
        for field_name in simple_fields:
            if field_name in data:
                kwargs[field_name] = data[field_name]

        # 处理枚举字段
        if "cooling_strategy" in data:
            kwargs["cooling_strategy"] = CoolingStrategy(data["cooling_strategy"])

        # 处理嵌套对象
        if "obj_config" in data:
            kwargs["obj_config"] = ObjConfig.from_dict(data["obj_config"])

        return cls(**kwargs)

    def __post_init__(self) -> None:
        if any(
            x <= 0
            for x in [
                self.max_iter,
                self.pop_size,
                self.num_epsilon_points,
                self.time_limit,
            ]
        ):
            raise ValueError("Iteration counts and sizes must be positive")
        if not 0 <= self.crossover_prob <= 1 or not 0 <= self.mutation_prob <= 1:
            raise ValueError("Probabilities must be in [0, 1]")
        if not 0 < self.destroy_degree_min < self.destroy_degree_max <= 1:
            raise ValueError("Invalid destroy degree range")
        if self.alns_initial_temp <= self.alns_final_temp:
            raise ValueError("Initial temperature must be > final temperature")
        if not 0 < self.alns_cooling_rate < 1:
            raise ValueError("Cooling rate must be in (0, 1)")


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
            f"Steiner点 = {self.instance_param.steiner_num}个, "
            f"地面车 = {self.instance_param.ground_veh_num}个, "
            f"无人机 = {self.instance_param.drone_num}个), "
            f"算法配置: 最大迭代 = {self.algorithm_config.max_iter}, "
            f"种群规模 = {self.algorithm_config.pop_size}, "
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
