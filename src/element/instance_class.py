from __future__ import annotations  # 必须在文件的第一行

from dataclasses import dataclass

import numpy as np

from src.config.prob_config import Config


@dataclass
class InstanceClass:
    """表示算例的类,包含算例的各种参数和配置信息"""

    prob_config: Config

    # 算例基本信息
    name: str

    # 算例节点数和id列表
    base_num: int
    demand_num: int
    ground_veh_num: int
    drone_num: int

    base_ids: list[int]
    demand_ids: list[int]
    ground_veh_ids: list[int]
    drone_ids: list[int]

    # 算例节点属性
    coords: list[tuple[float, float]]  # 统一存储坐标 [(x1, y1), (x2, y2), ...]
    min_com_time: list[float]
    max_visit_time: list[float]  # 各节点的最晚访问时间
    priority: list[int]
    accessible: list[int]
    # 分组信息（基地点为-1，需求点为分组ID）
    group_id: list[int] | None = None

    def __post_init__(self):
        for field_name, field_value in self.__dataclass_fields__.items():
            if isinstance(getattr(self, field_name), list):
                setattr(self, field_name, tuple(getattr(self, field_name)))
        if self.name.endswith(".tsp"):
            self.name = "TSPLIB-" + self.name[:-4]

        # 计算总数，方便调用
        self.total_node_num = self.base_num + self.demand_num
        self.total_veh_num = self.ground_veh_num + self.drone_num

        # 创建虚拟终点ID（在所有真实节点之后，每次自动生成）
        # 虚拟终点用于建模路径结束，ID为 total_node_num
        self.dummy_end_id = self.total_node_num
        # 扩展节点总数（包含虚拟终点）
        self.total_node_num_with_dummy = self.total_node_num + 1

        # 初始化分组信息
        if self.group_id is None:
            self.group_id = [-1] * self.base_num + [0] * self.demand_num

        # 构建 group_id 到节点集合的映射
        self.group_sets: dict[int, set[int]] = {}
        for node_id, gid in enumerate(self.group_id):
            if gid >= 0:  # 只包括需求点，不包括基地（gid=-1）
                if gid not in self.group_sets:
                    self.group_sets[gid] = set()
                self.group_sets[gid].add(node_id)

        # 将虚拟终点的坐标添加到coords中（与基地坐标相同，表示返回基地）
        base_coord = (
            self.coords[self.base_ids[0]] if len(self.base_ids) > 0 else (0.0, 0.0)
        )
        # coords已经是tuple类型，直接拼接
        coords_list = list(self.coords) + [base_coord]
        self.coords_with_dummy = tuple(coords_list)

        # 预处理:转换为numpy数组,便于计算
        self.np_coords: np.ndarray = np.array(self.coords)
        self.np_coords_with_dummy: np.ndarray = np.array(self.coords_with_dummy)

        # 预处理：计算容量向量
        vehicle_config = self.prob_config.instance_param.vehicle_config
        ground_capacity = vehicle_config.ground_veh_range
        drone_capacity = vehicle_config.drone_range
        self.capacity_array: np.ndarray = np.array(
            [ground_capacity] * self.ground_veh_num + [drone_capacity] * self.drone_num
        )
        ground_radius = 0
        drone_radius = vehicle_config.drone_comm_radius
        self.radius_array: np.ndarray = np.array(
            [ground_radius] * self.ground_veh_num + [drone_radius] * self.drone_num
        )

        # 预处理:计算距离矩阵、覆盖矩阵、行驶时间矩阵
        self.distance_matrix = compute_distance_matrix(self)
        self.comm_coverage_matrix = compute_coverage_matrix(self)
        self.travel_time_matrix = compute_travel_time_matrix(self)

    def __str__(self) -> str:
        str_info = (
            f"算例名称: {self.name} "
            f"(需求点 = {self.demand_num}个, "
            f"地面车 = {self.ground_veh_num}个, "
            f"无人机 = {self.drone_num}个)"
        )
        return str_info

    __repr__ = __str__

    def to_dict(self) -> dict:
        """将实例对象转换为字典形式,便于存储和传输"""
        return {
            "name": self.name,
            "base_num": self.base_num,
            "demand_num": self.demand_num,
            "ground_veh_num": self.ground_veh_num,
            "drone_num": self.drone_num,
            "base_ids": self.base_ids,
            "demand_ids": self.demand_ids,
            "ground_veh_ids": self.ground_veh_ids,
            "drone_ids": self.drone_ids,
            "coords": self.coords,
            "min_com_time": self.min_com_time,
            "max_visit_time": self.max_visit_time,
            "priority": self.priority,
            "accessible": self.accessible,
            "group_id": self.group_id,
            "prob_config": self.prob_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InstanceClass":
        """从字典形式创建实例对象"""
        # 兼容旧格式:如果存在x_coords和y_coords,转换为coords
        if "coords" in data:
            coords = [tuple(coord) for coord in data["coords"]]
        elif "x_coords" in data and "y_coords" in data:
            coords = list(zip(data["x_coords"], data["y_coords"]))
        else:
            coords = []

        return cls(
            name=data["name"],
            base_num=data["base_num"],
            demand_num=data["demand_num"],
            ground_veh_num=data["ground_veh_num"],
            drone_num=data["drone_num"],
            base_ids=list(data["base_ids"]),
            demand_ids=list(data["demand_ids"]),
            ground_veh_ids=list(data["ground_veh_ids"]),
            drone_ids=list(data["drone_ids"]),
            coords=coords,
            min_com_time=list(data["min_com_time"]),
            max_visit_time=list(data["max_visit_time"]),
            priority=list(data["priority"]),
            accessible=list(data["accessible"]),
            group_id=(
                list(data["group_id"])
                if "group_id" in data and data["group_id"]
                else None
            ),
            prob_config=Config.from_dict(data["prob_config"]),
        )

    @classmethod
    def new_instance(
        cls,
        prob_config: "Config",
        coords: list[tuple[float, float]],
        priority: list[int],
        accessible: list[int],
        group_id: list[int] | None = None,
    ) -> "InstanceClass":
        """根据配置对象创建算例实例对象(占位符方法,需根据实际需求实现)"""
        # [ ] 根据prob_config中的信息,生成对应的InstanceClass对象
        name = prob_config.instance_param.name
        base_num = prob_config.instance_param.base_num
        demand_num = prob_config.instance_param.demand_num
        ground_veh_num = prob_config.instance_param.ground_veh_num
        drone_num = prob_config.instance_param.drone_num

        base_ids, demand_ids = allocate_ids_np(0, base_num, demand_num)
        ground_veh_ids, drone_ids = allocate_ids_np(0, ground_veh_num, drone_num)

        com_time = prob_config.instance_param.min_visit_time

        min_com_time = [0.0] * base_num + [com_time] * demand_num

        # 生成最晚访问时间
        max_visit_time_range = prob_config.instance_param.max_visit_time_range
        if max_visit_time_range is not None:
            # 使用区间随机生成
            import numpy as np

            max_visit_time = [0.0] * base_num + np.random.uniform(
                max_visit_time_range[0], max_visit_time_range[1], demand_num
            ).tolist()
        else:
            # 使用统一值
            default_max_time = prob_config.instance_param.max_visit_time
            max_visit_time = [0.0] * base_num + [default_max_time] * demand_num

        return cls(
            name=name,
            base_num=base_num,
            demand_num=demand_num,
            ground_veh_num=ground_veh_num,
            drone_num=drone_num,
            base_ids=base_ids,
            demand_ids=demand_ids,
            ground_veh_ids=ground_veh_ids,
            drone_ids=drone_ids,
            coords=coords,
            min_com_time=min_com_time,
            max_visit_time=max_visit_time,
            priority=priority,
            accessible=accessible,
            group_id=group_id,
            prob_config=prob_config,
        )


def allocate_ids_np(start, *lengths):
    a = np.arange(start, start + sum(lengths))
    cs = np.cumsum((0,) + lengths)  # 分段点
    return [a[cs[i] : cs[i + 1]].tolist() for i in range(len(lengths))]


# ============ 计算工具类（专注计算逻辑）============


def compute_distance_matrix(instance: InstanceClass) -> np.ndarray:
    """计算距离矩阵（扩展维度包含虚拟终点）"""
    # 使用包含虚拟终点的坐标（虚拟终点坐标与基地相同）
    np_coords_with_dummy = instance.np_coords_with_dummy
    # 使用广播机制一次性计算所有距离
    diff = (
        np_coords_with_dummy[:, np.newaxis, :] - np_coords_with_dummy[np.newaxis, :, :]
    )
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    extended_matrix: np.ndarray = np.round(dist_matrix, 2)

    # 从虚拟终点出发的距离设为大M（表示不应该从虚拟终点出发）
    big_M = instance.prob_config.algorithm_config.big_m
    n = instance.total_node_num
    extended_matrix[n, :] = big_M
    extended_matrix[0, n] = big_M  # 虚拟终点到基地距离为0

    return extended_matrix


def compute_coverage_matrix(instance: InstanceClass):
    """计算覆盖关系矩阵

    coverage_matrix[v, i, j] 表示:
    - 车辆v在节点i(demand)时,能否覆盖节点j(仅demand)
    - i 可以是 demand_ids 中的点(执行覆盖的位置)
    - j 只能是 demand_ids 中的点(被覆盖的目标)

    覆盖逻辑:
    - 地面车辆: 在节点i时,可以覆盖节点i本身以及与i同组的所有需求点
    - 无人机: 在节点i时,可以覆盖通信半径范围内的所有需求点

    使用示例:
    - 查询车辆v在节点i可以覆盖的所有需求节点ID:
        covered_demands = [j for j in instance.demand_ids if coverage_matrix[v, i, j]]
        或使用numpy:
        covered_demands = instance.demand_ids[coverage_matrix[v, i, instance.demand_ids]]

    - 查询车辆v在节点i是否能覆盖需求节点j:
        can_cover = coverage_matrix[v, i, j]
    """
    config: Config = instance.prob_config
    total_veh_num = instance.total_veh_num
    total_node_num = instance.total_node_num
    demand_ids = instance.demand_ids
    group_id = instance.group_id

    if instance.distance_matrix is None:
        raise ValueError("距离矩阵 (Distance matrix)尚未计算!")

    # 扩展矩阵维度包含虚拟终点
    coverage_matrix = np.zeros(
        (total_veh_num, total_node_num + 1, total_node_num + 1), dtype=bool
    )

    for v in range(total_veh_num):
        if v in instance.ground_veh_ids:
            # 地面车辆: 覆盖访问点及其同组节点
            if group_id is None:
                # 如果没有分组信息，则只能覆盖自身
                for i in demand_ids:
                    coverage_matrix[v, i, i] = True
            else:
                for i in demand_ids:
                    group_i = group_id[i]
                    for j in demand_ids:
                        # 覆盖自身或同组节点
                        if i == j or (group_i >= 0 and group_id[j] == group_i):
                            coverage_matrix[v, i, j] = True

        elif v in instance.drone_ids:
            # 无人机: 按通信半径覆盖
            cover_radius = config.instance_param.vehicle_config.drone_comm_radius
            for i in demand_ids:
                for j in demand_ids:
                    coverage_matrix[v, i, j] = (
                        instance.distance_matrix[i, j] < cover_radius
                    )
        else:
            raise ValueError(f"未知车辆ID: {v}")

    return coverage_matrix


def compute_travel_time_matrix(instance: InstanceClass) -> np.ndarray:
    """计算行驶时间矩阵（NumPy优化版本）"""
    total_veh_num = instance.total_veh_num
    distance_matrix = instance.distance_matrix
    accessible = instance.accessible
    ground_veh_ids = instance.ground_veh_ids
    drone_ids = instance.drone_ids

    big_M = instance.prob_config.algorithm_config.big_m

    ground_veh_speed = (
        instance.prob_config.instance_param.vehicle_config.ground_veh_speed
    )
    drone_speed = instance.prob_config.instance_param.vehicle_config.drone_speed

    # 确保 accessible 是 NumPy 数组
    accessible_array = np.array(accessible)

    # 初始化 travel_time_matrix为0（扩展维度包含虚拟终点）
    n = instance.total_node_num
    travel_time_matrix = np.zeros((total_veh_num, n + 1, n + 1))

    # 分别处理地面车辆和无人机（只计算真实节点部分）
    for v in range(total_veh_num):
        if v in ground_veh_ids:
            speed = ground_veh_speed
            if speed > 0:
                # 计算基础时间（包括到虚拟终点的时间）
                travel_time_matrix[v, :, :] = distance_matrix[:, :] / speed
                # 创建不可达掩码（向量化操作，只对真实节点）
                inaccessible_i = accessible_array == 0  # shape: (n,)
                # 弧线任意一端不可达，则时间设为无穷大
                inaccessible_mask = inaccessible_i[:, None] | inaccessible_i[None, :]
                travel_time_matrix[v, :n, :n][inaccessible_mask] = big_M

                # 对于不可达的需求节点，到虚拟终点的时间也设为大M
                for i in instance.demand_ids:
                    if accessible_array[i] == 0:
                        travel_time_matrix[v, i, n] = big_M

                # 从虚拟终点出发的时间设为大M（表示不应该从虚拟终点出发）
                travel_time_matrix[v, n, :] = big_M
            else:
                travel_time_matrix[v, :, :] = big_M

        elif v in drone_ids:
            speed = drone_speed
            if speed > 0:
                # 计算基础时间（包括到虚拟终点的时间）
                travel_time_matrix[v, :, :] = distance_matrix[:, :] / speed
                # 从虚拟终点出发的时间设为大M（表示不应该从虚拟终点出发）
                travel_time_matrix[v, n, :] = big_M
            else:
                travel_time_matrix[v, :, :] = big_M
        else:
            raise ValueError(f"未知车辆ID: {v}")
    return np.round(travel_time_matrix, 2)


def compute_travel_consum_matrix(instance: InstanceClass) -> np.ndarray:
    """计算行驶能耗矩阵"""
    total_veh_num = instance.total_veh_num
    travel_time_matrix = instance.travel_time_matrix
    ground_veh_ids = np.array(instance.ground_veh_ids)
    drone_ids = np.array(instance.drone_ids)

    vehicle_config = instance.prob_config.instance_param.vehicle_config
    ground_veh_energy_rate = vehicle_config.ground_veh_travel_power_rate
    drone_energy_rate = vehicle_config.drone_travel_power_rate

    # 创建能耗率向量 shape: (total_veh_num,)
    energy_rate_vector = np.zeros(total_veh_num)
    energy_rate_vector[ground_veh_ids] = ground_veh_energy_rate
    energy_rate_vector[drone_ids] = drone_energy_rate

    # 利用广播计算能耗矩阵
    # energy_rate_vector[:, np.newaxis, np.newaxis] 的形状: (total_veh_num, 1, 1)
    # travel_time_matrix 的形状: (total_veh_num, total_node_num, total_node_num)
    # 相乘后的形状: (total_veh_num, total_node_num, total_node_num)
    travel_consum_matrix = (
        travel_time_matrix * energy_rate_vector[:, np.newaxis, np.newaxis]
    )

    return travel_consum_matrix


def compute_comm_consum_matrix(instance: InstanceClass) -> np.ndarray:
    """计算通信能耗矩阵"""
    total_veh_num = instance.total_veh_num
    min_com_time = np.array(instance.min_com_time)
    ground_veh_ids = np.array(instance.ground_veh_ids)
    drone_ids = np.array(instance.drone_ids)

    vehicle_config = instance.prob_config.instance_param.vehicle_config
    ground_veh_comm_power_rate = vehicle_config.ground_veh_comm_power_rate
    drone_comm_power_rate = vehicle_config.drone_comm_power_rate

    # 创建能耗率向量 shape: (total_veh_num,)
    comm_power_rate_vector = np.zeros(total_veh_num)
    comm_power_rate_vector[ground_veh_ids] = ground_veh_comm_power_rate
    comm_power_rate_vector[drone_ids] = drone_comm_power_rate

    # 利用广播计算能耗矩阵
    # comm_power_rate_vector[:, np.newaxis] 的形状: (total_veh_num, 1)
    # min_com_time 的形状: (total_node_num,)
    # 相乘后的形状: (total_veh_num, total_node_num)
    comm_consum_matrix: np.ndarray = (
        comm_power_rate_vector[:, np.newaxis] * min_com_time
    )

    return comm_consum_matrix
