from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np

from src.element import Solution
from src.element.instance_class import InstanceClass


class Evaluator:
    """
    外置的评估器,负责可行性检查和目标函数计算
    避免在Solution中存储距离矩阵等大量数据
    """

    def __init__(
        self,
        instance: "InstanceClass",
    ):
        """
        Args:
        """
        self.instance = instance
        self.capacity = instance.capacity_array
        self.comm_coverage_matrix = instance.comm_coverage_matrix
        self.demand_num = instance.demand_num
        self.M = instance.prob_config.algorithm_config.big_m
        self.min_comm_time = instance.prob_config.instance_param.min_comm_time

        # ---------- 节点集合 ----------
        self.id_base = list(instance.base_ids)
        self.id_demand = list(instance.demand_ids)
        self.accessible_flag = instance.accessible
        self.priority = instance.priority
        self.id_steiner = list(instance.steiner_ids)
        self.id_all_node = list(range(instance.total_node_num))
        self.id_node_no_base = self.id_demand + self.id_steiner

        # ---------- 车辆集合 ----------
        self.id_Ground = instance.ground_veh_ids
        self.id_Drone = instance.drone_ids
        self.id_vehicle = list(range(instance.total_veh_num))

        # ---------- 矩阵提取 ----------
        self.distance_matrix = instance.distance_matrix
        self.coverage_matrix = instance.comm_coverage_matrix
        self.travel_time_matrix = instance.travel_time_matrix
        self.travel_cost_matrix = instance.travel_consum_matrix
        self.comm_cost_matrix = instance.comm_consum_matrix

        # ---------- 设置目标函数信息 ----------
        self.obj_config = instance.prob_config.algorithm_config.obj_config
        obj_names = self.obj_config.name
        obj_sense = self.obj_config.sense
        self.objective_info = list(zip(obj_names, obj_sense))

    def is_route_feasible(
        self,
        veh_id: int,
        candidate_route: List[int],
        candidate_coverage: dict[Tuple[int, int], set],
    ) -> Tuple[bool, str]:
        """检查单一路径的可行性"""
        # --------------- 1. 判断路径是否完整 ---------------
        if candidate_route[0] != 0 or candidate_route[-1] != 0:
            infeasible_reason = (
                f"路径必须以仓库节点开始和结束，当前路径: {candidate_route}"
            )
            return False, infeasible_reason

        # --------------- 2. 判断是否存在车辆访问不可达点 ---------------
        accessible_flag = self.accessible_flag
        if veh_id in self.id_Ground:
            for node in candidate_route:
                if accessible_flag[node] == 0:
                    infeasible_reason = f"地面车辆{veh_id}访问了不可达节点{node}，当前路径: {candidate_route}"
                    return False, infeasible_reason

        # --------------- 3. 判断是否超过车辆覆盖半径 ---------------
        for key, value in candidate_coverage.items():
            visit_veh_id, visit_node = key
            covered_nodes = value
            for covered_node in covered_nodes:
                coverable_flag = self.coverage_matrix[
                    visit_veh_id, visit_node, covered_node
                ]
                if not coverable_flag:
                    infeasible_reason = (
                        f"车辆{visit_veh_id}在节点{visit_node}"
                        f"覆盖了超出覆盖半径的节点{covered_node}，"
                        f"当前路径: {candidate_route}"
                    )
                    # debug:
                    raise ValueError(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                        f"不可行信息: {infeasible_reason}"
                    )

                    return False, infeasible_reason

        # --------------- 4. 判断是否超过当前车辆能耗上限 ---------------
        # 计算路径耗能
        route_energy_consum = self._calculate_route_energy_consumption(
            candidate_route,
            candidate_coverage,
            veh_id,
        )
        if route_energy_consum > self.capacity[veh_id]:
            infeasible_reason = (
                f"路径能耗{route_energy_consum:.2f}"
                f"超过车辆{veh_id}容量{self.capacity[veh_id]:.2f}，"
                f"当前路径: {candidate_route}"
            )
            return False, infeasible_reason

        # -------------- 5. 判断是否出现重复访问 -------------------
        visited_nodes = set()
        for node in candidate_route:
            if node in visited_nodes and node != 0:
                infeasible_reason = (
                    f"路径中出现重复访问节点{node}，" f"当前路径: {candidate_route}"
                )
                raise ValueError(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                    f"不可行信息: {infeasible_reason}"
                )
            visited_nodes.add(node)

        # --------------- 6. 判断是否出现访问不登记 --------------
        for node in candidate_route:
            if node in self.id_demand:
                if (veh_id, node) not in candidate_coverage:
                    infeasible_reason = (
                        f"路径中访问节点{node}未登记覆盖情况，"
                        f"当前路径: {candidate_route}"
                    )
                    raise ValueError(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                        f"Debug Info: 访问节点{node}未在coverage中找到记录"
                    )
                else:
                    covered_nodes = candidate_coverage[(veh_id, node)]
                    if node not in covered_nodes:
                        infeasible_reason = (
                            f"路径中访问节点{node}未覆盖自身，\n"
                            f"当前路径: {candidate_route}, \n"
                            f"覆盖节点: {covered_nodes}"
                        )
                        raise ValueError(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                            f"不可行信息: {infeasible_reason}"
                        )
            elif node in self.id_steiner:
                if (veh_id, node) in candidate_coverage:
                    covered_nodes = candidate_coverage[(veh_id, node)]
                    if node in covered_nodes:
                        infeasible_reason = (
                            f"路径中访问节点{node}为中继节点，"
                            f"不应覆盖自身，\n"
                            f"当前路径: {candidate_route}, \n"
                            f"覆盖节点: {covered_nodes}"
                        )
                        raise ValueError(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                            f"不可行信息: {infeasible_reason}"
                        )
                    elif not covered_nodes:
                        infeasible_reason = (
                            f"路径中访问节点{node}为中继节点，"
                            f"应至少覆盖一个需求节点，\n"
                            f"当前路径: {candidate_route}, \n"
                            f"覆盖节点: {covered_nodes}"
                        )
                        raise ValueError(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                            f"不可行信息: {infeasible_reason}"
                        )

        return True, "当前路径可行"

    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算单一路径的距离"""
        if len(route) < 2:
            return 0.0

        # 使用NumPy的高级索引,一次性获取所有边的距离
        from_nodes = route[:-1]
        to_nodes = route[1:]
        total_distance = self.distance_matrix[from_nodes, to_nodes].sum()

        return float(total_distance)

    def _calculate_route_duration(
        self,
        route: List[int],
        coverage: Dict[Tuple[int, int], Set[int]],
        veh_id: int,
    ) -> float:
        """计算车辆路径的部分耗时组成。

        该方法计算车辆完成其路径所需的总时间，
        并将其分解为行驶时间和通信时间两个组成部分。

        Args:
            route: 表示车辆路径的节点ID有序列表
            coverage: 将(vehicle_id, node_id)元组映射到每个位置覆盖的节点集合的字典
            veh_id: 车辆的唯一标识符

        Returns:
            包含以下三个元素的元组:
                - total_duration: 路径总耗时(行驶 + 通信)
                - total_travel_time: 节点间行驶耗时
                - communication_time: 通信活动耗时
        """

        # 计算路径耗时
        from_nodes = route[:-1]
        to_nodes = route[1:]
        travel_times = self.travel_time_matrix[veh_id, from_nodes, to_nodes]
        total_travel_time = float(travel_times.sum())

        # 计算通信耗时
        min_comm_time = self.min_comm_time
        communication_count = 0
        for visit_node in route:
            covered_nodes = coverage.get((veh_id, visit_node), set())
            if covered_nodes:
                communication_count += 1
        communication_time = communication_count * min_comm_time

        # 计算总耗时
        total_duration = total_travel_time + communication_time

        return total_duration

    def _calculate_route_energy_consumption(
        self,
        route: List[int],
        coverage: Dict[Tuple[int, int], Set[int]],
        veh_id: int,
    ) -> float:
        """计算单一路径的能耗"""

        # 计算路径耗能
        from_nodes = route[:-1]
        to_nodes = route[1:]
        travel_consum = self.travel_cost_matrix[veh_id, from_nodes, to_nodes]
        travel_energy_consum = float(travel_consum.sum())

        # 计算通信耗时
        min_com_time = self.instance.prob_config.instance_param.min_comm_time
        communication_count = 0
        for visit_node in route:
            covered_nodes = coverage.get((veh_id, visit_node), set())
            if covered_nodes:
                communication_count += 1
        communication_time = communication_count * min_com_time

        # 提取通信耗能率
        vehicle_config = self.instance.prob_config.instance_param.vehicle_config
        if veh_id in self.id_Ground:
            comm_energy_rate = vehicle_config.ground_veh_comm_power_rate
        elif veh_id in self.id_Drone:
            comm_energy_rate = vehicle_config.drone_comm_power_rate
        # 计算通信耗能
        communication_energy_consum = communication_time * comm_energy_rate

        # 计算总耗时
        total_energy_consum = travel_energy_consum + communication_energy_consum

        return total_energy_consum

    def sol_feasible(self, solution: Solution) -> bool:
        """检查解的可行性"""

        # 检查是否出现重复覆盖
        sol_coverage: Set[int] = set()

        all_route_coverage: Dict[int, Dict[Tuple[int, int], Set[int]]] = dict()
        for key, covered_nodes in solution.coverage.items():
            veh_id, visit_node = key
            if veh_id not in all_route_coverage:
                all_route_coverage[veh_id] = dict()
            all_route_coverage[veh_id][(veh_id, visit_node)] = covered_nodes
            if covered_nodes & sol_coverage:
                raise ValueError(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                    f"解不可行: 覆盖节点重复覆盖，车辆{veh_id}访问节点{visit_node}覆盖节点"
                    f"{covered_nodes & sol_coverage}已被其他车辆覆盖"
                )
            sol_coverage.update(covered_nodes)

        visit_set: Set[int] = set()
        for route in solution.routes:
            if len(route) > 2:
                for node in route:
                    if node == 0:
                        continue
                    if node in visit_set:
                        raise ValueError(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                            f"解不可行: 访问节点重复访问，节点{node}被多次访问"
                        )
                    else:
                        visit_set.add(node)

        # debug: 检查每条路径的可行性
        for veh_id, route in enumerate(solution.routes):
            if veh_id in all_route_coverage:
                route_coverage = all_route_coverage[veh_id]
                feasible_flag, reason = self.is_route_feasible(
                    veh_id,
                    route,
                    route_coverage,
                )
                if not feasible_flag:
                    print(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                        f"车辆{veh_id}路径不可行，原因: {reason}"
                    )
                    return False
            else:
                if len(route) > 2:
                    raise ValueError(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                        f"解不可行: 车辆{veh_id}存在路径但无覆盖记录，"
                        f"当前路径: {route}"
                    )

        return True

    def sol_evaluate(self, solution: Solution) -> Tuple[List[float], bool]:
        """评估解的目标函数值"""
        # 兼容性补偿，适配之前mip的逻辑
        for route_idx, route in enumerate(solution.routes):
            if len(route) == 0:
                solution.routes[route_idx] = [0, 0]

        # [X] 可行性检查
        feasible_flag = self.sol_feasible(solution)
        if not feasible_flag:
            solution.objectives = [
                float("inf"),
                0.0,
            ]
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"解不可行，设置目标值为{solution.objectives}"
            )
            return solution.objectives, feasible_flag

        # [X] 计算目标函数值
        solution.objectives = self.calculate_objectives(solution)
        solution.status = "已评估，解可行"

        return solution.objectives, feasible_flag

    def calculate_objectives(self, solution: Solution) -> List[float]:
        """
        计算多目标函数值

        Returns:
            [总距离, 最大路径长度, 车辆数]
        """
        # -------------------- 计算makespan归一化参数 --------------------
        # 计算归一化因子（总旅行时间部分）
        feasible_travel_time = np.where(
            self.travel_time_matrix < self.M,
            self.travel_time_matrix,
            -np.inf,
        )

        # 理论总旅行时间上界为各个需求点均从基地往返
        # 注意：这里计算时候是每个车都计算了一遍，最后求平均
        _total_travel_time_norm_factor = sum(
            2
            * max(
                feasible_travel_time[k, i, j]
                for i in self.id_base
                for k in self.id_vehicle
            )  # Max travel time from any base to j for vehicle k
            for j in self.id_demand
        )

        # 计算归一化因子（总通信时间部分）
        min_comm_time = self.min_comm_time
        _total_comm_time_norm_factor = len(self.id_demand) * min_comm_time

        # 计算归一化因子，平均旅行时间+ 全覆盖最短通信时间
        # 归一化因子，平均旅行时间+平均全覆盖最短通信时间
        total_time_norm_factor = (
            _total_travel_time_norm_factor + _total_comm_time_norm_factor
        ) / len(self.id_vehicle)

        # -------------------- 计算目标函数值 (makespan) --------------------
        # 迭代求解各路径makespan
        max_duration = 0.0
        for veh_id, route in enumerate(solution.routes):
            route_coverage = {
                key: value
                for key, value in solution.coverage.items()
                if key[0] == veh_id
            }
            route_duration = self._calculate_route_duration(
                route,
                route_coverage,
                veh_id,
            )
            if route_duration > max_duration:
                max_duration = route_duration

        # 归一化处理
        normalized_makespan = float(max_duration / total_time_norm_factor)

        # -------------------- 计算收益归一化参数 --------------------
        # 理论最大收益为所有需求点均被覆盖
        total_priority_norm_factor = sum(self.priority)

        # -------------------- 计算总收益 --------------------
        covered_priority = 0
        covered_demands = set()
        for key, covered_nodes in solution.coverage.items():
            for node in covered_nodes:
                covered_demands.add(node)
        for demand_node in covered_demands:
            covered_priority += self.priority[demand_node]

        # 归一化处理
        normalized_priority_score = float(covered_priority / total_priority_norm_factor)

        return [normalized_makespan, normalized_priority_score]

    def calculate_energy_margin(
        self,
        veh_id: int,
        route: List[int],
        coverage: Dict[Tuple[int, int], Set[int]],
    ) -> float:
        """
        计算路径的能量余量

        Args:
            veh_id: 车辆ID
            route: 当前路径
            coverage: 覆盖信息

        Returns:
            float: 可用能量余量 = 车辆容量 - 当前路径能耗
        """
        current_energy = self._calculate_route_energy_consumption(
            route, coverage, veh_id
        )
        return float(self.capacity[veh_id] - current_energy)

    def calculate_insertion_energy_increase(
        self,
        veh_id: int,
        route: List[int],
        insert_pos: int,
        insert_node: int,
        new_covered_tasks: Set[int],
        current_coverage: Dict[Tuple[int, int], Set[int]],
    ) -> float:
        """
        计算插入节点导致的能耗增量

        Args:
            veh_id: 车辆ID
            route: 当前路径
            insert_pos: 插入位置
            insert_node: 待插入节点
            new_covered_tasks: 新覆盖的任务集合
            current_coverage: 当前覆盖信息

        Returns:
            float: 能耗增量
        """
        # 1. 计算旅行能耗增量
        prev_node = route[insert_pos - 1]
        next_node = route[insert_pos]

        # 原路径: prev -> next
        old_travel_energy = self.travel_cost_matrix[veh_id, prev_node, next_node]

        # 新路径: prev -> insert -> next
        new_travel_energy = (
            self.travel_cost_matrix[veh_id, prev_node, insert_node]
            + self.travel_cost_matrix[veh_id, insert_node, next_node]
        )

        travel_energy_increase = new_travel_energy - old_travel_energy

        # 2. 计算通信能耗增量
        comm_energy_increase = 0.0
        if new_covered_tasks:
            # 提取通信耗能率
            vehicle_config = self.instance.prob_config.instance_param.vehicle_config
            if veh_id in self.id_Ground:
                comm_energy_rate = vehicle_config.ground_veh_comm_power_rate
            elif veh_id in self.id_Drone:
                comm_energy_rate = vehicle_config.drone_comm_power_rate

            min_comm_time = self.min_comm_time
            comm_energy_increase = min_comm_time * comm_energy_rate

        return float(travel_energy_increase + comm_energy_increase)
