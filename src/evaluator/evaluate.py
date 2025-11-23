from datetime import datetime
from typing import Dict, List, Set, Tuple

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
        self.min_comm_time = instance.prob_config.instance_param.min_visit_time

        # ---------- 节点集合 ----------
        self.id_base = list(instance.base_ids)
        self.id_demand = list(instance.demand_ids)
        self.accessible_flag = instance.accessible
        self.priority = instance.priority
        self.id_all_node = list(range(instance.total_node_num))
        self.id_node_no_base = self.id_demand

        self.dummy_end_id = instance.dummy_end_id

        # ---------- 车辆集合 ----------
        self.id_Ground = instance.ground_veh_ids
        self.id_Drone = instance.drone_ids
        self.id_vehicle = list(range(instance.total_veh_num))

        # ---------- 矩阵提取 ----------
        self.distance_matrix = instance.distance_matrix
        self.coverage_matrix = instance.comm_coverage_matrix
        self.travel_time_matrix = instance.travel_time_matrix

    def is_route_feasible(
        self,
        veh_id: int,
        candidate_route: List[int],
        candidate_coverage: Dict[Tuple[int, int], Set[int]],
    ) -> Tuple[bool, str]:
        """检查单一路径的可行性"""
        # --------------- 1. 判断路径是否完整 ---------------
        base_id = self.id_base[0]
        if candidate_route[0] != base_id or candidate_route[-1] != base_id:
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

        # --------------- 3. 判断任务是否符合要求 ---------------
        for key, value in candidate_coverage.items():
            visit_veh_id, visit_node = key
            covered_nodes = value
            # --------------- 3.1 判断无人机任务是否符合半径要求 ---------------
            if visit_veh_id in self.id_Drone:
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
            # --------------- 3.2 判断地面车辆任务是否符合分组要求 ---------------
            elif visit_veh_id in self.id_Ground:
                # 地面车辆覆盖的节点必须是同一组
                visit_node_id = self.instance.group_id[visit_node]  # type: ignore
                for covered_node in covered_nodes:
                    covered_group_id = self.instance.group_id[covered_node]  # type: ignore
                    if covered_group_id != visit_node_id:
                        infeasible_reason = (
                            f"地面车辆{visit_veh_id}在节点{visit_node}覆盖了不同组的节点"
                            f"{covered_node}（组{covered_group_id}），"
                            f"当前路径: {candidate_route}"
                        )
                        raise ValueError(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                            f"不可行信息: {infeasible_reason}"
                        )

        # --------------- 4. 判断是否超过当前车辆续航上限 ---------------
        # 计算路径长度
        route_length = self._calculate_route_distance(candidate_route)
        # 判断是否超过当前车辆能力上限
        if route_length > self.capacity[veh_id]:
            infeasible_reason = (
                f"路径长度{route_length}超过车辆{veh_id}能量容量{self.capacity[veh_id]}，"
                f"当前路径: {candidate_route}"
            )
            raise ValueError(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"不可行信息: {infeasible_reason}"
            )

        # --------------- 5. 判断任务是否超过最晚访问时间 ---------------
        route_duration = self._calculate_last_visit_finish_time(
            candidate_route,
            candidate_coverage,
            veh_id,
        )

        max_allowed_time = self.instance.max_visit_time[candidate_route[-2]]

        if route_duration > max_allowed_time:
            infeasible_reason = (
                f"路径耗时{route_duration}超过车辆{veh_id}最晚访问时间"
                f"{max_allowed_time}，当前路径: {candidate_route}"
            )
            return False, infeasible_reason

        # -------------- 6. 判断是否出现重复访问 -------------------
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
            if node in self.id_base:
                continue
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

    def _calculate_last_visit_finish_time(
        self,
        route: List[int],
        coverage: Dict[Tuple[int, int], Set[int]],
        veh_id: int,
    ) -> float:
        # 计算路径耗时
        from_nodes = route[:-2]
        to_nodes = route[1:-1]
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
            [综合得分]
        """

        # 计算优先级收益项
        priority_term = 0.0

        for coverage_key, covered_nodes in solution.coverage.items():
            veh_id, visit_node = coverage_key
            for covered_node in covered_nodes:
                priority_term += self.priority[covered_node]

        # 计算makespan扣除
        max_duration = 0.0
        for veh_id, route in enumerate(solution.routes):
            route_coverage: Dict[Tuple[int, int], Set[int]] = dict()
            for key, covered_nodes in solution.coverage.items():
                c_veh_id, visit_node = key
                if c_veh_id == veh_id:
                    route_coverage[key] = covered_nodes
            route_duration = self._calculate_route_duration(
                route,
                route_coverage,
                veh_id,
            )
            if route_duration > max_duration:
                max_duration = route_duration

        final_score = priority_term - 1e-2 * max_duration

        return [final_score]
