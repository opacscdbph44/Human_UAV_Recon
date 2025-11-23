import random
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple

from src.element import InstanceClass, Solution
from src.evaluator import Evaluator
from src.solver.alns_history import ALNSHistory
from src.solver.consr_heuristic import randomized_greedy_heuristic


class ALNS:
    """自适应大邻域框架"""

    def __init__(
        self,
        instance: InstanceClass,
    ):
        # ---------- 控制开关 ----------
        self.enable_debug = True

        # ---------- 提取必要参数 ----------
        self.instance = instance
        self.comm_coverage_matrix = instance.comm_coverage_matrix
        self.priority_list = instance.priority
        self.travel_time_matrix = instance.travel_time_matrix
        accessible_flag = instance.accessible
        self.id_base = list(instance.base_ids)
        self.id_demand = list(instance.demand_ids)
        self.demand_num = len(self.id_demand)
        self.id_isolated = [i for i in self.id_demand if not accessible_flag[i]]
        self.id_connected = [i for i in self.id_demand if accessible_flag[i]]
        self.min_comm_time = instance.prob_config.instance_param.min_visit_time

        vehicle_config = self.instance.prob_config.instance_param.vehicle_config
        ground_veh_speed = vehicle_config.ground_veh_speed
        drone_speed = vehicle_config.drone_speed
        self.veh_speeds = [ground_veh_speed] * instance.ground_veh_num + [
            drone_speed
        ] * instance.drone_num

        # ---------- 车辆集合 ----------
        self.id_Ground = instance.ground_veh_ids
        self.id_Drone = instance.drone_ids
        self.id_vehicle = list(range(instance.total_veh_num))

        # ---------- 矩阵提取 ----------
        self.distance_matrix = instance.distance_matrix
        self.coverage_matrix = instance.comm_coverage_matrix
        self.travel_time_matrix = instance.travel_time_matrix

        # ---------- 初始化评价器 ----------
        self.evaluator = Evaluator(instance)

        # ---------- 提取算法参数 ----------

        algorithm_param = instance.prob_config.algorithm_config
        # 最大迭代次数
        self.max_iter = algorithm_param.max_iter

        # 允许早停
        self.enable_early_stop = False
        # 早停相关参数
        self.max_not_improve_iter = algorithm_param.max_not_improve_iter
        self.early_stop_threshold = algorithm_param.early_stop_threshold

        # 破坏率范围
        self.destroy_degree_min = algorithm_param.destroy_degree_min
        self.destroy_degree_max = algorithm_param.destroy_degree_max

        # 遗忘率
        self.decay_rate = algorithm_param.decay_rate

        # ---------- 破坏操作算子 ----------
        self.destroy_operators = [
            "random_removal",
            "worst_removal",
        ]

        # ---------- 修复操作算子 ----------
        self.repair_operators = [
            "greedy_insertion",
            "random_insertion",
        ]

        # ---------- 初始化参数 ----------
        # 初始化早停计数器
        self.not_improve_counter: int = 0
        # 最优目标函数值
        self.best_objective: float = 0.0
        # 历代情况记录器
        self.history = ALNSHistory()

    def _init_solution(self) -> Solution:
        """初始化一个解"""
        initial_solution = Solution.new_solution(
            instance=self.instance,
            solver_name="ALNS",
        )
        self.current_sol = randomized_greedy_heuristic(
            initial_solution,
            self.instance,
            success_rate=1.0,
        )

        self.best_sol = self.current_sol.copy()

        self.best_objective = self.best_sol.objectives[0]

        # 记录初始解
        self.history.record_initial(
            best_objective=self.best_objective,
            current_objective=self.current_sol.objectives[0],
        )

        return initial_solution

    def _init_weights(self):
        """初始化操作算子权重和得分记录"""
        # 初始化权重字典(用于记录)
        self.destroy_weights = {op: 1.0 for op in self.destroy_operators}
        self.repair_weights = {op: 1.0 for op in self.repair_operators}

        # 初始化得分数组(用于轮盘赌选择)
        self.destroy_scores = {op: 1.0 for op in self.destroy_operators}
        self.repair_scores = {op: 1.0 for op in self.repair_operators}

        # 记录本次迭代各算子的使用情况和得分
        self.iteration_operator_stats = {
            "destroy": {op: [] for op in self.destroy_operators},
            "repair": {op: [] for op in self.repair_operators},
        }

    def _select_destroy_op(self) -> str:
        """基于轮盘赌选择破坏算子"""
        # 获取所有算子和对应得分
        operators = list(self.destroy_scores.keys())
        scores = list(self.destroy_scores.values())

        # 计算总得分
        total_score = sum(scores)

        # 计算选择概率
        probabilities = [score / total_score for score in scores]

        # 轮盘赌选择
        selected_op = random.choices(operators, weights=probabilities, k=1)[0]

        return selected_op

    def _select_repair_op(self) -> str:
        """基于轮盘赌选择修复算子"""
        # 获取所有算子和对应得分
        operators = list(self.repair_scores.keys())
        scores = list(self.repair_scores.values())

        # 计算总得分
        total_score = sum(scores)

        # 计算选择概率
        probabilities = [score / total_score for score in scores]

        # 轮盘赌选择
        selected_op = random.choices(operators, weights=probabilities, k=1)[0]

        return selected_op

    def _apply_destroy_operator(
        self, operator: str, solution: Solution
    ) -> Tuple["Solution", List[int]]:
        """应用破坏算子"""
        # 随机选择破坏率
        destroy_rate = random.uniform(
            self.destroy_degree_min,
            self.destroy_degree_max,
        )
        if operator == "random_removal":
            return self._random_removal(solution, destroy_rate)
        elif operator == "worst_removal":
            return self._worst_removal(solution, destroy_rate)
        else:
            raise ValueError(f"Unknown destroy operator: {operator}")

    def _apply_repair_operator(
        self,
        operator: str,
        solution: Solution,
        destroyed_nodes: List[int],
    ) -> Solution:
        """应用修复算子"""
        if operator == "greedy_insertion":
            return self._greedy_insertion(solution, destroyed_nodes)
        elif operator == "random_insertion":
            return self._random_insertion(solution, destroyed_nodes)
        else:
            raise ValueError(f"Unknown repair operator: {operator}")

    def _random_removal(
        self,
        solution: Solution,
        destroy_rate: float,
    ) -> Tuple["Solution", List[int]]:
        """随机移除算子"""
        new_solution = solution.copy()
        destroyed_nodes = []

        # 提取已经包含的任务数
        included_nodes = new_solution.get_included_nodes()
        total_included = len(included_nodes)

        if total_included == 0:
            raise ValueError(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[warning]当前解中无已包含节点，无法进行随机移除。"
            )

        num_nodes_to_remove = int(destroy_rate * total_included)
        num_nodes_to_remove = max(1, num_nodes_to_remove)  # 至少移除一个节点

        # 确定非空路径
        non_empty_veh_id = [
            veh_id for veh_id in self.id_vehicle if len(new_solution.routes[veh_id]) > 2
        ]

        # 预防错误，解为空解
        if not non_empty_veh_id:
            raise ValueError("当前解中无非空路径，无法进行随机移除操作。")

        # 记录移除点位和移除的任务集[索引为(veh_id, node_idx),值为覆盖任务集]
        destroy_plan: Dict[Tuple[int, int], Set[int]] = {}

        # 随机选择移除点位
        while len(destroyed_nodes) < num_nodes_to_remove:
            random_veh = random.choice(non_empty_veh_id)
            # 排除起点和终点
            if len(new_solution.routes[random_veh]) <= 2:
                non_empty_veh_id.remove(random_veh)
                continue
            # 随机选择点位
            random_idx = random.randint(1, len(new_solution.routes[random_veh]) - 2)
            node_to_remove = new_solution.routes[random_veh][random_idx]
            # 提取该节点覆盖的任务
            covered_tasks = new_solution.get_coverage(random_veh, node_to_remove)
            if not covered_tasks:
                raise ValueError(
                    f"车辆 {random_veh} 访问节点 {node_to_remove} 未覆盖"
                    "任何任务，无法进行移除操作。"
                )
            # 存入待移除位置
            destroy_plan[(random_veh, random_idx)] = covered_tasks
            # 记录已移除任务
            destroyed_nodes.extend(covered_tasks)

        # 执行移除操作
        # 先把所有的plan，按照索引从大到小移除（）先按照veh_id排序，再按照node_idx，防止索引变化
        for veh_id, node_idx in sorted(
            destroy_plan.keys(),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        ):
            # 移除节点
            node_to_remove = new_solution.routes[veh_id][node_idx]
            new_solution.routes[veh_id].pop(node_idx)
            # 清除覆盖关系
            new_solution.coverage.pop((veh_id, node_to_remove), None)

        new_solution._init_attrs()

        return new_solution, destroyed_nodes

    def _worst_removal(
        self,
        solution: Solution,
        destroy_rate: float,
    ) -> Tuple["Solution", List[int]]:
        """移除最差节点算子

        计算每个访问节点删除前后的成本变化(目标函数变化),
        按成本变化由大到小排序,优先删除对目标函数贡献最小(或负贡献)的节点
        """
        new_solution = solution.copy()
        destroyed_nodes = []

        # 提取已经包含的任务数
        included_nodes = new_solution.get_included_nodes()
        total_included = len(included_nodes)

        if total_included == 0:
            raise ValueError(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[warning]当前解中无已包含节点，无法进行最差移除。"
            )

        num_nodes_to_remove = int(destroy_rate * total_included)
        num_nodes_to_remove = max(1, num_nodes_to_remove)  # 至少移除一个节点

        # 确定非空路径
        non_empty_veh_id = [
            veh_id for veh_id in self.id_vehicle if len(new_solution.routes[veh_id]) > 2
        ]

        # 预防错误，解为空解
        if not non_empty_veh_id:
            raise ValueError("当前解中无非空路径，无法进行最差移除操作。")

        # 计算当前解的目标函数值
        current_objective = new_solution.objectives[0]

        # 存储每个可移除节点的信息: (成本变化, veh_id, node_idx, 覆盖任务集)
        removal_candidates = []

        # 遍历所有非空路径中的节点
        for veh_id in non_empty_veh_id:
            route = new_solution.routes[veh_id]
            # 排除起点和终点
            for node_idx in range(1, len(route) - 1):
                node = route[node_idx]

                # 提取该节点覆盖的任务
                covered_tasks = new_solution.get_coverage(veh_id, node)
                if not covered_tasks:
                    continue

                # 创建临时解，测试移除该节点后的目标函数值
                temp_solution = new_solution.copy()
                temp_solution.routes[veh_id].pop(node_idx)
                temp_solution.coverage.pop((veh_id, node), None)
                temp_solution._init_attrs()

                # 评估临时解
                self.evaluator.sol_evaluate(temp_solution)
                temp_objective = temp_solution.objectives[0]

                # 计算成本变化(原目标函数值 - 新目标函数值)
                # 成本变化越大，说明删除该节点导致目标函数下降越多(该节点贡献越大)
                # 我们要删除的是贡献最小的节点，即成本变化最小的节点
                cost_change = current_objective - temp_objective

                # 存储候选节点信息
                removal_candidates.append(
                    (cost_change, veh_id, node_idx, covered_tasks)
                )

        # 如果没有可移除的候选节点
        if not removal_candidates:
            return new_solution, destroyed_nodes

        # 按成本变化从小到大排序(优先删除贡献最小的节点)
        # 对于成本变化相同的节点，保持原有顺序(稳定排序)
        removal_candidates.sort(key=lambda x: x[0])

        # 选择要删除的节点
        # 记录移除计划: Dict[Tuple[int, int], Set[int]] = {(veh_id, node_idx): covered_tasks}
        destroy_plan: Dict[Tuple[int, int], Set[int]] = {}

        # 从成本变化最小的开始选择
        idx = 0
        while len(destroyed_nodes) < num_nodes_to_remove and idx < len(
            removal_candidates
        ):
            # 处理成本变化相同的节点组
            current_cost = removal_candidates[idx][0]
            same_cost_group = []

            # 收集所有成本变化相同的节点
            while (
                idx < len(removal_candidates)
                and removal_candidates[idx][0] == current_cost
            ):
                same_cost_group.append(removal_candidates[idx])
                idx += 1

            # 如果有多个成本相同的节点，随机打乱顺序
            if len(same_cost_group) > 1:
                random.shuffle(same_cost_group)

            # 从这组节点中选择
            for cost_change, veh_id, node_idx, covered_tasks in same_cost_group:
                if len(destroyed_nodes) >= num_nodes_to_remove:
                    break

                # 检查该节点是否已被标记删除(通过检查是否在同一路径的更前位置)
                conflict = False
                for existing_veh, existing_idx in destroy_plan.keys():
                    if existing_veh == veh_id:
                        conflict = True
                        break

                if conflict:
                    continue

                # 加入删除计划
                destroy_plan[(veh_id, node_idx)] = set(covered_tasks)
                destroyed_nodes.extend(covered_tasks)

        # 执行移除操作
        # 按照(veh_id, node_idx)排序，从后向前删除，避免索引变化
        for veh_id, node_idx in sorted(
            destroy_plan.keys(),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        ):
            # 移除节点
            node_to_remove = new_solution.routes[veh_id][node_idx]
            new_solution.routes[veh_id].pop(node_idx)
            # 清除覆盖关系
            new_solution.coverage.pop((veh_id, node_to_remove), None)

        # 重新初始化解的属性
        new_solution._init_attrs()

        return new_solution, destroyed_nodes

    def _greedy_insertion(
        self,
        solution: Solution,
        destroyed_nodes: List[int],
    ) -> Solution:
        """贪婪插入算子

        对每个待插入的任务,计算所有可行插入位置的成本和收益,
        选择(收益-成本)最大的位置进行插入。
        收益按照能覆盖的任务的优先级之和计算。
        """
        new_solution = solution.copy()

        # 获取当前未覆盖的任务集合
        included_tasks = new_solution.get_included_nodes()
        unincluded_tasks = set(self.id_demand) - included_tasks

        # 将待插入节点转换为集合,便于动态删除
        remaining_nodes = set(destroyed_nodes)

        # 持续尝试插入,直到没有节点可以插入
        while remaining_nodes:
            best_insertion = (
                None  # (净收益, veh_id, insert_pos, task, covered_tasks_set)
            )
            best_net_benefit = float("-inf")

            # 评估当前解的目标函数(用于计算插入成本)
            current_objective = new_solution.objectives[0]

            # 遍历所有待插入的任务
            for task in remaining_nodes:
                # 跳过已经被覆盖的任务
                if task in included_tasks:
                    continue

                # 遍历所有车辆
                for veh_id in self.id_vehicle:
                    # 检查地面车辆的可达性
                    is_ground_veh = veh_id in self.id_Ground
                    if is_ground_veh and not self.instance.accessible[task]:
                        continue

                    # 计算该节点在该车辆上能覆盖的任务
                    covered_mask = self.comm_coverage_matrix[veh_id][task]
                    all_covered_tasks = [
                        t
                        for t in range(len(covered_mask))
                        if covered_mask[t] and t in unincluded_tasks
                    ]

                    # 如果无法覆盖任何未覆盖任务,跳过
                    if not all_covered_tasks:
                        continue

                    # 计算收益:能覆盖的任务的优先级之和
                    benefit = sum(self.priority_list[t] for t in all_covered_tasks)

                    # 遍历该车辆路径上的所有可能插入位置
                    route = new_solution.routes[veh_id]
                    for insert_pos in range(1, len(route)):
                        # 组成临时插入后路径
                        candidate_route = (
                            route[:insert_pos] + [task] + route[insert_pos:]
                        )

                        # 提取当前解中的覆盖信息(只提取当前车辆相关的)
                        current_coverage = {
                            key: new_solution.get_coverage(key[0], key[1])
                            for key in new_solution.coverage.keys()
                            if key[0] == veh_id
                        }
                        current_coverage[(veh_id, task)] = set(all_covered_tasks)

                        # 计算路径信息，判断是否可行
                        route_feasible, feasible_reason = (
                            self.evaluator.is_route_feasible(
                                veh_id,
                                candidate_route,
                                current_coverage,
                            )
                        )

                        # 如果不可行,跳过该插入位置
                        if not route_feasible:
                            continue

                        # 创建临时解计算插入成本
                        temp_solution = new_solution.copy()
                        temp_solution.routes[veh_id].insert(insert_pos, task)
                        temp_solution.set_coverage(veh_id, task, set(all_covered_tasks))
                        temp_solution._init_attrs()

                        # 评估临时解
                        self.evaluator.sol_evaluate(temp_solution)
                        new_objective = temp_solution.objectives[0]

                        # 计算插入成本(目标函数的变化,注意目标是最大化)
                        # 如果目标函数增加,成本为负;如果目标函数减少,成本为正
                        cost = current_objective - new_objective

                        # 计算净收益 = 收益 - 成本
                        net_benefit = benefit - cost

                        # 更新最佳插入位置
                        if net_benefit > best_net_benefit:
                            best_net_benefit = net_benefit
                            best_insertion = (
                                net_benefit,
                                veh_id,
                                insert_pos,
                                task,
                                set(all_covered_tasks),
                            )

            # 如果没有找到可行的插入位置,结束插入
            if best_insertion is None:
                break

            # 执行最佳插入
            _, veh_id, insert_pos, task, covered_tasks_set = best_insertion
            new_solution.routes[veh_id].insert(insert_pos, task)
            new_solution.set_coverage(veh_id, task, covered_tasks_set)

            # 更新已覆盖任务集合
            included_tasks.update(covered_tasks_set)
            unincluded_tasks -= covered_tasks_set

            # 从待插入节点集合中移除已成功插入的节点
            remaining_nodes.discard(task)
            # 同时移除所有被覆盖的任务(因为它们已经不需要再插入)
            remaining_nodes -= covered_tasks_set

            # 重新初始化解的属性(为下次迭代准备)
            new_solution._init_attrs()

        # 最终重新初始化解的属性
        new_solution._init_attrs()

        return new_solution

    def _random_insertion(
        self,
        solution: Solution,
        destroyed_nodes: List[int],
    ) -> Solution:
        """随机插入算子

        将被移除的节点打乱顺序后,依次尝试插入到随机选择的车辆路径中
        """
        new_solution = solution.copy()

        # 获取当前未覆盖的任务集合
        included_tasks = new_solution.get_included_nodes()
        unincluded_tasks = set(self.id_demand) - included_tasks

        # 打乱待插入节点的顺序
        random.shuffle(destroyed_nodes)

        # 将待插入节点转换为集合,便于动态删除
        remaining_nodes = set(destroyed_nodes)

        # 依次尝试插入每个节点
        for task in destroyed_nodes:
            # 如果该节点已经被插入过,跳过
            if task not in remaining_nodes:
                continue

            # 如果任务已被覆盖,跳过
            if task in included_tasks:
                raise ValueError(f"任务 {task} 已被覆盖，无法重复插入。")

            # 随机选择一个车辆
            veh_id = random.choice(self.id_vehicle)

            # 检查地面车辆的可达性
            is_ground_veh = veh_id in self.id_Ground
            if is_ground_veh and not self.instance.accessible[task]:
                continue

            # 计算该节点在该车辆上能覆盖的任务
            covered_mask = self.comm_coverage_matrix[veh_id][task]
            all_covered_tasks = [
                t
                for t in range(len(covered_mask))
                if covered_mask[t] and t in unincluded_tasks
            ]

            # 如果无法覆盖任何未覆盖任务,跳过
            if not all_covered_tasks:
                raise ValueError(
                    f"任务 {task} 在车辆 {veh_id} 上无法覆盖任何未覆盖任务，"
                    "无法插入。"
                )

            # 随机选择一个插入位置(排除起点和终点之间)
            route = new_solution.routes[veh_id]
            if len(route) <= 2:
                # 空路径,插入到起点和终点之间
                insert_pos = 1
            else:
                # 随机选择插入位置(1到len(route)-1之间)
                insert_pos = random.randint(1, len(route) - 1)

            # 组成临时插入后路径
            candidate_route = route[:insert_pos] + [task] + route[insert_pos:]

            # 提取当前解中的覆盖信息(只提取当前车辆相关的)
            current_coverage = {
                key: new_solution.get_coverage(key[0], key[1])
                for key in new_solution.coverage.keys()
                if key[0] == veh_id
            }
            current_coverage[(veh_id, task)] = set(all_covered_tasks)

            # 计算路径信息，判断是否可行
            route_feasible, feasible_reason = self.evaluator.is_route_feasible(
                veh_id,
                candidate_route,
                current_coverage,
            )

            # 如果不可行,跳过该插入
            if not route_feasible:
                continue

            # 执行插入
            new_solution.routes[veh_id].insert(insert_pos, task)

            # 登记覆盖情况
            covered_tasks_set = set(all_covered_tasks)
            new_solution.set_coverage(veh_id, task, covered_tasks_set)

            # 更新已覆盖任务集合
            included_tasks.update(covered_tasks_set)
            unincluded_tasks -= covered_tasks_set

            # 从待插入节点集合中移除已成功插入的节点
            remaining_nodes.discard(task)
            # 同时移除所有被覆盖的任务(因为它们已经不需要再插入)
            remaining_nodes -= covered_tasks_set

        # 重新初始化解的属性
        new_solution._init_attrs()

        return new_solution

    def _accept_solution(
        self, new_solution: Solution, current_solution: Solution
    ) -> bool:
        """判断是否接受新解(可使用模拟退火准则)"""
        if new_solution.objectives[0] > current_solution.objectives[0]:
            return True
        else:
            return False

    def _update_weights(
        self,
        destroy_op: str,
        repair_op: str,
        new_solution: Solution,
        accepted: bool,
    ):
        """
        更新操作算子权重

        使用分层奖励机制:
        - Level 3: 新解成为最优解 → score = 3.0
        - Level 2: 新解被接受且有改进 → score = 2.0
        - Level 1: 新解被接受但无改进 → score = 1.0
        - Level 0: 新解未被接受 → score = 0.0

        使用遗忘率更新公式:
        new_score = decay_rate * old_score + (1 - decay_rate) * current_score
        """
        # 计算本次得分
        if accepted:
            if new_solution.objectives[0] < self.best_objective:
                # 新解成为最优解
                score = 3.0
            elif new_solution.objectives[0] < self.current_sol.objectives[0]:
                # 新解优于当前解
                score = 2.0
            else:
                # 新解被接受但无改进(可能是模拟退火接受的劣解)
                score = 1.0
        else:
            # 新解未被接受
            score = 0.0

        # 使用遗忘率更新破坏算子得分
        # new_score = decay_rate * old_score + (1 - decay_rate) * current_score
        self.destroy_scores[destroy_op] = (
            self.decay_rate * self.destroy_scores[destroy_op]
            + (1 - self.decay_rate) * score
        )

        # 使用遗忘率更新修复算子得分
        self.repair_scores[repair_op] = (
            self.decay_rate * self.repair_scores[repair_op]
            + (1 - self.decay_rate) * score
        )

        # 防止权重过小(设置最小阈值)
        min_weight = 0.1
        for op in self.destroy_scores:
            self.destroy_scores[op] = max(self.destroy_scores[op], min_weight)
        for op in self.repair_scores:
            self.repair_scores[op] = max(self.repair_scores[op], min_weight)

        # 记录本次迭代的算子使用情况和得分(可选,用于后续分析)
        self.iteration_operator_stats["destroy"][destroy_op].append(score)
        self.iteration_operator_stats["repair"][repair_op].append(score)

    def solve(self) -> Solution:
        """使用ALNS算法进行求解"""

        # 开始计时
        start_time = time.perf_counter()

        # 初始化解
        self._init_solution()

        # 初始化操作算子权重
        self._init_weights()

        # 打印调试信息的列名称(仅一次)
        if self.enable_debug:
            print(
                f"\n{'='*80}\n"
                f"{'time':<20} | {'iteration':<10} | {'current':<12} | "
                f"{'candidate':<12} | {'best':<12}\n"
                f"{'-'*80}"
            )

        for iteration in range(self.max_iter):
            # 判断是否早停
            if self.enable_early_stop:
                if self.not_improve_counter >= self.max_not_improve_iter:
                    print(
                        "\n"
                        "=" * 50
                        + f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        + f"\n达到早停条件，迭代终止于第 {iteration} 次迭代。"
                    )
                    break
            # 选择破坏和修复算子
            destroy_op = self._select_destroy_op()
            repair_op = self._select_repair_op()

            # 执行破坏和修复操作
            destroyed_sol, removed_elements = self._apply_destroy_operator(
                destroy_op, self.current_sol
            )
            new_sol = self._apply_repair_operator(
                repair_op,
                destroyed_sol,
                removed_elements,
            )

            # 评估新解
            self.evaluator.sol_evaluate(new_sol)

            # 接受或拒绝新解
            accept = self._accept_solution(new_sol, self.current_sol)

            # 记录本次迭代信息(在接受判断之后,便于记录接受状态)
            self.history.record_iteration(
                iteration=iteration + 1,
                best_objective=self.best_objective,
                current_objective=self.current_sol.objectives[0],
                candidate_objective=new_sol.objectives[0],
                destroy_operator=destroy_op,
                repair_operator=repair_op,
                accepted=accept,
            )

            # 更新当前解和最优解
            if accept:
                self.current_sol = new_sol
                if new_sol.objectives[0] > self.best_objective:
                    # 更新最优解
                    self.best_sol = new_sol.copy()
                    # 更新最优目标函数值
                    self.best_objective = new_sol.objectives[0]
                    # 初始化早停计数器
                    self.not_improve_counter = 0
                else:
                    # 未改进最优解，早停计数器加1
                    self.not_improve_counter += 1

            # 更新操作算子权重
            self._update_weights(destroy_op, repair_op, new_sol, accept)

            # 调试打印信息(仅打印数值)
            if self.enable_debug and (iteration + 1) % 20 == 0:
                iter_str = f"{iteration + 1:>4}/{self.max_iter:<5}"
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} | "
                    f"{iter_str} | "
                    f"{self.current_sol.objectives[0]:>12.4f} | "
                    f"{new_sol.objectives[0]:>12.4f} | "
                    f"{self.best_objective:>12.4f}"
                )

        # 结束计时
        end_time = time.perf_counter()
        self.best_sol.solve_time = end_time - start_time
        self.best_sol.status = f"ALNS_{iteration+1}_iters"

        return self.best_sol


def alns_solve(
    solution: Solution,
    instance: InstanceClass,
) -> Solution:
    """使用ALNS算法求解算例"""
    alns_solver = ALNS(instance)
    best_solution = alns_solver.solve()
    return best_solution
