import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.element import InstanceClass, Solution
from src.solver.alns import ALNS
from src.solver.alns_history import ALNSHistory
from src.solver.consr_heuristic import multi_start_initial_solution


class MSALNS(ALNS):
    """多起点自适应大邻域搜索算法类"""

    def __init__(
        self,
        instance: InstanceClass,
    ):
        """初始化多起点ALNS算法

        Args:
            instance (InstanceClass): 问题实例
        """
        super().__init__(instance)

        algorithm_config = instance.prob_config.algorithm_config

        self.enable_early_stop = algorithm_config.enable_early_stop

        self.num_starts = algorithm_config.multi_start_num
        self.solution_bank: List[Solution] = []

        # MS-ALNS专用变量
        self.global_best_objective: float = 0.0  # 全局最优目标值
        self.global_best_solution: Optional[Solution] = None  # 全局最优解

        # ---------- 破坏操作算子 ----------
        self.destroy_operators = [
            "random_removal",
            "worst_removal",
            "heavy_route_removal",
            "coverage_based_removal",
        ]

        # ---------- 修复操作算子 ----------
        self.repair_operators = [
            "greedy_insertion",
            "random_insertion",
            "memory_based_insertion",
        ]

        # 重写历史记录器
        self.history = ALNSHistory()

        # 初始化信息素矩阵,全部初始化为1.0
        veh_num = len(self.id_vehicle)
        node_num = len(self.id_base) + self.demand_num
        self.pheromone_matrix = np.zeros((veh_num, node_num, node_num))
        sub_travel_time = self.travel_time_matrix[:, :node_num, :node_num]
        mask = (sub_travel_time != 1e6) & (sub_travel_time != 0)
        self.pheromone_matrix[mask] = 1.0

        # 提取费洛蒙矩阵相关参数
        # 挥发率
        self.evaporation_rate = algorithm_config.pheromone_evaporation_rate
        # 信息素最大值和最小值
        self.tau_max = algorithm_config.tau_max
        self.tau_min = algorithm_config.tau_min

    def _init_solutions(self) -> None:
        """初始化多起点解集"""
        self.solution_bank = multi_start_initial_solution(
            self.instance, self.num_starts
        )

        # 评估所有初始解
        for sol in self.solution_bank:
            self.evaluator.sol_evaluate(sol)

        # 排序解集
        self._sort_solution_bank()

        # 初始化全局最优解
        self.global_best_solution = self.solution_bank[0].copy()
        self.global_best_objective = self.solution_bank[0].objectives[0]

        # 记录初始状态
        self.history.record_initial(
            best_objective=self.global_best_objective,
            current_objective=self.global_best_objective,
        )

        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"{len(self.solution_bank)} 个初始解已生成，准备进入MS-ALNS主循环。"
        )
        print(
            f"初始解集目标值范围: "
            f"[{self.solution_bank[-1].objectives[0]:.4f}, "
            f"{self.solution_bank[0].objectives[0]:.4f}]"
        )

    def _sort_solution_bank(self) -> None:
        """对解集按照目标函数值进行排序（降序）"""
        self.solution_bank.sort(
            key=lambda sol: sol.objectives[0],
            reverse=True,
        )

    def _update_weights_ms(
        self,
        destroy_op: str,
        repair_op: str,
        new_solution: Solution,
        current_solution: Solution,
        accepted: bool,
    ):
        """
        更新操作算子权重（多起点版本）

        使用分层奖励机制:
        - Level 4: 新解优于全局最优解 → score = 4.0
        - Level 3: 新解优于解池中最优解但不超过全局最优 → score = 3.0
        - Level 2: 新解优于当前候选解 → score = 2.0
        - Level 1: 新解被接受但无改进 → score = 1.0
        - Level 0: 新解未被接受 → score = 0.0

        使用遗忘率更新公式:
        new_score = decay_rate * old_score + (1 - decay_rate) * current_score
        """
        # 计算本次得分
        if accepted:
            if new_solution.objectives[0] > self.global_best_objective:
                # 新解优于全局最优解
                score = 4.0
            elif new_solution.objectives[0] > self.solution_bank[0].objectives[0]:
                # 新解优于解池中最优解
                score = 3.0
            elif new_solution.objectives[0] > current_solution.objectives[0]:
                # 新解优于当前候选解
                score = 2.0
            else:
                # 新解被接受但无改进（可能是模拟退火接受的劣解）
                score = 1.0
        else:
            # 新解未被接受
            score = 0.0

        # 使用遗忘率更新破坏算子得分
        self.destroy_scores[destroy_op] = (
            self.decay_rate * self.destroy_scores[destroy_op]
            + (1 - self.decay_rate) * score
        )

        # 使用遗忘率更新修复算子得分
        self.repair_scores[repair_op] = (
            self.decay_rate * self.repair_scores[repair_op]
            + (1 - self.decay_rate) * score
        )

        # 防止权重过小（设置最小阈值）
        min_weight = 0.1
        for op in self.destroy_scores:
            self.destroy_scores[op] = max(self.destroy_scores[op], min_weight)
        for op in self.repair_scores:
            self.repair_scores[op] = max(self.repair_scores[op], min_weight)

        # 记录本次迭代的算子使用情况和得分
        self.iteration_operator_stats["destroy"][destroy_op].append(score)
        self.iteration_operator_stats["repair"][repair_op].append(score)

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
        elif operator == "heavy_route_removal":
            return self._heavy_route_removal(solution, destroy_rate)
        elif operator == "coverage_based_removal":
            return self._coverage_based_removal(solution, destroy_rate)
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
        elif operator == "memory_based_insertion":
            return self._memory_based_insertion(solution, destroyed_nodes)
        else:
            raise ValueError(f"Unknown repair operator: {operator}")

    def _heavy_route_removal(
        self, solution: Solution, destroy_rate: float
    ) -> Tuple["Solution", List[int]]:
        # 实现重载路径移除算子逻辑
        new_solution = solution.copy()
        destroyed_nodes: List[int] = []

        # 提取已经包含的任务数
        included_nodes = new_solution.get_included_nodes()
        total_included = len(included_nodes)

        if total_included == 0:
            raise ValueError(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[error]当前解中无已包含节点，无法进行覆盖基移除。"
            )

        num_to_remove = int(total_included * destroy_rate)
        num_to_remove = max(1, num_to_remove)  # 至少移除一个节点

        # 提取各车辆的coverage
        coverage_info = new_solution.coverage

        veh_coverage_dict: Dict[int, Dict[Tuple[int, int], Set[int]]] = dict()
        for key, coverage_set in coverage_info.items():
            veh_id, visit_node = key
            if veh_id not in veh_coverage_dict:
                veh_coverage_dict[veh_id] = dict()
            veh_coverage_dict[veh_id].update({key: coverage_set})

        # 计算每条路径的路径时长，并按时长排序
        route_durations: List[Tuple[int, float]] = []
        for veh_id, route in enumerate(new_solution.routes):
            route_duration = self.evaluator._calculate_route_duration(
                route, veh_coverage_dict.get(veh_id, dict()), veh_id
            )
            route_durations.append((veh_id, route_duration))
        route_durations.sort(key=lambda x: x[1], reverse=True)
        # 依次从长路径中移除节点，直到达到移除数量
        del_plans: Dict[int, List[int]] = dict()
        for veh_id, _ in route_durations:
            route = new_solution.routes[veh_id]
            for visit_idx, visit_node in enumerate(route):
                visit_coverage = coverage_info.get((veh_id, visit_node), set())
                if visit_node in included_nodes:
                    if veh_id not in del_plans:
                        del_plans[veh_id] = []
                    del_plans[veh_id].append(visit_idx)
                    destroyed_nodes.extend(list(visit_coverage))
                if len(destroyed_nodes) >= num_to_remove:
                    break  # 达到移除数量，跳出循环
            if len(destroyed_nodes) >= num_to_remove:
                break  # 达到移除数量，跳出循环
        # 执行删除计划
        # 先对每条路径的删除位置进行排序（从大到小），
        for veh_id, visit_indices in del_plans.items():
            visit_indices.sort(reverse=True)
            for visit_idx in visit_indices:
                del_visit_node = new_solution.routes[veh_id].pop(visit_idx)
                if (veh_id, del_visit_node) in new_solution.coverage:
                    del new_solution.coverage[(veh_id, del_visit_node)]
        # 更新解的属性
        new_solution._init_attrs()
        return new_solution, destroyed_nodes

    def _coverage_based_removal(
        self, solution: Solution, destroy_rate: float
    ) -> Tuple["Solution", List[int]]:
        # 实现基于覆盖的移除算子逻辑
        new_solution = solution.copy()
        destroyed_nodes: List[int] = []

        # 提取已经包含的任务数
        included_nodes = new_solution.get_included_nodes()
        total_included = len(included_nodes)

        if total_included == 0:
            raise ValueError(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[error]当前解中无已包含节点，无法进行覆盖基移除。"
            )

        num_to_remove = int(total_included * destroy_rate)
        num_to_remove = max(1, num_to_remove)  # 至少移除一个节点

        # 提取覆盖信息
        coverage_info = new_solution.coverage

        # coverage_matrix: shape = (total_veh_num, total_node_num+1, total_node_num+1)
        # coverage_matrix[v, i, j] 表示车辆v在节点i时能否覆盖节点j
        coverage_matrix = self.coverage_matrix

        # 计算每个included_node可以被多少种(车辆v, 位置i)组合覆盖
        # 即统计有多少个(v, i)使得coverage_matrix[v, i, node]=True
        candidate_counts = {
            node: int(coverage_matrix[:, :, node].sum()) for node in included_nodes
        }

        # 按照candidate_counts从大到小排序，优先移除覆盖选择多的节点
        sorted_nodes = sorted(
            candidate_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # 生成删除计划
        del_plans: Dict[int, List[int]] = dict()
        # 依次移除节点
        for node, count in sorted_nodes:
            found_flag = False
            for veh_id, route in enumerate(new_solution.routes):
                for visit_idx, visit_node in enumerate(route):
                    visit_coverage = coverage_info.get((veh_id, visit_node), set())
                    if visit_node == node:
                        if veh_id not in del_plans:
                            del_plans[veh_id] = []
                        del_plans[veh_id].append(visit_idx)
                        destroyed_nodes.extend(list(visit_coverage))
                        break  # 跳出visit_idx循环
                    elif node in visit_coverage:
                        if not visit_coverage - {node}:
                            raise ValueError(
                                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                                f"[error]节点 {node} 覆盖关系异常，无法进行覆盖基移除。"
                            )
                        else:
                            coverage_info[(veh_id, visit_node)].remove(node)
                            destroyed_nodes.append(node)
                            found_flag = True
                            break  # 跳出visit_idx循环
                if found_flag:
                    break  # 跳出veh_id循环
            if len(destroyed_nodes) >= num_to_remove:
                break  # 达到移除数量，跳出循环

        # 执行删除计划
        # 先对每条路径的删除位置进行排序（从大到小），以免索引错乱
        for veh_id, visit_indices in del_plans.items():
            visit_indices.sort(reverse=True)
            for visit_idx in visit_indices:
                del_visit_node = new_solution.routes[veh_id].pop(visit_idx)
                if (veh_id, del_visit_node) in new_solution.coverage:
                    del new_solution.coverage[(veh_id, del_visit_node)]

        # 更新解的属性
        new_solution._init_attrs()
        return new_solution, destroyed_nodes

    # todo: 基于记忆的插入算子(参考各任务出现在历史解中的表现，设计一个根据记忆表的插入策略)
    def _memory_based_insertion(
        self, solution: Solution, destroyed_nodes: List[int]
    ) -> "Solution":
        # 实现费洛蒙信息矩阵的插入算子逻辑
        pheromone_matrix = self.pheromone_matrix
        new_solution = solution.copy()

        # 分别计算每个被移除节点到每个插入位置的费洛蒙值
        insert_plan_pheromone: Dict[Tuple[int, int, int], float] = dict()
        for node in destroyed_nodes:
            for veh_id, route in enumerate(new_solution.routes):
                for visit_idx in range(1, len(route)):
                    from_node = route[visit_idx - 1]
                    to_node = route[visit_idx]
                    plan_pheromone = (
                        pheromone_matrix[veh_id, from_node, node]
                        + pheromone_matrix[veh_id, node, to_node]
                    )
                    insert_plan_pheromone[(veh_id, visit_idx, node)] = plan_pheromone

        # 按照费洛蒙值从大到小排序插入计划
        sorted_inserts = sorted(
            insert_plan_pheromone.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # 获取当前未覆盖的任务集合
        included_tasks = new_solution.get_included_nodes()
        unincluded_tasks = set(self.id_demand) - included_tasks

        # 将待插入节点转换为集合，便于动态删除
        remaining_nodes = set(destroyed_nodes)

        # 将插入计划转换为列表以便操作
        insert_plans = list(sorted_inserts)
        plan_idx = 0

        # 依次评估插入计划是否可行，如果可行，则执行，否则跳过该计划
        while plan_idx < len(insert_plans):
            (veh_id, insert_pos, task), pheromone_value = insert_plans[plan_idx]

            # 如果该任务已经被插入或覆盖，跳过
            if task not in remaining_nodes or task in included_tasks:
                plan_idx += 1
                continue

            # 检查地面车辆的可达性
            is_ground_veh = veh_id in self.id_Ground
            if is_ground_veh and not self.instance.accessible[task]:
                plan_idx += 1
                continue

            # 计算该节点在该车辆上能覆盖的任务
            covered_mask = self.comm_coverage_matrix[veh_id][task]
            all_covered_tasks = [
                t
                for t in range(len(covered_mask))
                if covered_mask[t] and t in unincluded_tasks
            ]

            # 如果无法覆盖任何未覆盖任务，跳过
            if not all_covered_tasks:
                plan_idx += 1
                continue

            # 获取当前路径
            route = new_solution.routes[veh_id]

            # 确保插入位置有效（路径可能在之前的插入中发生了变化）
            if insert_pos > len(route):
                plan_idx += 1
                continue

            # 组成临时插入后路径
            candidate_route = route[:insert_pos] + [task] + route[insert_pos:]

            # 提取当前解中的覆盖信息（只提取当前车辆相关的）
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

            # 如果不可行，跳过该插入位置
            if not route_feasible:
                plan_idx += 1
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
            # 同时移除所有被覆盖的任务（因为它们已经不需要再插入）
            remaining_nodes -= covered_tasks_set

            # 从插入计划列表中删除所有与已覆盖任务相关的插入方案
            # 只要任务出现在covered_tasks_set中的方案都要删除
            insert_plans = [
                plan for plan in insert_plans if plan[0][2] not in covered_tasks_set
            ]
            # 重置索引，因为列表已经被过滤
            plan_idx = 0

        # 最终重新初始化解的属性
        new_solution._init_attrs()

        return new_solution

    def _process_single_solution(
        self,
        solution: Solution,
        sol_idx: int,
    ) -> Solution:
        """
        对单个解进行ALNS破坏和修复操作

        Args:
            solution: 待处理的解
            sol_idx: 解在解池中的索引

        Returns:
            处理后的新解
        """
        # 选择破坏和修复算子
        destroy_op = self._select_destroy_op()
        repair_op = self._select_repair_op()

        # 执行破坏和修复操作
        destroyed_sol, removed_elements = self._apply_destroy_operator(
            destroy_op, solution
        )
        new_sol = self._apply_repair_operator(
            repair_op,
            destroyed_sol,
            removed_elements,
        )

        # todo: 增加扰动，把未完成的任务尝试插入新解中

        # 评估新解
        self.evaluator.sol_evaluate(new_sol)

        # 接受或拒绝新解
        accept = self._accept_solution(new_sol, solution)

        # 更新操作算子权重（使用MS版本）
        self._update_weights_ms(destroy_op, repair_op, new_sol, solution, accept)

        # 如果接受新解，返回新解；否则返回原解
        if accept:
            return new_sol
        else:
            return solution

    def update_sol_bank(self, new_solution_bank: List[Solution]) -> None:
        # 信息素挥发
        self.evaporate_pheromones()

        # 提取解池中的弧信息，更新费洛蒙矩阵
        edge_frequency = self.count_edge_frenquency(self.solution_bank)
        self.pheromone_reinforcement(edge_frequency)

        # 筛选解池，挑选进入下一代的解池，如果解数量不足则新生成解
        self.solution_bank_update(new_solution_bank)

    def solution_bank_update(
        self,
        new_solution_bank: List[Solution],
    ) -> None:
        """更新解池，选择进入下一代的解池

        Args:
            new_solution_bank (List[Solution]): 新生成的解池
        """
        # 合并当前解池和新解池
        combined_solutions = self.solution_bank + new_solution_bank

        # 按照目标函数排序（降序）
        combined_solutions.sort(
            key=lambda sol: sol.objectives[0],
            reverse=True,
        )

        # 去重：对于相同目标函数值的解，识别并剔除完全重复的解
        unique_solutions: List[Solution] = []
        seen_signatures: Set[str] = set()

        # 按目标函数值分组
        current_obj = None
        obj_group: List[Solution] = []

        for sol in combined_solutions:
            sol_obj = sol.objectives[0]
            # 如果目标函数值变化，处理上一组
            if current_obj is not None and sol_obj != current_obj:
                # 对上一组进行去重
                for s in obj_group:
                    sig = s.get_route_signature()
                    if sig not in seen_signatures:
                        unique_solutions.append(s)
                        seen_signatures.add(sig)
                obj_group = []

            current_obj = sol_obj
            obj_group.append(sol)

        # 处理最后一组
        if obj_group:
            for s in obj_group:
                sig = s.get_route_signature()
                if sig not in seen_signatures:
                    unique_solutions.append(s)
                    seen_signatures.add(sig)

        if len(unique_solutions) < self.num_starts:
            # todo: 如果解数量不足，考虑重新生成解以补充
            solutions_needed = self.num_starts - len(unique_solutions)
            while solutions_needed > 0:
                new_sol = self._new_solution_by_pheromone()
                self.evaluator.sol_evaluate(new_sol)
                unique_solutions.append(new_sol)
                solutions_needed -= 1

        # 选择前num_starts个解作为新的解池
        self.solution_bank = unique_solutions[: self.num_starts]

    def _new_solution_by_pheromone(self) -> Solution:
        """基于当前信息素矩阵生成新解

        纯粹基于费洛蒙信息进行构造:
        1. 从空解开始,初始化所有车辆路径
        2. 对每个待插入任务,计算所有可能的(车辆, 位置)插入方案
        3. 使用费洛蒙值作为权重进行轮盘赌选择
        4. 评估可行性并执行插入
        5. 更新覆盖信息和待分配任务集

        Returns:
            Solution: 基于费洛蒙构造的新解
        """
        new_solution = Solution.new_solution(self.instance)
        new_solution.init_all_routes()

        # 待分配任务集合
        task_to_assign = self.id_demand[:]
        remaining_tasks = set(task_to_assign)

        # 提取必要信息
        comm_coverage_matrix = self.comm_coverage_matrix
        accessible_list = self.instance.accessible
        vehicle_ids = self.id_vehicle
        pheromone_matrix = self.pheromone_matrix

        # 获取当前未覆盖的任务集合
        included_tasks = new_solution.get_included_nodes()
        unincluded_tasks = set(self.id_demand) - included_tasks

        # 主循环: 不断选择任务插入直到无法继续
        while remaining_tasks:
            # 计算每个插入方案的费洛蒙得分
            insert_scores = {}

            for task in remaining_tasks:
                # 如果任务已被覆盖,跳过
                if task in included_tasks:
                    continue

                for veh_id in vehicle_ids:
                    # 检查地面车辆的可达性
                    is_ground_veh = veh_id in self.id_Ground
                    if is_ground_veh and not accessible_list[task]:
                        continue

                    # 计算该任务在该车辆上能覆盖的未覆盖任务
                    covered_mask = comm_coverage_matrix[veh_id][task]
                    all_covered_tasks = [
                        t
                        for t in range(len(covered_mask))
                        if covered_mask[t] and t in unincluded_tasks
                    ]

                    # 如果无法覆盖任何未覆盖任务,跳过
                    if not all_covered_tasks:
                        continue

                    route = new_solution.routes[veh_id]

                    # 对每个可能的插入位置计算费洛蒙值
                    for insert_pos in range(1, len(route)):
                        from_node = route[insert_pos - 1]
                        to_node = route[insert_pos]

                        # 跳过信息素为0的弧（这些弧在历史解中从未出现过）
                        pheromone_from_task = pheromone_matrix[veh_id, from_node, task]
                        pheromone_task_to = pheromone_matrix[veh_id, task, to_node]

                        # 如果任一弧的信息素为0，跳过该插入方案
                        if pheromone_from_task == 0 or pheromone_task_to == 0:
                            continue

                        # 计算费洛蒙得分: (from->task) + (task->to) - (from->to)
                        # 减去原有弧的费洛蒙,体现插入的净收益
                        pheromone_score = (
                            pheromone_from_task
                            + pheromone_task_to
                            - pheromone_matrix[veh_id, from_node, to_node]
                        )

                        insert_scores[(veh_id, insert_pos, task)] = pheromone_score

            if not insert_scores:
                break

            # 使用轮盘赌选择插入方案
            # 将所有得分转换为非负值
            min_score = min(insert_scores.values())
            if min_score < 0:
                adjusted_scores = {
                    k: v - min_score + 1e-6 for k, v in insert_scores.items()
                }
            else:
                adjusted_scores = {k: v + 1e-6 for k, v in insert_scores.items()}

            # 轮盘赌选择
            total_score = sum(adjusted_scores.values())
            probabilities = [score / total_score for score in adjusted_scores.values()]
            plans = list(adjusted_scores.keys())

            # 随机选择一个插入方案
            selected_plan = np.random.choice(
                len(plans),
                p=probabilities,
            )
            veh_id, insert_pos, task = plans[selected_plan]

            # 计算覆盖任务
            covered_mask = comm_coverage_matrix[veh_id][task]
            all_covered_tasks = [
                t
                for t in range(len(covered_mask))
                if covered_mask[t] and t in unincluded_tasks
            ]

            # 构建候选路径
            route = new_solution.routes[veh_id]
            candidate_route = route[:insert_pos] + [task] + route[insert_pos:]

            # 提取当前车辆的覆盖信息
            current_coverage = {
                key: new_solution.get_coverage(key[0], key[1])
                for key in new_solution.coverage.keys()
                if key[0] == veh_id
            }
            current_coverage[(veh_id, task)] = set(all_covered_tasks)

            # 评估可行性
            route_feasible, _ = self.evaluator.is_route_feasible(
                veh_id,
                candidate_route,
                current_coverage,
            )

            if route_feasible:
                # 执行插入
                new_solution.routes[veh_id] = candidate_route
                covered_tasks_set = set(all_covered_tasks)
                new_solution.set_coverage(veh_id, task, covered_tasks_set)

                # 更新已覆盖任务集合
                included_tasks.update(covered_tasks_set)
                unincluded_tasks -= covered_tasks_set

                # 从待插入节点集合中移除已成功插入的节点
                remaining_tasks.discard(task)
                # 同时移除所有被覆盖的任务（因为它们已经不需要再插入）
                remaining_tasks -= covered_tasks_set
            else:
                # 不可行则从候选中移除该任务
                remaining_tasks.discard(task)

        # 初始化解的属性
        new_solution._init_attrs()
        return new_solution

    def pheromone_reinforcement(
        self,
        edge_frequency: Dict[Tuple[int, int, int], int],
    ) -> None:
        for (veh_id, from_node, to_node), freq in edge_frequency.items():
            delta_tau = freq / len(self.solution_bank)  # 归一化增量
            self.pheromone_matrix[veh_id, from_node, to_node] += delta_tau
        mask = self.pheromone_matrix != 0
        np.clip(
            self.pheromone_matrix[mask],
            self.tau_min,
            self.tau_max,
        )

    def evaporate_pheromones(self):
        """信息素挥发"""
        mask = self.pheromone_matrix != 0
        self.pheromone_matrix[mask] *= 1 - self.evaporation_rate
        # 保持信息素在设定范围内
        np.clip(
            self.pheromone_matrix[mask],
            self.tau_min,
            self.tau_max,
        )

    def count_edge_frenquency(
        self,
        solution_list: List[Solution],
    ) -> Dict[Tuple[int, int, int], int]:
        """统计解中各弧出现的频次，更新信息素矩阵"""
        dege_count: Dict[Tuple[int, int, int], int] = dict()
        for solution in solution_list:
            for veh_id, route in enumerate(solution.routes):
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    if (veh_id, from_node, to_node) not in dege_count:
                        dege_count[(veh_id, from_node, to_node)] = 0
                    dege_count[(veh_id, from_node, to_node)] += 1
        return dege_count

    def solve(self) -> Solution:
        """执行多起点ALNS算法求解

        Returns:
            Solution: 最优解对象
        """
        # 开始计时
        start_time = time.perf_counter()

        # 初始化多起点解集
        self._init_solutions()

        # 初始化操作算子权重
        self._init_weights()

        # 打印调试信息的列名称（仅一次）
        if self.enable_debug:
            print(
                f"\n{'='*100}\n"
                f"{'time':<20} | {'iteration':<10} | "
                f"{'pool_best':<12} | {'pool_worst':<12} | "
                f"{'global_best':<12} | {'not_improve':<12}\n"
                f"{'-'*100}"
            )

        # 主循环
        for iteration in range(self.max_iter):
            # 判断是否早停
            if self.enable_early_stop:
                if self.not_improve_counter >= self.max_not_improve_iter:
                    print(
                        "=========================================\n"
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        f"达到早停条件，迭代终止于第 {iteration} 次迭代。"
                    )
                    break

            # 对解池中的每个解进行ALNS破坏和修复
            new_solution_bank: List[Solution] = []
            for sol_idx, solution in enumerate(self.solution_bank):
                new_sol = self._process_single_solution(solution, sol_idx)
                new_solution_bank.append(new_sol)

            # 对解池重新排序
            self.update_sol_bank(new_solution_bank)

            # 更新全局最优解
            pool_best_objective = self.solution_bank[0].objectives[0]
            if pool_best_objective > self.global_best_objective:
                self.global_best_solution = self.solution_bank[0].copy()
                self.global_best_objective = pool_best_objective
                self.not_improve_counter = 0
            else:
                self.not_improve_counter += 1

            # 记录本次迭代信息
            self.history.record_iteration(
                iteration=iteration + 1,
                best_objective=self.global_best_objective,
                current_objective=pool_best_objective,
                candidate_objective=pool_best_objective,
                destroy_operator="mixed",  # 多起点使用混合算子
                repair_operator="mixed",
                accepted=True,  # 解池更新视为接受
            )

            # 调试打印信息
            if self.enable_debug and (iteration + 1) % 20 == 0:
                iter_str = f"{iteration + 1:>4}/{self.max_iter:<5}"
                pool_worst_objective = self.solution_bank[-1].objectives[0]
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} | "
                    f"{iter_str} | "
                    f"{pool_best_objective:>12.4f} | "
                    f"{pool_worst_objective:>12.4f} | "
                    f"{self.global_best_objective:>12.4f} | "
                    f"{self.not_improve_counter:>12}"
                )

        # 结束计时
        end_time = time.perf_counter()

        # 确保有最优解
        if self.global_best_solution is None:
            raise RuntimeError("未找到有效解，算法求解失败。")

        self.global_best_solution.solve_time = end_time - start_time
        self.global_best_solution.Solver_name = "MS_ALNS"
        self.global_best_solution.status = f"MS-ALNS_{iteration+1}_iters"

        # 打印最终统计信息
        print(
            f"\n{'='*100}\n"
            f"MS-ALNS算法求解完成\n"
            f"总迭代次数: {iteration + 1}\n"
            f"最优目标值: {self.global_best_objective:.4f}\n"
            f"求解时间: {self.global_best_solution.solve_time:.2f}秒\n"
            f"{'='*100}"
        )

        return self.global_best_solution


def ms_alns_solve(
    solution: Solution,
    instance: InstanceClass,
) -> Solution:
    """使用多起点ALNS算法求解算例"""
    ms_alns_solver = MSALNS(instance)
    best_solution = ms_alns_solver.solve()
    return best_solution
