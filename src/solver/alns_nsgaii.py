import random
import time
from datetime import datetime
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from src.config.prob_config import OptimizeSense
from src.element import (
    InstanceClass,
    Solution,
)
from src.evaluator import Population
from src.solver.nsgaii import NSGAII


class ALNSNSGAII(NSGAII):
    """结合ALNS的NSGA-II算法框架"""

    def __init__(
        self,
        population: "Population",
        instance: "InstanceClass",
        use_adaptive_selection: bool = False,  # 新增参数：是否使用自适应选择
        use_niching: bool = True,  # 新增参数：是否使用小生境技术
    ):
        super().__init__(
            population,
            instance,
            use_adaptive_selection=use_adaptive_selection,
            use_niching=use_niching,
        )

        # ============ 定义评估器 ============
        self.evaluator = population.evaluator

        # ============ 算例参数 ============
        self.comm_coverage_matrix = instance.comm_coverage_matrix
        self.priority_list = instance.priority
        self.travel_time_matrix = instance.travel_time_matrix
        self.id_steiner = list(instance.steiner_ids)
        accessible_flag = instance.accessible
        self.id_base = list(instance.base_ids)
        self.id_demand = list(instance.demand_ids)
        self.id_isolated = [i for i in self.id_demand if not accessible_flag[i]]
        self.id_connected = [i for i in self.id_demand if accessible_flag[i]]
        self.min_comm_time = instance.prob_config.instance_param.min_comm_time

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
        self.travel_cost_matrix = instance.travel_consum_matrix
        self.comm_cost_matrix = instance.comm_consum_matrix

        # ========== 提取算法参数 ==========
        algorithm_config = instance.prob_config.algorithm_config

        # ---------- ALNS基本参数 ----------
        self.alns_iters_per_offspring = algorithm_config.alns_iters_per_offspring
        self.enable_alns_improvement = algorithm_config.enable_alns_improvement

        # ---------- 超体积早停参数 ----------
        self.enable_hv_early_stopping = algorithm_config.enable_hv_early_stopping
        self.hv_improvement_threshold = algorithm_config.hv_improvement_threshold
        self.hv_stagnation_limit = algorithm_config.hv_stagnation_limit

        # ---------- 破坏算子参数 ----------
        self.destroy_degree_min = algorithm_config.destroy_degree_min
        self.destroy_degree_max = algorithm_config.destroy_degree_max

        # 初始化操作算子列表
        self.destroy_operators = [
            "random_removal",
            "worst_removal",
            "heavy_route_removal",
            "coverage_based_removal",
        ]

        self.repair_operators = [
            "greedy_insertion",
            "random_insertion",
            "light_veh_insertion",
            "cost_free_insertion",
        ]

        # 初始化操作算子得分
        self.destroy_operator_scores = np.ones(len(self.destroy_operators))
        self.repair_operator_scores = np.ones(len(self.repair_operators))

        # 操作算子得分遗忘率
        self.decay_rate = algorithm_config.alns_operator_score_decay_rate

    def solve(self) -> Population:
        """执行 NSGA-II 算法,返回最终种群"""

        # 开始计时
        start_time = time.perf_counter()
        # 初始化种群
        current_pop = self._init_population()

        # 初始化HV记录列表
        hv_history = []

        # 初始化停滞计数器
        stagnation_counter = 0

        # 迭代进化
        for iteration in range(self.max_iter):
            # 【新增】动态调整选择压力
            if self.use_adaptive_selection:
                progress = iteration / self.max_iter
                # 线性增长：前期探索(30%)，后期开发(70%)
                self.selection_pressure = (
                    self.min_selection_pressure
                    + (self.max_selection_pressure - self.min_selection_pressure)
                    * progress
                )

            # 记录当前代的HV值 - 使用Population类的内置方法
            current_hv = current_pop.calculate_hypervolume()
            current_hv_info = {
                "iteration": iteration,
                "hypervolume": current_hv,
            }

            # debug: 打印当前迭代的HV值
            # debug: 打印当前迭代的HV值和选择压力
            if self.use_adaptive_selection:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                    f"Iteration {iteration:3d}: Hypervolume = {current_hv:.4f}, "
                    f"Selection Pressure = {self.selection_pressure:.2%}"
                )
            else:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                    f"Iteration {iteration:3d}: Hypervolume = {current_hv:.4f}"
                )
            hv_history.append(current_hv_info)

            # 判断是否早停
            if self.enable_hv_early_stopping:
                if len(hv_history) < 2:
                    continue  # 需要至少两代数据进行比较
                last_hv = hv_history[-2]["hypervolume"]
                current_hv = hv_history[-1]["hypervolume"]
                if current_hv - last_hv < self.hv_improvement_threshold:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
            if stagnation_counter >= self.hv_stagnation_limit:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                    f"连续 {stagnation_counter} 代 HV 改进小于 {self.hv_improvement_threshold}，"
                    f"触发早停机制，终止迭代。"
                )
                break

            # 生成子代
            child_pop = self._generate_offspring(current_pop)

            # 评估子代
            child_pop.evaluate_all()

            # 如果执行alns
            if self.enable_alns_improvement:
                # debug:
                # hv_before_alns = child_pop.calculate_hypervolume()

                # 注：解评估已在alns内部完成
                child_pop = self._alns_improve_offspring(child_pop)

                # debug:
                # hv_after_alns = child_pop.calculate_hypervolume()
                # if hv_after_alns > hv_before_alns:
                #     print(
                #         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                #         f"  ALNS 改进了 HV: "
                #         f"{hv_before_alns:.4f} -> {hv_after_alns:.4f}"
                #     )

            # 精英选择(返回新种群)
            new_pop = self._elitist_selection(current_pop, child_pop)

            # 显式删除旧种群以释放内存
            del current_pop
            del child_pop

            # 更新当前种群
            current_pop = new_pop

            # 添加: 记录当前代所有解的目标函数值
            objectives = np.array([sol.objectives for sol in current_pop.solutions])
            self.generation_objectives.append(objectives)

        # 结束计时
        end_time = time.perf_counter()
        current_pop.solve_time = end_time - start_time

        # 将HV历史记录附加到最终种群
        current_pop.hv_history = hv_history

        return current_pop  # 返回最终的 current_pop

    def _alns_improve_offspring(self, pop: Population):
        """使用ALNS改进子代种群中的每个个体"""
        improved_pop = pop.copy()
        evaluator = self.evaluator

        # 清空本代算子使用统计
        current_gen_stats: Dict[str, Dict[str, List[float]]] = {
            "destroy": {op: [] for op in self.destroy_operators},
            "repair": {op: [] for op in self.repair_operators},
        }

        # 对每个个体进行ALNS改进
        for idx, solution in enumerate(improved_pop.solutions):
            # ✅ 确保初始解已评估
            if (not hasattr(solution, "objectives")) or (
                solution.objectives
                == [
                    float("inf"),
                    float("inf"),
                ]
            ):
                evaluator.sol_evaluate(solution)

            best_solution = solution.copy()
            current_sol = solution.copy()

            # ALNS迭代
            for alns_iter in range(self.alns_iters_per_offspring):
                # TODO: 1. 选择破坏算子、修复算子
                destroy_op, repair_op = self._select_operator()

                # TODO: 2. 破坏、修复当前解
                destroyed_sol, destroyed_task = self._destroy_sol(
                    current_sol.copy(),
                    destroy_op,
                )

                new_sol = self._repair_sol(destroyed_sol, destroyed_task, repair_op)

                # debug: 尝试improve coverage
                new_sol = self._improve_solution_coverage(new_sol)

                # TODO: 3. 评估新解
                evaluator.sol_evaluate(new_sol)

                # TODO: 4. 计算得分
                score = self._calculate_improvement_score(
                    current_sol=current_sol,
                    new_sol=new_sol,
                    best_solution=best_solution,
                    population=pop,  # 传入种群用于非支配判断
                )

                # TODO: 5. 记录算子使用情况和得分
                current_gen_stats["destroy"][destroy_op].append(score)
                current_gen_stats["repair"][repair_op].append(score)

                # TODO: 6. 接受准则（爬山）
                if score > 0:
                    # 接受新解
                    current_sol = new_sol.copy()

                    # 7. 更新最优解
                    if not self._dominates(best_solution, [new_sol]):
                        best_solution = new_sol.copy()

                # 每次破坏修复都更新权重
                self.update_operator_scores(current_gen_stats)

            # 用改进后的最优解替换原解
            improved_pop.solutions[idx] = best_solution

        # 所有子代个体的alns改进循环结束后才更新权重
        # self.update_operator_scores(current_gen_stats)

        return improved_pop

    def _improve_solution_coverage(self, solution: Solution):
        veh_id_list = self.id_vehicle.copy()
        random.shuffle(veh_id_list)

        # 记录，如果是空路径则不必要improve
        for veh_id in veh_id_list:
            current_route = solution.routes[veh_id]
            if len(current_route) <= 2:
                continue  # 空路径，无需改进
            self._improve_coverage(solution, veh_id)

        solution._init_attrs()

        return solution

    def update_operator_scores(
        self,
        current_gen_stats: dict,
    ):
        # 更新破坏算子权重
        for idx, op_name in enumerate(self.destroy_operators):
            scores = current_gen_stats["destroy"][op_name]

            if len(scores) > 0:
                # 计算平均得分
                avg_score = np.mean(scores)

                # 衰减历史权重 + 加入当前得分
                self.destroy_operator_scores[idx] = (
                    self.decay_rate * self.destroy_operator_scores[idx]
                    + (1 - self.decay_rate) * avg_score
                )
            else:
                # 如果本代未使用，仅衰减
                self.destroy_operator_scores[idx] *= self.decay_rate

        # 更新修复算子权重（同理）
        for idx, op_name in enumerate(self.repair_operators):
            scores = current_gen_stats["repair"][op_name]

            if len(scores) > 0:
                avg_score = np.mean(scores)
                self.repair_operator_scores[idx] = (
                    self.decay_rate * self.repair_operator_scores[idx]
                    + (1 - self.decay_rate) * avg_score
                )
            else:
                self.repair_operator_scores[idx] *= self.decay_rate

        # 防止权重过小（设置最小阈值）
        min_weight = 0.1
        self.destroy_operator_scores = np.maximum(
            self.destroy_operator_scores, min_weight
        )
        self.repair_operator_scores = np.maximum(
            self.repair_operator_scores, min_weight
        )

    def _calculate_improvement_score(
        self,
        current_sol: "Solution",
        new_sol: "Solution",
        best_solution: "Solution",
        population: "Population",
    ) -> float:
        """
        计算解的改进得分，用于更新算子权重
        得分策略（分层奖励）：
        - Level 3 (最高): 新解是种群的非支配解 → score = 3.0
        - Level 2 (中等): 新解支配当前最优解 → score = 2.0
        - Level 1 (较低): 新解支配当前解 → score = 1.0
        - Level 0 (最低): 新解无改进 → score = 0.0
        """

        current_elite = population.get_elite_solutions()
        # Level 3: 检查新解是否为非支配解（不被精英集中任何解支配）
        if self._is_non_dominated_by(new_sol, current_elite):
            return 3.0

        # Level 2: 检查是否支配当前最优解
        elif self._dominates(new_sol, [best_solution]):
            return 2.0

        # Level 1: 检查是否支配当前解
        elif self._dominates(new_sol, [current_sol]):
            return 1.0

        # Level 0: 无改进
        else:
            return 0.0

    def _is_non_dominated_by(
        self,
        target_sol: "Solution",
        reference_sol_list: List["Solution"],
    ) -> bool:
        """
        检查目标解是否不被参考解列表中的任何解支配
        返回True表示target_sol是非支配解
        """
        for ref_sol in reference_sol_list:
            if self._dominates(ref_sol, [target_sol]):
                return False  # 被某个参考解支配
        return True  # 不被任何参考解支配

    def _dominates(
        self,
        target_sol: "Solution",
        reference_sol_list: List["Solution"],
    ) -> bool:
        """判断目标解是否支配参考解列表中的所有解"""
        obj_config = self.instance.prob_config.algorithm_config.obj_config

        sense = obj_config.sense  # 目标函数方向
        tolerance = 1e-6  # 容差值

        # 如果参考列表为空,返回True
        if not reference_sol_list:
            return True

        target_obj = np.array(target_sol.objectives)

        # 遍历参考解列表,检查目标解是否支配每一个参考解
        for ref_sol in reference_sol_list:
            ref_obj = np.array(ref_sol.objectives)

            # 计算差值
            diff = target_obj - ref_obj

            # 判断是否在所有目标上不比参考解差,且至少在一个目标上更好
            not_worse = True
            at_least_one_better = False

            for m, s in enumerate(sense):
                if s == OptimizeSense.MINIMIZE:
                    # 最小化: target比ref小超过容差才算更好
                    if diff[m] < -tolerance:
                        at_least_one_better = True
                    elif diff[m] > tolerance:
                        not_worse = False
                        break
                else:  # OptimizeSense.MAXIMIZE
                    # 最大化: target比ref大超过容差才算更好
                    if diff[m] > tolerance:
                        at_least_one_better = True
                    elif diff[m] < -tolerance:
                        not_worse = False
                        break

            # 如果target不支配当前参考解,返回False
            if not (not_worse and at_least_one_better):
                return False

        # target支配所有参考解
        return True

    def _find_best_insertion_position(
        self,
        veh_id: int,
        route: List[int],
        node_id: int,
        covered_tasks: Set[int],
        current_coverage: Dict[Tuple[int, int], Set[int]],
        vehicle_energy_margin: float,
        criterion: str = "distance",
    ) -> Tuple[Optional[int], Optional[float]]:
        """为指定节点找到最优插入位置

        Args:
            veh_id: 车辆ID
            route: 当前路径
            node_id: 要插入的节点ID
            covered_tasks: 该节点覆盖的任务集合
            current_coverage: 当前车辆的覆盖信息
            vehicle_energy_margin: 车辆能量余量
            criterion: 优化准则 ("distance": 最小距离增加, "first": 第一个可行位置)

        Returns:
            (最优位置索引, 对应的评价值), 如果无可行位置则返回 (None, None)
        """
        evaluator = self.evaluator
        best_position = None
        best_value = float("inf")

        # 遍历所有可能的插入位置
        for pos in range(1, len(route)):
            # ========== 快速能耗检查 ==========
            energy_increase = evaluator.calculate_insertion_energy_increase(
                veh_id=veh_id,
                route=route,
                insert_pos=pos,
                insert_node=node_id,
                new_covered_tasks=covered_tasks,
                current_coverage=current_coverage,
            )

            # 能耗不可行，跳过
            if energy_increase > vehicle_energy_margin:
                continue

            # ========== 构造候选路径 ==========
            candidate_route = route[:pos] + [node_id] + route[pos:]

            # ========== 准备覆盖信息 ==========
            temp_coverage = current_coverage.copy()
            temp_coverage[(veh_id, node_id)] = covered_tasks

            # ========== 路径可行性检查 ==========
            # debug:
            # route_feasible, reason = evaluator.is_route_feasible(
            #     veh_id,
            #     candidate_route,
            #     temp_coverage,
            # )
            route_feasible = True

            if route_feasible:
                # 根据优化准则计算评价值
                if criterion == "first":
                    # 找到第一个可行位置即返回
                    return pos, 0.0

                elif criterion == "distance":
                    # 计算插入距离增加
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    dist_increase = (
                        self.distance_matrix[prev_node][node_id]
                        + self.distance_matrix[node_id][next_node]
                        - self.distance_matrix[prev_node][next_node]
                    )

                    if dist_increase < best_value:
                        best_value = dist_increase
                        best_position = pos

                elif criterion == "duration":
                    # 计算插入旅行增加时长
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    travel_time_increase = (
                        self.travel_time_matrix[veh_id][prev_node][node_id]
                        + self.travel_time_matrix[veh_id][node_id][next_node]
                        - self.travel_time_matrix[veh_id][prev_node][next_node]
                    )
                    comm_time_increase = self.min_comm_time
                    time_increase = travel_time_increase + comm_time_increase
                    if time_increase < best_value:
                        best_value = time_increase
                        best_position = pos

            # else:
            #     # 调试信息：插入失败原因
            #     print(
            #         f"{datetime.now()} - [Warning] 插入失败！ 车辆 {veh_id}, "
            #         f"节点 {node_id} 在位置 {pos} 由于: {reason}."
            #     )
            #     raise ValueError(
            #         f"插入失败！ 车辆 {veh_id}, 节点 {node_id} 在位置 {pos} 由于: {reason}."
            #     )

        return best_position, best_value if best_position is not None else None

    def _insert_node_at_position(
        self,
        solution: "Solution",
        veh_id: int,
        node_id: int,
        position: int,
        covered_tasks: Set[int],
        vehicle_energy_margins: Dict[int, float],
        update_margins: bool = True,
    ) -> bool:
        """尝试在指定位置插入节点到解中

        Args:
            solution: 当前解
            veh_id: 车辆ID
            node_id: 要插入的节点ID
            position: 插入位置
            covered_tasks: 该节点覆盖的任务集合
            vehicle_energy_margins: 所有车辆的能量余量字典
            update_margins: 是否更新能量余量

        Returns:
            是否成功插入
        """
        route = solution.routes[veh_id]

        # 提取当前车辆的覆盖信息
        current_coverage = {
            key: value for key, value in solution.coverage.items() if key[0] == veh_id
        }

        # 计算能耗增加
        if update_margins:
            energy_increase = self.evaluator.calculate_insertion_energy_increase(
                veh_id=veh_id,
                route=route,
                insert_pos=position,
                insert_node=node_id,
                new_covered_tasks=covered_tasks,
                current_coverage=current_coverage,
            )

        # 执行插入
        solution.routes[veh_id] = route[:position] + [int(node_id)] + route[position:]
        solution.set_coverage(veh_id, int(node_id), covered_tasks)

        # 更新能量余量
        if update_margins:
            vehicle_energy_margins[veh_id] -= energy_increase

        # # debug: 检查插入后解的可行性
        # route_feasible, reason = self.evaluator.is_route_feasible(
        #     veh_id,
        #     solution.routes[veh_id],
        #     solution.coverage,
        # )

        return True

    def _select_operator(self):
        """根据算子得分选择破坏和修复算子"""
        # 计算选择概率
        destroy_probs = self.destroy_operator_scores / np.sum(
            self.destroy_operator_scores
        )
        repair_probs = self.repair_operator_scores / np.sum(self.repair_operator_scores)

        # 选择算子索引
        destroy_index = np.random.choice(
            len(self.destroy_operators),
            p=destroy_probs,
        )
        repair_index = np.random.choice(
            len(self.repair_operators),
            p=repair_probs,
        )

        return (
            self.destroy_operators[destroy_index],
            self.repair_operators[repair_index],
        )

    def _destroy_sol(
        self,
        solution: "Solution",
        operator_name: str,
    ) -> Tuple["Solution", List[int]]:
        """应用破坏算子"""
        if operator_name not in self.destroy_operators:
            raise ValueError(f"Unknown destroy operator: {operator_name}")

        covered_task: Set[int] = set()
        for cover_set in solution.coverage.values():
            covered_task.update(cover_set)
        covered_task_num = len(covered_task)

        # 确定破坏程度
        destroy_degree = random.uniform(
            self.destroy_degree_min, self.destroy_degree_max
        )
        num_tasks_to_remove = max(0, int(covered_task_num * destroy_degree))

        if num_tasks_to_remove == 0:
            return solution, []

        # 选择破坏算子并应用
        if operator_name == "random_removal":
            return self._random_removal(solution, num_tasks_to_remove)
        elif operator_name == "worst_removal":
            return self._worst_removal(solution, num_tasks_to_remove)
        elif operator_name == "heavy_route_removal":
            return self._heavy_route_removal(solution, num_tasks_to_remove)
        elif operator_name == "coverage_based_removal":
            return self._coverage_based_removal(solution, num_tasks_to_remove)
        else:
            raise ValueError(f"Unhandled destroy operator: {operator_name}")

    def _random_removal(
        self,
        solution: "Solution",
        num_tasks_to_remove: int,
    ) -> Tuple["Solution", List[int]]:
        """随机移除解中的部分元素"""

        # # debug: 保留一份原始solution
        # original_solution = solution.copy()

        destroyed_task: List[int] = []

        while len(destroyed_task) < num_tasks_to_remove:
            # 随机选择一个路径，route
            vehicle_idx = random.randint(0, solution.num_vehicles - 1)
            target_route = solution.routes[vehicle_idx]

            # # debug:
            # origin_route = target_route.copy()
            # origin_coverage = [
            #     {(vehicle_idx, node): solution.coverage[(vehicle_idx, node)]}
            #     for node in target_route
            #     if (vehicle_idx, node) in solution.coverage
            # ]

            # 随机选择一个任务节点进行移除
            if len(target_route) > 2:
                task_idx_in_route = random.randint(1, len(target_route) - 2)
                task_node = target_route[task_idx_in_route]

                # 删除覆盖信息
                if (vehicle_idx, task_node) in solution.coverage:
                    destroyed_task.extend(
                        list(solution.coverage[(vehicle_idx, task_node)])
                    )
                    del solution.coverage[(vehicle_idx, task_node)]

                # 移除任务节点
                target_route.pop(task_idx_in_route)

        solution._init_attrs()  # 重新初始化解的属性

        return solution, destroyed_task

    def _worst_removal(
        self,
        solution: "Solution",
        num_tasks_to_remove: int,
    ) -> Tuple["Solution", List[int]]:
        """移除解中表现最差的部分元素(基于成本收益比)"""

        # # debug: 保留一份原始solution
        # original_solution = solution.copy()

        destroyed_task: List[int] = []

        # 如果没有需要移除的任务,直接返回
        if num_tasks_to_remove == 0:
            return solution, destroyed_task

        # 存储每个访问节点的性价比信息
        visit_performance = (
            []
        )  # [(veh_id, node_idx, node_id, benefit/cost, coverage_set), ...]

        # 遍历所有车辆的路径
        for veh_id in self.id_vehicle:
            route = solution.routes[veh_id]

            # 跳过只有起终点的路径(长度<=2)
            if len(route) <= 2:
                continue

            # 遍历路径中的每个访问节点(排除起点和终点)
            for node_idx in range(1, len(route) - 1):
                node_id = route[node_idx]
                prev_node = route[node_idx - 1]
                next_node = route[node_idx + 1]

                # ========== 计算成本: 删除该节点减少的能量消耗 ==========
                # 原始路径的能量消耗: prev -> node -> next
                original_travel_cost = (
                    self.travel_cost_matrix[veh_id][prev_node][node_id]
                    + self.travel_cost_matrix[veh_id][node_id][next_node]
                )

                # 删除后的能量消耗: prev -> next
                new_travel_cost = self.travel_cost_matrix[veh_id][prev_node][next_node]

                # 行驶成本节省
                travel_cost_saved = original_travel_cost - new_travel_cost

                # 获取该访问的覆盖信息
                coverage_set = solution.get_coverage(veh_id, node_id)

                # 通信成本节省
                comm_cost_saved = self.min_comm_time

                # 总成本节省
                total_cost_saved = travel_cost_saved + comm_cost_saved

                # ========== 计算收益: 该访问覆盖的任务的优先级和 ==========
                benefit = 0.0
                if coverage_set:
                    benefit = sum(
                        self.priority_list[task_id] for task_id in coverage_set
                    )

                # ========== 计算性价比: 收益/成本 ==========
                # 成本节省越大、收益越小,说明该访问性价比越差
                if total_cost_saved > 1e-6:  # 避免除零
                    # 性价比 = 收益 / 成本节省
                    # 性价比越低,说明删除该访问越划算
                    performance_ratio = benefit / total_cost_saved
                else:
                    # 如果删除该节点几乎没有成本节省,设为极大值(不删除)
                    performance_ratio = float("inf")

                # 记录该访问的信息
                visit_performance.append(
                    (veh_id, node_idx, node_id, performance_ratio, coverage_set)
                )

        # 如果没有可移除的节点
        if not visit_performance:
            return solution, destroyed_task

        # ========== 按性价比升序排序(性价比越低越差,优先删除) ==========
        visit_performance.sort(key=lambda x: x[3])  # x[3]是performance_ratio

        # ========== 选择要删除的节点 ==========
        nodes_to_remove = []  # [(veh_id, node_idx, node_id, coverage_set), ...]
        removed_count = 0

        for veh_id, node_idx, node_id, ratio, coverage_set in visit_performance:
            if removed_count >= num_tasks_to_remove:
                break

            # 记录要删除的节点和其覆盖信息
            nodes_to_remove.append((veh_id, node_idx, node_id, coverage_set))

            # 累计已移除的任务数
            if coverage_set:
                removed_count += len(coverage_set)

        # ========== 执行删除操作 ==========
        # 按车辆ID和节点索引分组,方便按倒序删除(避免索引偏移)
        # {veh_id: [(node_idx, node_id, coverage_set), ...]}
        removal_dict: Dict[int, List[Tuple[int, int, Set[int]]]] = {}

        for veh_id, node_idx, node_id, coverage_set in nodes_to_remove:
            if veh_id not in removal_dict:
                removal_dict[veh_id] = []
            removal_dict[veh_id].append((node_idx, node_id, coverage_set))

        # 对每个车辆,按节点索引倒序删除
        for veh_id, remove_list in removal_dict.items():
            # 按node_idx倒序排序
            remove_list.sort(key=lambda x: x[0], reverse=True)

            target_route = solution.routes[veh_id]

            for node_idx, node_id, coverage_set in remove_list:
                # 删除覆盖信息
                if (veh_id, node_id) in solution.coverage:
                    del solution.coverage[(veh_id, node_id)]

                # 记录被破坏的任务
                if coverage_set:
                    destroyed_task.extend(list(coverage_set))

                # 从路径中删除节点
                target_route.pop(node_idx)

        # 重新初始化解的属性
        solution._init_attrs()

        # # debug: 检查是否存在删除了task但路径中仍包含该点的情况
        # included_tasks = solution.get_included_nodes()
        # destroyed_task_set = set(destroyed_task)
        # if included_tasks & destroyed_task_set:
        #     raise ValueError(
        #         f"随机移除错误: 删除的任务 {included_tasks & destroyed_task_set} "
        #         f"仍在路径中 {included_tasks}."
        #     )

        return solution, destroyed_task

    def _heavy_route_removal(
        self,
        solution: "Solution",
        num_tasks_to_remove: int,
    ) -> Tuple["Solution", List[int]]:
        """移除解中持续时间最长的路径"""

        destroyed_task: List[int] = []

        # 如果没有需要移除的任务,直接返回
        if num_tasks_to_remove == 0:
            return solution, destroyed_task

        # ========== 计算每条路径的持续时间 ==========
        route_durations = []  # [(veh_id, duration, route_length), ...]

        for veh_id in self.id_vehicle:
            route = solution.routes[veh_id]

            # 跳过只有起终点的路径
            if len(route) <= 2:
                continue

            # 使用向量化计算路径总时间
            route_array = np.array(route)
            # 计算相邻节点间的旅行时间
            travel_times = self.travel_time_matrix[veh_id][
                route_array[:-1], route_array[1:]
            ]
            total_travel_time = np.sum(travel_times)

            # 计算通信时间(中间节点数量 * 最小通信时间)
            num_visits = len(route) - 2
            total_comm_time = num_visits * self.min_comm_time

            # 总持续时间
            total_duration = total_travel_time + total_comm_time

            route_durations.append((veh_id, total_duration, len(route)))

        # 如果没有可移除的路径
        if not route_durations:
            return solution, destroyed_task

        # ========== 按持续时间降序排序 ==========
        route_durations.sort(key=lambda x: x[1], reverse=True)

        # ========== 逐条处理路径直到满足删除数量 ==========
        removed_count = 0

        for veh_id, duration, route_length in route_durations:
            if removed_count >= num_tasks_to_remove:
                break

            route = solution.routes[veh_id]
            num_visits = len(route) - 2  # 排除起终点

            # ========== 情况1: 路径任务数不超过剩余需求,直接删除整条路径 ==========
            remaining_to_remove = num_tasks_to_remove - removed_count

            if num_visits <= remaining_to_remove:
                # 收集所有访问的覆盖任务
                for node_idx in range(1, len(route) - 1):
                    node_id = route[node_idx]
                    coverage_set = solution.get_coverage(veh_id, node_id)

                    if coverage_set:
                        destroyed_task.extend(list(coverage_set))
                        removed_count += len(coverage_set)

                    # 删除覆盖信息
                    if (veh_id, node_id) in solution.coverage:
                        del solution.coverage[(veh_id, node_id)]

                # 将路径重置为起终点
                solution.routes[veh_id] = [route[0], route[-1]]

            # ========== 情况2: 路径任务数超过剩余需求,按收益选择性删除 ==========
            else:
                # 计算每个访问的收益
                visit_performance = (
                    []
                )  # [(node_idx, node_id, benefit, coverage_set), ...]

                for node_idx in range(1, len(route) - 1):
                    node_id = route[node_idx]
                    coverage_set = solution.get_coverage(veh_id, node_id)

                    # 计算收益
                    benefit = 0.0
                    if coverage_set:
                        # 向量化计算优先级和
                        coverage_array = np.array(list(coverage_set))
                        benefit = np.sum(
                            [self.priority_list[i] for i in coverage_array]
                        )

                    visit_performance.append((node_idx, node_id, benefit, coverage_set))

                # ========== 按收益升序排序(收益低的优先删除) ==========
                visit_performance.sort(key=lambda x: x[2])

                # ========== 选择要删除的访问 ==========
                nodes_to_remove = []
                for node_idx, node_id, benefit, coverage_set in visit_performance:
                    if removed_count >= num_tasks_to_remove:
                        break

                    nodes_to_remove.append((node_idx, node_id, coverage_set))

                    if coverage_set:
                        removed_count += len(coverage_set)

                # ========== 按索引倒序删除(避免索引偏移) ==========
                nodes_to_remove.sort(key=lambda x: x[0], reverse=True)

                for node_idx, node_id, coverage_set in nodes_to_remove:
                    # 删除覆盖信息
                    if (veh_id, node_id) in solution.coverage:
                        del solution.coverage[(veh_id, node_id)]

                    # 记录被破坏的任务
                    if coverage_set:
                        destroyed_task.extend(list(coverage_set))

                    # 从路径中删除节点
                    route.pop(node_idx)

        # 重新初始化解的属性
        solution._init_attrs()

        # # debug: 检查是否存在删除了task但路径中仍包含该点的情况
        # included_tasks = solution.get_included_nodes()
        # destroyed_task_set = set(destroyed_task)
        # if included_tasks & destroyed_task_set:
        #     raise ValueError(
        #         f"随机移除错误: 删除的任务 {destroyed_task_set} 仍在路径中 {included_tasks}."
        #     )

        return solution, destroyed_task

    def _coverage_based_removal(
        self,
        solution: "Solution",
        num_tasks_to_remove: int,
    ) -> Tuple["Solution", List[int]]:
        """基于可被覆盖情况移除解中的部分元素

        删除策略:
        1. 删除可被其他已访问点完全覆盖的访问
        2. 寻找能覆盖多个已访问点的Steiner点或未访问需求点,删除被覆盖的访问
        3. 如果还未达到删除数量,随机删除剩余访问
        """
        destroyed_task: List[int] = []

        # 如果没有需要移除的任务,直接返回
        if num_tasks_to_remove == 0:
            return solution, destroyed_task

        # ========== 收集所有访问信息 ==========
        visits: List[Tuple[int, int, int, Set[int]]] = (
            []
        )  # [(veh_id, node_idx, node_id, coverage_set), ...]

        for veh_id in self.id_vehicle:
            route = solution.routes[veh_id]

            # 跳过只有起终点的路径
            if len(route) <= 2:
                continue

            # 收集路径中的访问信息
            for node_idx in range(1, len(route) - 1):
                node_id = route[node_idx]
                coverage_set = solution.get_coverage(veh_id, node_id)

                if coverage_set:
                    visits.append((veh_id, node_idx, node_id, coverage_set))

        # 如果没有可删除的访问
        if not visits:
            return solution, destroyed_task

        # 记录要删除的访问 [(veh_id, node_idx, node_id, coverage_set), ...]
        nodes_to_remove: List[Tuple[int, int, int, Set[int]]] = []
        removed_coverage_count = 0

        # ========== 第一轮: 删除可被其他已访问点完全覆盖的访问 ==========
        # 构建所有已访问点的覆盖能力字典 {node_id: {可覆盖的任务集合}}
        visited_nodes_coverage: Dict[int, Set[int]] = {}

        for veh_id, node_idx, node_id, coverage_set in visits:
            # 获取该访问点的覆盖能力(包括当前已覆盖和可以覆盖但未覆盖的)
            covered_mask = self.comm_coverage_matrix[veh_id][node_id]
            all_coverable_tasks = set(np.where(covered_mask)[0].tolist())
            visited_nodes_coverage[node_id] = all_coverable_tasks

        # 检查每个访问,其覆盖集是否可被其他已访问点完全覆盖
        for veh_id, node_idx, node_id, coverage_set in visits:
            if removed_coverage_count >= num_tasks_to_remove:
                break

            # 检查该访问的覆盖集是否可被其他访问点覆盖
            other_nodes_coverage = set()
            for other_node_id, other_coverage_ability in visited_nodes_coverage.items():
                if other_node_id != node_id:
                    other_nodes_coverage |= other_coverage_ability

            # 如果该访问的覆盖集完全包含在其他访问的覆盖能力中
            if coverage_set.issubset(other_nodes_coverage):
                nodes_to_remove.append((veh_id, node_idx, node_id, coverage_set))
                removed_coverage_count += len(coverage_set)

        # ========== 第二轮: 寻找能覆盖多个已访问点的Steiner点或未访问需求点 ==========
        if removed_coverage_count < num_tasks_to_remove:
            # 收集所有已访问的节点集合
            visited_node_ids = set(node_id for _, _, node_id, _ in visits)

            # 候选替代点: Steiner点 + 未访问的需求点
            candidate_nodes = set(self.id_steiner) | (
                set(self.id_demand) - visited_node_ids
            )

            # 评估每个候选点能替代的访问
            replacement_options: List[
                Tuple[int, int, List[Tuple[int, int, int, Set[int]]]]
            ] = []  # [(candidate_node, veh_id, [可替代的访问列表]), ...]

            for candidate_node in candidate_nodes:
                for veh_id in self.id_vehicle:
                    # 检查车辆可达性
                    is_ground = veh_id in self.id_Ground
                    if is_ground and not self.instance.accessible[candidate_node]:
                        continue

                    # 获取该候选点的覆盖能力
                    covered_mask = self.comm_coverage_matrix[veh_id][candidate_node]
                    candidate_coverage = set(np.where(covered_mask)[0].tolist())

                    # 找出该候选点能替代的访问
                    replaceable_visits: List[Tuple[int, int, int, Set[int]]] = []
                    for visit_info in visits:
                        v_id, n_idx, n_id, cov_set = visit_info

                        # 跳过已标记删除的访问
                        if visit_info in nodes_to_remove:
                            continue

                        # 如果候选点能覆盖该访问的所有任务
                        if cov_set.issubset(candidate_coverage):
                            replaceable_visits.append(visit_info)

                    # 只记录能替代多个访问的候选点
                    if len(replaceable_visits) >= 2:
                        replacement_options.append(
                            (candidate_node, veh_id, replaceable_visits)
                        )

            # 按可替代访问数量降序排序
            replacement_options.sort(key=lambda x: len(x[2]), reverse=True)

            # 贪婪选择替代方案
            for candidate_node, veh_id, replaceable_visits in replacement_options:
                if removed_coverage_count >= num_tasks_to_remove:
                    break

                # 将可替代的访问加入删除列表
                for visit_info in replaceable_visits:
                    if visit_info not in nodes_to_remove:
                        v_id, n_idx, n_id, cov_set = visit_info
                        nodes_to_remove.append(visit_info)
                        removed_coverage_count += len(cov_set)

                        if removed_coverage_count >= num_tasks_to_remove:
                            break

        # ========== 第三轮: 如果还未达到删除数量,随机删除 ==========
        if removed_coverage_count < num_tasks_to_remove:
            # 收集未被标记删除的访问
            remaining_visits = [
                visit for visit in visits if visit not in nodes_to_remove
            ]

            # 随机打乱
            random.shuffle(remaining_visits)

            # 随机选择删除
            for visit_info in remaining_visits:
                if removed_coverage_count >= num_tasks_to_remove:
                    break

                veh_id, node_idx, node_id, coverage_set = visit_info
                nodes_to_remove.append(visit_info)
                removed_coverage_count += len(coverage_set)

        # ========== 执行删除操作 ==========
        # 按车辆ID和节点索引分组
        removal_dict: Dict[int, List[Tuple[int, int, Set[int]]]] = {}

        for veh_id, node_idx, node_id, coverage_set in nodes_to_remove:
            if veh_id not in removal_dict:
                removal_dict[veh_id] = []
            removal_dict[veh_id].append((node_idx, node_id, coverage_set))

        # 对每个车辆,按节点索引倒序删除
        for veh_id, remove_list in removal_dict.items():
            # 按node_idx倒序排序
            remove_list.sort(key=lambda x: x[0], reverse=True)

            target_route = solution.routes[veh_id]

            for node_idx, node_id, coverage_set in remove_list:
                # 删除覆盖信息
                if (veh_id, node_id) in solution.coverage:
                    del solution.coverage[(veh_id, node_id)]

                # 记录被破坏的任务
                if coverage_set:
                    destroyed_task.extend(list(coverage_set))

                # 从路径中删除节点
                target_route.pop(node_idx)

        # 重新初始化解的属性
        solution._init_attrs()

        # # debug: 检查是否存在删除了task但路径中仍包含该点的情况
        # included_tasks = solution.get_included_nodes()
        # destroyed_task_set = set(destroyed_task)
        # if included_tasks & destroyed_task_set:
        #     raise ValueError(
        #         f"随机移除错误: 删除的任务 {destroyed_task_set} 仍在路径中 {included_tasks}."
        #     )

        return solution, destroyed_task

    def _repair_sol(
        self,
        solution: "Solution",
        destroyed_task: List[int],
        operator_name: str,
    ) -> "Solution":
        """应用修复算子"""
        if operator_name not in self.repair_operators:
            raise ValueError(f"Unknown repair operator: {operator_name}")

        if not destroyed_task:
            return solution  # 无需修复

        # debug: 记录算子时间
        start_time = time.perf_counter()

        # 选择修复算子并应用
        if operator_name == "greedy_insertion":
            solution = self._greedy_insertion(solution, destroyed_task)
        elif operator_name == "random_insertion":
            solution = self._random_insertion(solution, destroyed_task)
        elif operator_name == "light_veh_insertion":
            solution = self._light_veh_insertion(solution, destroyed_task)
        elif operator_name == "cost_free_insertion":
            solution = self._cost_free_insertion(solution, destroyed_task)
        else:
            raise ValueError(f"Unhandled repair operator: {operator_name}")

        # debug: 记录算子时间
        end_time = time.perf_counter()
        op_time = end_time - start_time
        if op_time > 0.01:
            print(f"算子；{operator_name}耗时超过0.1秒，实际耗时: {op_time:.4f} 秒")

        return solution

    def _greedy_insertion(
        self,
        solution: "Solution",
        destroyed_task: List[int],
    ) -> "Solution":
        """贪婪插入被移除的任务"""

        # 如果没有需要插入的任务,直接返回
        if not destroyed_task:
            return solution

        # 将待插入任务转换为集合,方便快速查找
        task_set = set(destroyed_task)

        # 构建评价器
        evaluator = self.evaluator

        # 提取实例参数
        comm_coverage_matrix = self.comm_coverage_matrix
        priority_list = self.priority_list
        accessible_list = self.instance.accessible

        # ========== 预计算每个车辆的能量余量 ==========
        vehicle_energy_margins = {}
        for veh_id in self.id_vehicle:
            route = solution.routes[veh_id]
            route_coverage = {
                key: value
                for key, value in solution.coverage.items()
                if key[0] == veh_id
            }
            vehicle_energy_margins[veh_id] = evaluator.calculate_energy_margin(
                veh_id, route, route_coverage
            )

        # 计算每个待分配任务的可覆盖信息
        coverage_info = {}

        for veh_id in self.id_vehicle:
            # 判断是否需要检查可达性(地面车辆需要)
            check_accessible = veh_id in self.id_Ground

            # 遍历所有待插入的任务
            for task in destroyed_task:
                # 跳过地面车辆不可达点
                if check_accessible and not accessible_list[task]:
                    continue

                # 找出该任务可以覆盖的其他任务
                covered_mask = comm_coverage_matrix[veh_id][task]
                all_covered_tasks = np.where(covered_mask)[0].tolist()

                # 只保留在 task_set 中的任务
                covered_tasks = [t for t in all_covered_tasks if t in task_set]

                # 计算覆盖收益
                coverage_benefit = sum(priority_list[t] for t in covered_tasks)

                # 只记录有价值的访问
                if coverage_benefit > 0:
                    coverage_info[(veh_id, task)] = {
                        "count": len(covered_tasks),
                        "benefit": coverage_benefit,
                        "covered_tasks": covered_tasks,
                    }

        # 计算Steiner节点信息
        for veh_id in self.id_vehicle:
            for steiner in self.id_steiner:
                covered_mask = comm_coverage_matrix[veh_id][steiner]
                all_covered_tasks = np.where(covered_mask)[0].tolist()
                covered_tasks = [t for t in all_covered_tasks if t in task_set]
                coverage_benefit = sum(priority_list[t] for t in covered_tasks)

                if coverage_benefit > 0:
                    coverage_info[(veh_id, steiner)] = {
                        "count": len(covered_tasks),
                        "benefit": coverage_benefit,
                        "covered_tasks": covered_tasks,
                    }

        # 提取当前解覆盖的点
        already_covered_tasks = solution.get_included_nodes()
        # 贪婪插入循环
        while task_set and coverage_info:
            # 贪婪选择:选择收益最大的覆盖方案
            best_plan = max(
                coverage_info.keys(),
                key=lambda x: coverage_info[x]["benefit"],
            )

            # 提取选择的信息
            veh_id, node_id = best_plan

            # 检查这个plan是否还能用
            if node_id in already_covered_tasks:
                del coverage_info[best_plan]
                continue
            completed_tasks = coverage_info[best_plan]["covered_tasks"]

            # 提取实际上的可覆盖情况
            true_covered_tasks = set(completed_tasks) - already_covered_tasks

            if not true_covered_tasks:
                # 如果没有实际覆盖任务,则移除该方案并继续
                del coverage_info[best_plan]
                continue

            # 获取当前路径
            target_route = solution.routes[veh_id]

            # 提取当前车辆的覆盖信息
            current_coverage = {
                key: solution.get_coverage(key[0], key[1])
                for key in solution.coverage.keys()
                if key[0] == veh_id
            }

            # 使用内置函数寻找最优插入位置
            best_position, min_cost_increase = self._find_best_insertion_position(
                veh_id=veh_id,
                route=target_route,
                node_id=node_id,
                covered_tasks=true_covered_tasks,
                current_coverage=current_coverage,
                vehicle_energy_margin=vehicle_energy_margins[veh_id],
                criterion="distance",
            )

            # 如果找到可行的插入位置
            if best_position is not None:
                # 使用内置函数执行插入
                self._insert_node_at_position(
                    solution=solution,
                    veh_id=veh_id,
                    node_id=node_id,
                    position=best_position,
                    covered_tasks=true_covered_tasks,
                    vehicle_energy_margins=vehicle_energy_margins,
                    update_margins=True,
                )

                already_covered_tasks.update(true_covered_tasks)

                # 从待插入任务中移除已完成任务
                for t in completed_tasks:
                    task_set.discard(t)

            # 更新 coverage_info
            keys_to_remove = []
            for key in coverage_info.keys():
                veh_id_key, node_id_key = key
                covered_tasks_key = coverage_info[key]["covered_tasks"]

                # 情况1: 访问的需求节点已被覆盖
                if (node_id_key in self.id_demand) and (node_id_key not in task_set):
                    keys_to_remove.append(key)
                # 情况2: 访问的Steiner节点已被使用
                elif (node_id in self.id_steiner) and (node_id_key == node_id):
                    keys_to_remove.append(key)
                # 情况3: 更新仍有效的覆盖信息
                else:
                    updated_covered_tasks = [
                        t for t in covered_tasks_key if t in task_set
                    ]

                    if not updated_covered_tasks:
                        keys_to_remove.append(key)
                    else:
                        coverage_info[key]["covered_tasks"] = updated_covered_tasks
                        coverage_info[key]["count"] = len(updated_covered_tasks)
                        coverage_info[key]["benefit"] = sum(
                            priority_list[t] for t in updated_covered_tasks
                        )

            # 删除无效的覆盖方案
            for key in keys_to_remove:
                del coverage_info[key]

        # 重新初始化解的属性
        solution._init_attrs()

        return solution

    def _random_insertion(
        self,
        solution: "Solution",
        destroyed_task: List[int],
    ) -> "Solution":
        """随机插入被移除的任务（优化版）"""

        # 如果没有需要插入的任务,直接返回
        if not destroyed_task:
            return solution

        # 将待插入任务转换为集合
        task_set = set(destroyed_task)

        # 构建评价器
        evaluator = self.evaluator

        # 提取实例参数（避免重复访问）
        comm_coverage_matrix = self.comm_coverage_matrix
        accessible_list = self.instance.accessible

        # ========== 预计算每个车辆的能量余量 ==========
        vehicle_energy_margins = {}
        for veh_id in self.id_vehicle:
            route = solution.routes[veh_id]
            route_coverage = {
                key: value
                for key, value in solution.coverage.items()
                if key[0] == veh_id
            }
            vehicle_energy_margins[veh_id] = evaluator.calculate_energy_margin(
                veh_id, route, route_coverage
            )

        # ========== 预分组：按可达性分类任务和车辆 ==========
        # 地面车辆只能访问可达任务
        accessible_tasks = [t for t in destroyed_task if accessible_list[t]]
        inaccessible_tasks = [t for t in destroyed_task if not accessible_list[t]]

        # 构建任务到可用车辆的映射（避免每次都重新计算）
        task_to_vehicles = {}
        for task in accessible_tasks:
            # 地面车辆和无人机都可访问
            task_to_vehicles[task] = [
                v for v in self.id_vehicle if vehicle_energy_margins[v] > 0
            ]
        for task in inaccessible_tasks:
            # 只有无人机可访问
            task_to_vehicles[task] = [
                v for v in self.id_Drone if vehicle_energy_margins[v] > 0
            ]

        # ========== 随机打乱任务顺序 ==========
        random_task_list = list(task_set)
        random.shuffle(random_task_list)

        # ========== 随机插入循环 ==========
        for task in random_task_list:
            # 跳过已被覆盖的任务
            if task not in task_set:
                continue

            # ========== 获取可用车辆列表 ==========
            candidate_vehicles = task_to_vehicles.get(task, [])

            # 过滤掉能量不足的车辆（动态更新）
            candidate_vehicles = [
                v for v in candidate_vehicles if vehicle_energy_margins[v] > 0
            ]

            if not candidate_vehicles:
                continue

            # ========== 随机选择车辆 ==========
            veh_id = random.choice(candidate_vehicles)
            target_route = solution.routes[veh_id]

            # ========== 预计算覆盖信息（只计算一次） ==========
            covered_mask = comm_coverage_matrix[veh_id][task]
            all_covered_tasks = np.where(covered_mask)[0].tolist()
            covered_tasks = [t for t in all_covered_tasks if t in task_set]

            # 如果没有覆盖任何待分配任务，跳过
            if not covered_tasks:
                continue

            # 提取当前车辆的覆盖信息（缓存）
            current_coverage = {
                key: value
                for key, value in solution.coverage.items()
                if key[0] == veh_id
            }

            # ========== 生成并随机打乱插入位置，尝试插入 ==========
            inserted = False

            # 生成随机位置序列
            possible_positions = list(range(1, len(target_route)))
            random.shuffle(possible_positions)

            # 尝试每个随机位置
            for pos in possible_positions:
                # 使用内置函数检查该位置是否可行
                check_pos, _ = self._find_best_insertion_position(
                    veh_id=veh_id,
                    route=target_route,
                    node_id=task,
                    covered_tasks=set(covered_tasks),
                    current_coverage=current_coverage,
                    vehicle_energy_margin=vehicle_energy_margins[veh_id],
                    criterion="first",  # 找到第一个可行位置即返回
                )

                # 如果当前位置可行，执行插入
                if check_pos == pos:
                    self._insert_node_at_position(
                        solution=solution,
                        veh_id=veh_id,
                        node_id=task,
                        position=pos,
                        covered_tasks=set(covered_tasks),
                        vehicle_energy_margins=vehicle_energy_margins,
                        update_margins=True,
                    )

                    # 移除已覆盖任务
                    for t in covered_tasks:
                        task_set.discard(t)

                    inserted = True
                    break  # 成功插入，处理下一个任务

            # 如果未成功插入，从集合中移除该任务
            if not inserted:
                task_set.discard(task)

        # 重新初始化解的属性
        solution._init_attrs()

        return solution

    def _light_veh_insertion(
        self,
        solution: "Solution",
        destroyed_task: List[int],
    ) -> "Solution":
        """优先插入路径最短的车辆"""

        # 如果没有需要插入的任务,直接返回
        if not destroyed_task:
            return solution

        # 将待插入任务转换为集合,方便快速查找
        task_set = set(destroyed_task)

        # 构建评价器
        evaluator = self.evaluator

        # 提取实例参数
        comm_coverage_matrix = self.comm_coverage_matrix
        priority_list = self.priority_list
        accessible_list = self.instance.accessible

        # ========== 预计算每个车辆的能量余量 ==========
        vehicle_energy_margins = {}
        for veh_id in self.id_vehicle:
            route = solution.routes[veh_id]
            route_coverage = {
                key: value
                for key, value in solution.coverage.items()
                if key[0] == veh_id
            }
            vehicle_energy_margins[veh_id] = evaluator.calculate_energy_margin(
                veh_id, route, route_coverage
            )

        # 计算每个待分配任务的可覆盖信息
        coverage_info = {}

        # 计算每个路径的长度
        veh_route_lengths = {}

        for veh_id in self.id_vehicle:
            # 计算车辆路径长度信息
            route_distance = evaluator._calculate_route_distance(
                solution.routes[veh_id]
            )
            veh_route_lengths[veh_id] = route_distance

            # 判断是否需要检查可达性(地面车辆需要)
            check_accessible = veh_id in self.id_Ground

            # 遍历所有待插入的任务
            for task in destroyed_task:
                # 跳过地面车辆不可达点
                if check_accessible and not accessible_list[task]:
                    continue

                # 找出该任务可以覆盖的其他任务
                covered_mask = comm_coverage_matrix[veh_id][task]
                all_covered_tasks = np.where(covered_mask)[0].tolist()

                # 只保留在 task_set 中的任务
                covered_tasks = [t for t in all_covered_tasks if t in task_set]

                # 计算覆盖收益
                coverage_benefit = sum(priority_list[t] for t in covered_tasks)

                # 只记录有价值的访问
                if coverage_benefit > 0:
                    coverage_info[(veh_id, task)] = {
                        "count": len(covered_tasks),
                        "benefit": coverage_benefit,
                        "covered_tasks": covered_tasks,
                    }

        # 计算Steiner节点信息
        for veh_id in self.id_vehicle:
            for steiner in self.id_steiner:
                covered_mask = comm_coverage_matrix[veh_id][steiner]
                all_covered_tasks = np.where(covered_mask)[0].tolist()
                covered_tasks = [t for t in all_covered_tasks if t in task_set]
                coverage_benefit = sum(priority_list[t] for t in covered_tasks)

                if coverage_benefit > 0:
                    coverage_info[(veh_id, steiner)] = {
                        "count": len(covered_tasks),
                        "benefit": coverage_benefit,
                        "covered_tasks": covered_tasks,
                    }

        # 提取当前解覆盖的点
        already_covered_tasks = solution.get_included_nodes()

        # 按路径长度升序排序车辆
        sorted_vehicles = sorted(
            self.id_vehicle,
            key=lambda v: veh_route_lengths[v],
        )

        # 按照路径长度循环插入
        while task_set and coverage_info:
            for veh_id in sorted_vehicles:
                if not task_set or not coverage_info:
                    break

                # 筛选出该车辆相关的覆盖方案
                veh_coverage_plans = {
                    key: value
                    for key, value in coverage_info.items()
                    if key[0] == veh_id
                }

                if not veh_coverage_plans:
                    continue

                # 贪婪选择:选择收益最大的覆盖方案
                best_plan = max(
                    veh_coverage_plans.keys(),
                    key=lambda x: veh_coverage_plans[x]["benefit"],
                )

                # 提取选择的信息
                _, node_id = best_plan

                # 检查这个plan是否还能用
                if node_id in already_covered_tasks:
                    del coverage_info[best_plan]
                    continue
                completed_tasks = coverage_info[best_plan]["covered_tasks"]

                # 提取实际上的可覆盖情况
                true_covered_tasks = set(completed_tasks) - already_covered_tasks

                if not true_covered_tasks:
                    # 如果没有实际覆盖任务,则移除该方案并继续
                    del coverage_info[best_plan]
                    continue

                # 获取当前路径
                target_route = solution.routes[veh_id]

                # 提取当前车辆的覆盖信息
                current_coverage = {
                    key: solution.get_coverage(key[0], key[1])
                    for key in solution.coverage.keys()
                    if key[0] == veh_id
                }

                # 使用内置函数寻找最优插入位置
                best_position, min_cost_increase = self._find_best_insertion_position(
                    veh_id=veh_id,
                    route=target_route,
                    node_id=node_id,
                    covered_tasks=true_covered_tasks,
                    current_coverage=current_coverage,
                    vehicle_energy_margin=vehicle_energy_margins[veh_id],
                    criterion="distance",
                )

                # 如果找到可行的插入位置
                if best_position is not None:
                    # 使用内置函数执行插入
                    self._insert_node_at_position(
                        solution=solution,
                        veh_id=veh_id,
                        node_id=node_id,
                        position=best_position,
                        covered_tasks=true_covered_tasks,
                        vehicle_energy_margins=vehicle_energy_margins,
                        update_margins=True,
                    )

                    already_covered_tasks.update(true_covered_tasks)
                    # 从待插入任务中移除已完成任务
                    for t in completed_tasks:
                        task_set.discard(t)
                # 更新 coverage_info
                keys_to_remove = []
                for key in coverage_info.keys():
                    veh_id_key, node_id_key = key
                    covered_tasks_key = coverage_info[key]["covered_tasks"]

                    # 情况1: 访问的需求节点已被覆盖
                    if (node_id_key in self.id_demand) and (
                        node_id_key not in task_set
                    ):
                        keys_to_remove.append(key)
                    # 情况2: 访问的Steiner节点已被使用
                    elif (node_id in self.id_steiner) and (node_id_key == node_id):
                        keys_to_remove.append(key)
                    # 情况3: 更新仍有效的覆盖信息
                    else:
                        updated_covered_tasks = [
                            t for t in covered_tasks_key if t in task_set
                        ]

                        if not updated_covered_tasks:
                            keys_to_remove.append(key)
                        else:
                            coverage_info[key]["covered_tasks"] = updated_covered_tasks
                            coverage_info[key]["count"] = len(updated_covered_tasks)
                            coverage_info[key]["benefit"] = sum(
                                priority_list[t] for t in updated_covered_tasks
                            )
                # 删除无效的覆盖方案
                for key in keys_to_remove:
                    del coverage_info[key]

                # 更新车辆路径长度排序
                route_distance = evaluator._calculate_route_distance(
                    solution.routes[veh_id]
                )
                veh_route_lengths[veh_id] = route_distance
                sorted_vehicles = sorted(
                    self.id_vehicle,
                    key=lambda v: veh_route_lengths[v],
                )
                break

        # 重新初始化解的属性
        solution._init_attrs()
        return solution

    def _cost_free_insertion(
        self,
        solution: "Solution",
        destroyed_task: List[int],
    ) -> "Solution":
        """仅插入不增加成本的任务"""

        evaluator = self.evaluator

        task_num = len(destroyed_task)

        included_tasks = solution.get_included_nodes()

        # ========== 预计算每个车辆的能量余量 ==========
        vehicle_energy_margins = {}
        for veh_id in self.id_vehicle:
            route = solution.routes[veh_id]
            route_coverage = {
                key: value
                for key, value in solution.coverage.items()
                if key[0] == veh_id
            }
            vehicle_energy_margins[veh_id] = evaluator.calculate_energy_margin(
                veh_id, route, route_coverage
            )

        # # 无视破坏的点，直接从所有未执行任务中进行分析
        # all_demand = set(self.id_demand)
        # uncovered_tasks = set(all_demand - included_tasks)
        uncovered_tasks = set(destroyed_task)

        # 提取所有路径的持续时长
        veh_route_durations = {}
        max_duration = 0.0
        for veh_id in self.id_vehicle:
            route_duration = evaluator._calculate_route_duration(
                solution.routes[veh_id],
                solution.coverage,
                veh_id,
            )
            veh_route_durations[veh_id] = route_duration
            if route_duration > max_duration:
                max_duration = route_duration

        # 计算无成本可增加的time
        veh_cost_free_time = {}
        for veh_id in self.id_vehicle:
            if veh_route_durations[veh_id] < max_duration:
                available_time = max_duration - veh_route_durations[veh_id]
                veh_cost_free_time[veh_id] = available_time
            else:
                veh_cost_free_time[veh_id] = 0.0

        # 初始化覆盖信息方案
        coverage_plan = {}
        for veh_id in self.id_vehicle:
            if veh_id in self.id_Ground:
                check_accessible = True
            else:
                check_accessible = False

            # 遍历所有未覆盖任务
            for task in uncovered_tasks:
                # 跳过地面车辆不可达点
                if check_accessible and not self.instance.accessible[task]:
                    continue

                # 获取该任务可以覆盖的其它任务
                covered_mask = self.comm_coverage_matrix[veh_id][task]
                all_covered_tasks = np.where(covered_mask)[0].tolist()

                # 保留在task_to_assign中的任务
                covered_tasks = [t for t in all_covered_tasks if t in uncovered_tasks]

                # 计算覆盖收益
                coverage_benefit = sum(self.priority_list[t] for t in covered_tasks)

                # 判断是否有价值
                if coverage_benefit == 0:
                    continue
                else:
                    coverage_plan[(veh_id, task)] = {
                        "benefit": coverage_benefit,
                        "covered_tasks": covered_tasks,
                    }

            # 遍历未使用的Steiner节点
            available_steiner = set(self.id_steiner) - included_tasks

            for steiner in available_steiner:
                # 找出该Steiner节点可以覆盖的任务（布尔索引）
                covered_mask = self.comm_coverage_matrix[veh_id][steiner]
                # 获取覆盖任务的索引列表
                all_covered_tasks = np.where(covered_mask)[0].tolist()
                # 只保留在 task_to_assign 中的任务
                covered_tasks = [t for t in all_covered_tasks if t in uncovered_tasks]
                # 计算覆盖收益（只统计 task_to_assign 中任务的优先级）
                coverage_benefit = sum(self.priority_list[t] for t in covered_tasks)

                # 判断这个访问是否有价值，如果完全没价值直接跳过
                if coverage_benefit == 0:
                    continue
                else:
                    # 存储所有信息
                    coverage_plan[(veh_id, steiner)] = {
                        "benefit": coverage_benefit,  # 覆盖收益
                        "covered_tasks": covered_tasks,  # 覆盖的任务列表
                    }

        # 选择冗余量最大的车，然后计算插入距离，如果距离最低的可以实现cost free插入，则执行
        still_available = [False for i in self.id_vehicle]
        for veh_id in self.id_vehicle:
            if veh_cost_free_time[veh_id] > self.min_comm_time:
                still_available[veh_id] = True

        while any(still_available) and coverage_plan:
            # 选择冗余量最大的车辆
            veh_id = max(
                self.id_vehicle,
                key=lambda v: veh_cost_free_time[v] if still_available[v] else -1,
            )

            # 提取它相关的coverage_plan
            veh_coverage_plans = {
                key: value for key, value in coverage_plan.items() if key[0] == veh_id
            }

            if not veh_coverage_plans:
                still_available[veh_id] = False
                continue

            # 贪婪选择:选择收益最大的覆盖方案
            best_plan = max(
                veh_coverage_plans.keys(),
                key=lambda x: veh_coverage_plans[x]["benefit"],
            )
            # 提取选择的信息
            _, node_id = best_plan
            completed_tasks = coverage_plan[best_plan]["covered_tasks"]
            # 计算最好的插入位置
            best_position, min_duration_increase = self._find_best_insertion_position(
                veh_id=veh_id,
                route=solution.routes[veh_id],
                node_id=node_id,
                covered_tasks=set(completed_tasks),
                current_coverage={
                    key: solution.get_coverage(key[0], key[1])
                    for key in solution.coverage.keys()
                    if key[0] == veh_id
                },
                vehicle_energy_margin=vehicle_energy_margins[veh_id],
                criterion="duration",
            )

            # 如果找到可行的插入位置
            if best_position is not None and min_duration_increase is not None:
                if min_duration_increase < veh_cost_free_time[veh_id]:
                    # 使用内置函数执行插入
                    self._insert_node_at_position(
                        solution=solution,
                        veh_id=veh_id,
                        node_id=node_id,
                        position=best_position,
                        covered_tasks=set(completed_tasks),
                        vehicle_energy_margins=vehicle_energy_margins,
                        update_margins=True,
                    )

                    included_tasks.update(completed_tasks)
                    # 从待插入任务中移除已完成任务
                    for t in completed_tasks:
                        uncovered_tasks.discard(t)

                    # 更新车辆的冗余时间
                    veh_cost_free_time[veh_id] -= min_duration_increase
                    if veh_cost_free_time[veh_id] <= self.min_comm_time:
                        still_available[veh_id] = False

                else:
                    coverage_plan.pop(best_plan)

            # 更新coverage_plan
            keys_to_remove = []
            for key in coverage_plan.keys():
                veh_id_key, node_id_key = key
                covered_tasks_key = coverage_plan[key]["covered_tasks"]

                # 情况1: 访问的需求节点已被覆盖
                if (node_id_key in self.id_demand) and (
                    node_id_key not in uncovered_tasks
                ):
                    keys_to_remove.append(key)
                # 情况2: 访问的Steiner节点已被使用
                elif (node_id in self.id_steiner) and (node_id_key == node_id):
                    keys_to_remove.append(key)
                # 情况3: 更新仍有效的覆盖信息
                else:
                    updated_covered_tasks = [
                        t for t in covered_tasks_key if t in uncovered_tasks
                    ]

                    if not updated_covered_tasks:
                        keys_to_remove.append(key)
                    else:
                        coverage_plan[key]["covered_tasks"] = updated_covered_tasks
                        coverage_plan[key]["benefit"] = sum(
                            self.priority_list[t] for t in updated_covered_tasks
                        )
            # 删除无效的覆盖方案
            for key in keys_to_remove:
                del coverage_plan[key]

        # 重新初始化解的属性
        solution._init_attrs()

        # # debug: 检查解是否可行
        self.evaluator.sol_feasible(solution)

        return solution


def base_alns_nsgaii_solver(
    population: "Population",
    instance: "InstanceClass",
    use_niching: bool = True,
    use_adaptive_selection: bool = True,
    plot_gif: bool = False,
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    solver = ALNSNSGAII(
        population,
        instance,
        use_niching=use_niching,
        use_adaptive_selection=use_adaptive_selection,
    )
    final_population = solver.solve()

    # debug: 生成目标函数进化GIF
    if plot_gif:
        solver.plot_generation_objectives_gif(
            save_path="alns_nsgaii_evolution.gif",
            fps=2,
            dpi=100,
            figsize=(10, 8),
            show_pareto_front=True,
        )
    return final_population


def alns_nsgaii_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    final_population = base_alns_nsgaii_solver(
        population,
        instance,
        use_niching=False,
        use_adaptive_selection=False,
    )
    return final_population


def alns_nsgaii_niching_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    """ALNS-NSGA-II求解器入口函数"""
    final_population = base_alns_nsgaii_solver(
        population,
        instance,
        use_niching=True,
        use_adaptive_selection=False,
    )
    return final_population


def alns_nsgaii_adaptive_select_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    """ALNS-NSGA-II求解器入口函数"""
    final_population = base_alns_nsgaii_solver(
        population,
        instance,
        use_niching=False,
        use_adaptive_selection=True,
    )
    return final_population


def alns_nsgaii_niching_adaptive_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    final_population = base_alns_nsgaii_solver(
        population,
        instance,
        use_niching=True,
        use_adaptive_selection=True,
    )
    return final_population
