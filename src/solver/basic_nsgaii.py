import os
import random
import time
from datetime import datetime
from typing import (
    List,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src.element import (
    InstanceClass,
    Solution,
)
from src.evaluator import Population
from src.solver.consr_heuristic import multi_randomized_greedy_heuristic


class BASICNSGAII:
    """basicNSGA-II 算法类"""

    def __init__(
        self,
        population: "Population",
        instance: "InstanceClass",
        use_adaptive_selection: bool = True,  # 新增参数：是否使用自适应选择
        use_niching: bool = True,  # 新增参数：是否使用小生境技术
    ) -> None:
        """初始化 NSGA-II 算法类

        Args:
            population (Population): 初始种群
            instance (InstanceClass): 问题实例
            use_adaptive_selection (bool): 是否使用自适应选择压力策略
            use_niching (bool): 是否使用小生境技术进行精英选择
        """

        self.population = population
        self.instance = instance

        # 提取id集合
        self.id_demand = population.evaluator.id_demand

        # 提取算法参数
        algorithm_config = instance.prob_config.algorithm_config
        self.pop_size = algorithm_config.pop_size
        self.max_iter = algorithm_config.max_iter
        self.crossover_prob = algorithm_config.crossover_prob
        self.mutation_prob = algorithm_config.mutation_prob

        # 【新增】自适应选择策略参数
        self.use_adaptive_selection = use_adaptive_selection
        self.selection_pressure = 0.3  # 初始选择压力（使用前30%的解）
        self.min_selection_pressure = 0.3  # 最小选择压力
        self.max_selection_pressure = 0.7  # 最大选择压力

        # 【新增】小生境技术参数
        self.use_niching = use_niching
        # 自适应小生境参数
        self.min_diversity_threshold = 0.02  # 最小多样性阈值(后期,允许更接近)
        self.max_diversity_threshold = 0.10  # 最大多样性阈值(早期,严格互斥)
        self.max_niche_capacity = 5  # 每个小生境最大容量(允许有限堆积)

        # 记录迭代过程
        self.iteration_data: List[dict] = []

        # 添加: 记录每代的种群目标函数值
        self.generation_objectives: List[np.ndarray] = []

    def solve(self) -> Population:
        """执行 NSGA-II 算法,返回最终种群"""

        # 开始计时
        start_time = time.perf_counter()
        # 初始化种群
        current_pop = self._init_population()

        # 初始化HV记录列表
        hv_history = []

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

            # 生成子代
            child_pop = self._generate_offspring(current_pop)

            # 评估子代
            child_pop.evaluate_all()

            # 精英选择(返回新种群) - 传递当前进度用于自适应小生境
            new_pop = self._elitist_selection(
                current_pop, child_pop, progress=iteration / self.max_iter
            )

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

    def _init_population(self) -> Population:
        """初始化种群"""

        init_pop = self.population.copy()

        init_pop = multi_randomized_greedy_heuristic(
            init_pop,
            self.instance,
        )

        return init_pop

    def _generate_offspring(
        self,
        parent_pop: Population,
    ) -> Population:
        """生成子代 - 支持自适应选择池策略"""

        # 【新增】如果使用自适应选择，构建精英选择池
        if self.use_adaptive_selection:
            elite_pool_indices = self._build_elite_pool(parent_pop)
        else:
            elite_pool_indices = None

        child_size = 0
        child_pop = Population(
            size=0,
            solver_name=parent_pop.solver_name,
            evaluator=parent_pop.evaluator,
        )

        while child_size < self.pop_size:
            # 选择父代
            if elite_pool_indices is not None:
                # 从精英池中选择
                parent1, idx1 = self._binary_tournament_selection_from_pool(
                    parent_pop, elite_pool_indices
                )
                parent2, idx2 = self._binary_tournament_selection_from_pool(
                    parent_pop, elite_pool_indices
                )
            else:
                # 从整个种群中选择（原始方法）
                parent1, idx1 = self._binary_tournament_selection(parent_pop)
                parent2, idx2 = self._binary_tournament_selection(parent_pop)

            # 交叉操作
            if random.random() < self.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
                # # debug: 检查解是否可行
                # self.population.evaluator.sol_feasible(child1)
                # self.population.evaluator.sol_feasible(child2)

            else:
                child1, child2 = parent1, parent2

            # 变异操作
            if random.random() < self.mutation_prob:
                child1 = self._mutation(child1)
            if random.random() < self.mutation_prob:
                child2 = self._mutation(child2)

            # 将子代添加到子代种群中
            child_pop.add_solution(child1)
            child_pop.add_solution(child2)

            # # debug: 检查解是否可行
            # self.population.evaluator.sol_feasible(child1)
            # self.population.evaluator.sol_feasible(child2)

            child_size += 2

        child_pop.size = child_size

        return child_pop

    def _build_elite_pool(self, parent_pop: Population) -> List[int]:
        """构建精英选择池 - 根据选择压力和非支配等级选择

        Args:
            parent_pop: 父代种群

        Returns:
            精英池中解的索引列表
        """

        # mark: 创新点——动态调整精英池大小

        # 确保种群已排序
        if not parent_pop.sorted:
            parent_pop.fast_non_dominated_sort_vectorized()

        # 根据选择压力确定目标池大小
        target_pool_size = int(self.pop_size * self.selection_pressure)
        target_pool_size = max(target_pool_size, 10)  # 至少10个解
        target_pool_size = min(target_pool_size, self.pop_size)  # 不超过种群大小

        elite_pool_indices = []

        # 逐层添加front，直到达到目标大小
        for front_idx_list in parent_pop.fronts_idx:
            elite_pool_indices.extend(front_idx_list)

            # 如果已经达到或超过目标大小，截断并返回
            if len(elite_pool_indices) >= target_pool_size:
                elite_pool_indices = elite_pool_indices[:target_pool_size]
                break

        # 如果所有front都加入后仍不足目标大小（极端情况），使用所有解
        if len(elite_pool_indices) < target_pool_size:
            elite_pool_indices = list(range(parent_pop.size))

        return elite_pool_indices

    def _binary_tournament_selection(
        self,
        current_pop: Population,
    ) -> Tuple[Solution, int]:
        """二进制锦标赛选择 - 原始版本（从整个种群中选择）

        从整个种群中随机选择两个个体,返回其中较优的一个
        比较标准:
        1. 首先比较非支配等级(rank),选择等级较低的
        2. 如果等级相同,则比较拥挤度距离(crowding_distance),选择距离较大的
        3. 避免选择退化解(0,0)

        Args:
            current_pop: 当前种群

        Returns:
            选中的解和其在种群中的索引
        """
        max_attempts = 10

        for attempt in range(max_attempts):
            # 随机选择两个不同的个体
            idx1, idx2 = random.sample(range(current_pop.size), 2)
            sol1 = current_pop.solutions[idx1]
            sol2 = current_pop.solutions[idx2]

            # 检查退化解
            is_sol1_degenerate = sol1.objectives[0] == 0 and sol1.objectives[1] == 0
            is_sol2_degenerate = sol2.objectives[0] == 0 and sol2.objectives[1] == 0

            # 如果有退化解，尝试重新选择
            if is_sol1_degenerate or is_sol2_degenerate:
                continue

            # 避免相同目标值的个体
            if (
                sol1.objectives[0] != sol2.objectives[0]
                or sol1.objectives[1] != sol2.objectives[1]
            ):
                break

        # 检查rank是否已计算
        if sol1.rank == -1 or sol2.rank == -1:
            raise ValueError("个体的非支配等级未计算，请先进行非支配排序。")

        # 如果一个是退化解，选择非退化解（双重保险）
        if is_sol1_degenerate and not is_sol2_degenerate:
            return sol2, idx2
        elif is_sol2_degenerate and not is_sol1_degenerate:
            return sol1, idx1

        # 比较非支配等级
        if sol1.rank < sol2.rank:
            return sol1, idx1
        elif sol2.rank < sol1.rank:
            return sol2, idx2
        else:
            # 非支配等级相同，比较拥挤度距离
            if sol1.crowding_distance > sol2.crowding_distance:
                return sol1, idx1
            else:
                return sol2, idx2

    def _binary_tournament_selection_from_pool(
        self,
        current_pop: Population,
        pool_indices: List[int],
    ) -> Tuple[Solution, int]:
        """从指定的选择池中进行二进制锦标赛选择 - 自适应版本

        Args:
            current_pop: 当前种群
            pool_indices: 可选择的解的索引列表

        Returns:
            选中的解和其在种群中的索引
        """
        max_attempts = 10

        for attempt in range(max_attempts):
            # 从池中随机选择两个索引
            idx1, idx2 = random.sample(pool_indices, 2)
            sol1 = current_pop.solutions[idx1]
            sol2 = current_pop.solutions[idx2]

            # 检查退化解
            is_sol1_degenerate = sol1.objectives[0] == 0 and sol1.objectives[1] == 0
            is_sol2_degenerate = sol2.objectives[0] == 0 and sol2.objectives[1] == 0

            if is_sol1_degenerate or is_sol2_degenerate:
                continue

            # 避免相同目标值
            if (
                sol1.objectives[0] != sol2.objectives[0]
                or sol1.objectives[1] != sol2.objectives[1]
            ):
                break

        # 检查rank是否已计算
        if sol1.rank == -1 or sol2.rank == -1:
            raise ValueError("个体的非支配等级未计算，请先进行非支配排序。")

        # 如果一个是退化解，选择非退化解（双重保险）
        if is_sol1_degenerate and not is_sol2_degenerate:
            return sol2, idx2
        elif is_sol2_degenerate and not is_sol1_degenerate:
            return sol1, idx1

        # 比较非支配等级
        if sol1.rank < sol2.rank:
            return sol1, idx1
        elif sol2.rank < sol1.rank:
            return sol2, idx2
        else:
            # 非支配等级相同，比较拥挤度距离
            if sol1.crowding_distance > sol2.crowding_distance:
                return sol1, idx1
            else:
                return sol2, idx2

    def _crossover(
        self,
        parent1: Solution,
        parent2: Solution,
    ) -> Tuple[Solution, Solution]:
        """交叉操作"""

        # 如果交叉，则需要执行交叉
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 交换路径片段（包括路径和覆盖情况）
        child1, child2 = self._crossover_visit_pice(child1, child2)

        return child1, child2

    def _crossover_visit_pice(
        self,
        parent1: Solution,
        parent2: Solution,
    ) -> Tuple[Solution, Solution]:
        """任务片段交叉操作（交换覆盖任务而非路径节点）

        交叉策略:
        1. 随机选择一个车辆
        2. 提取该车辆在两个父代中的覆盖任务(coverage信息)
        3. 随机选择部分任务进行交换
        4. 根据交换后的任务重新规划路径
        5. 检查路径可行性，只有可行时才返回子代
        """

        # 随机选择一个车辆
        vehicle_ids = list(self.instance.ground_veh_ids) + list(self.instance.drone_ids)
        selected_veh = random.choice(vehicle_ids)

        # 提取两个父代中该车辆的覆盖任务
        coverage1 = {k: v for k, v in parent1.coverage.items() if k[0] == selected_veh}
        coverage2 = {k: v for k, v in parent2.coverage.items() if k[0] == selected_veh}

        # 如果任一父代该车辆无任务，无法交叉
        if not coverage1 or not coverage2:
            return parent1, parent2

        # 获取任务列表（去重后的覆盖需求集合）
        tasks1 = self._extract_tasks_from_coverage(coverage1)
        tasks2 = self._extract_tasks_from_coverage(coverage2)

        # 任务数量不足无法交叉
        if len(tasks1) <= 1 or len(tasks2) <= 1:
            return parent1, parent2

        # 随机选择交叉区间（按任务索引）
        cut1_start = random.randint(0, len(tasks1) - 1)
        cut1_end = random.randint(cut1_start + 1, len(tasks1))
        cut2_start = random.randint(0, len(tasks2) - 1)
        cut2_end = random.randint(cut2_start + 1, len(tasks2))

        # 提取交换片段
        segment1 = tasks1[cut1_start:cut1_end]
        segment2 = tasks2[cut2_start:cut2_end]

        # 构造新的任务集合
        new_tasks1 = tasks1[:cut1_start] + segment2 + tasks1[cut1_end:]
        new_tasks2 = tasks2[:cut2_start] + segment1 + tasks2[cut2_end:]

        # 复制父代解
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 清除该车辆的原有路径和覆盖信息
        child1.routes[selected_veh] = [
            parent1.routes[selected_veh][0],
            parent1.routes[selected_veh][-1],
        ]
        child2.routes[selected_veh] = [
            parent2.routes[selected_veh][0],
            parent2.routes[selected_veh][-1],
        ]

        keys_to_remove_1 = [k for k in child1.coverage.keys() if k[0] == selected_veh]
        keys_to_remove_2 = [k for k in child2.coverage.keys() if k[0] == selected_veh]

        for key in keys_to_remove_1:
            del child1.coverage[key]
        for key in keys_to_remove_2:
            del child2.coverage[key]

        # 根据新任务集合重新构建路径和覆盖信息

        child1 = self._rebuild_route_from_tasks(
            child1, selected_veh, new_tasks1, parent1, parent2
        )
        child2 = self._rebuild_route_from_tasks(
            child2, selected_veh, new_tasks2, parent1, parent2
        )

        # 检查路径可行性
        evaluator = self.population.evaluator

        coverage1_new = {
            k: v for k, v in child1.coverage.items() if k[0] == selected_veh
        }
        coverage2_new = {
            k: v for k, v in child2.coverage.items() if k[0] == selected_veh
        }

        route_feasible1, reason1 = evaluator.is_route_feasible(
            selected_veh, child1.routes[selected_veh], coverage1_new
        )
        route_feasible2, reason2 = evaluator.is_route_feasible(
            selected_veh, child2.routes[selected_veh], coverage2_new
        )

        # 如果不可行，返回原始父代
        if not (route_feasible1 and route_feasible2):
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[warning] 交叉后路径不可行，放弃交叉操作 {route_feasible1, route_feasible2}"
            )
            raise ValueError(f"不可行原因{reason1}\n{reason2}")
            # return parent1, parent2

        # 调整覆盖信息，去除重复覆盖
        child1 = self._improve_coverage(child1, selected_veh)
        child2 = self._improve_coverage(child2, selected_veh)

        # debug: 检查是否出现新空解
        if self._is_empty_solution(child1) or self._is_empty_solution(child2):
            print(
                "-------------------"
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"{self._is_empty_solution(parent1), self._is_empty_solution(parent2)}"
                "交叉后，产生空子代子代解"
                f"{self._is_empty_solution(child1), self._is_empty_solution(child2)}"
            )
        # debug: 检查解是否可行
        # self.population.evaluator.sol_feasible(child1)
        # self.population.evaluator.sol_feasible(child2)

        # 重新初始化解属性
        child1._init_attrs()
        child2._init_attrs()

        return child1, child2

    def _extract_tasks_from_coverage(self, coverage: dict) -> List[frozenset]:
        """从coverage信息中提取任务列表

        每个任务是一个frozenset，包含该访问节点覆盖的所有需求节点

        Args:
            coverage: 覆盖信息字典 {(veh_id, visit_node): set(demand_nodes)}

        Returns:
            任务列表，每个任务是frozenset(demand_nodes)
        """
        tasks = []
        for (_, visit_node), demand_set in coverage.items():
            # 使用frozenset使任务可哈希，便于后续去重和比较
            task = frozenset(demand_set)
            tasks.append(task)

        return tasks

    def _rebuild_route_from_tasks(
        self,
        solution: Solution,
        veh_id: int,
        tasks: List[frozenset],
        parent1: Solution,
        parent2: Solution,
    ) -> Solution:
        """根据任务列表重建路径和覆盖信息

        策略:
        1. 对每个任务，优先从父代中查找能覆盖该任务的访问节点
        2. 如果父代中找不到，则寻找新的可行访问节点
        3. 按照最近邻策略排序访问节点构建路径

        Args:
            solution: 需要重建的解
            veh_id: 车辆ID
            tasks: 任务列表（需求节点集合的列表）
            parent1, parent2: 父代解，用于参考

        Returns:
            重建后的解
        """
        comm_coverage_matrix = self.instance.comm_coverage_matrix[veh_id]
        is_ground_veh = veh_id in self.instance.ground_veh_ids
        accessible_list = self.instance.accessible

        # 收集其他车辆已覆盖的需求
        covered_by_others = set()
        in_route_nodes = set()
        for (other_veh, node_id), covered_set in solution.coverage.items():
            in_route_nodes.add(node_id)
            if other_veh != veh_id:
                covered_by_others.update(covered_set)

        # 为每个任务找到访问节点
        visit_nodes = []
        new_coverage = {}
        del_task_idxs = []
        for idx, task in enumerate(tasks):
            # 过滤掉已被其他车辆覆盖的需求
            task_filtered = task - covered_by_others
            if not task_filtered:
                del_task_idxs.append(idx)
                continue  # 该任务已被其他车辆完全覆盖

            # 优先从父代中查找能覆盖该任务的访问节点
            visit_node = self._find_visit_node_from_parents(
                veh_id, task_filtered, parent1, parent2
            )

            # 如果父代中找不到，寻找新的可行访问节点
            if visit_node is None:
                visit_node = self._find_feasible_visit_node(
                    veh_id,
                    task_filtered,
                    is_ground_veh,
                    accessible_list,
                    comm_coverage_matrix,
                )

            if visit_node is not None:
                if visit_node in covered_by_others:
                    # 如果访问节点已被其他车辆覆盖，跳过该任务
                    del_task_idxs.append(idx)
                    continue

                if visit_node in in_route_nodes:
                    # 如果访问节点已在路径中，跳过该任务
                    del_task_idxs.append(idx)
                    continue

                visit_nodes.append(visit_node)
                # 修复覆盖情况
                if (visit_node in self.id_demand) and (visit_node not in task_filtered):
                    task_filtered = task_filtered.union({visit_node})
                # 更新coverage信息
                new_coverage[(veh_id, visit_node)] = set(
                    task_filtered
                )  # 将frozenset恢复为set
                covered_by_others.update(task_filtered)  # 更新已覆盖集合

        # 删除已经被覆盖的任务
        for index in sorted(del_task_idxs, reverse=True):
            del tasks[index]

        if (not visit_nodes) and tasks:
            raise ValueError("无法为任何任务找到可行的访问节点")

        # 去重访问节点
        visit_nodes = list(dict.fromkeys(visit_nodes))

        # 按照最近邻策略排序访问节点
        depot = solution.routes[veh_id][0]
        sorted_nodes = self._nearest_neighbor_sort(visit_nodes, depot)

        # 更新路径
        solution.routes[veh_id] = [depot] + sorted_nodes + [depot]

        # 更新覆盖信息
        solution.coverage.update(new_coverage)

        # # debug: 检查解的可行性
        # self.population.evaluator.sol_feasible(solution)

        return solution

    def _find_visit_node_from_parents(
        self,
        veh_id: int,
        task: frozenset,
        parent1: Solution,
        parent2: Solution,
    ) -> int | None:
        """从父代中查找能覆盖指定任务的访问节点

        Args:
            veh_id: 车辆ID
            task: 需要覆盖的需求节点集合
            parent1, parent2: 父代解

        Returns:
            访问节点ID，如果找不到返回None
        """
        # 检查parent1中该车辆的coverage
        for (v_id, visit_node), covered_set in parent1.coverage.items():
            if v_id == veh_id and task.issubset(covered_set):
                return visit_node

        # 检查parent2中该车辆的coverage
        for (v_id, visit_node), covered_set in parent2.coverage.items():
            if v_id == veh_id and task.issubset(covered_set):
                return visit_node

        return None

    def _find_feasible_visit_node(
        self,
        veh_id: int,
        task: frozenset,
        is_ground_veh: bool,
        accessible_list: list,
        comm_coverage_matrix: np.ndarray,
    ) -> int | None:
        """寻找能覆盖指定任务的可行访问节点

        Args:
            veh_id: 车辆ID
            task: 需要覆盖的需求节点集合
            is_ground_veh: 是否为地面车辆
            accessible_list: 可达性列表
            comm_coverage_matrix: 通信覆盖矩阵

        Returns:
            访问节点ID，如果找不到返回None
        """
        # 候选节点：排除base节点
        candidate_nodes = [
            n
            for n in range(self.instance.total_node_num)
            if n not in self.instance.base_ids
        ]

        # 地面车辆需要检查可达性
        if is_ground_veh:
            candidate_nodes = [n for n in candidate_nodes if accessible_list[n] == 1]

        # 找到能覆盖所有任务需求的节点
        for node in candidate_nodes:
            covered = set(np.where(comm_coverage_matrix[node])[0])
            if task.issubset(covered):
                return node

        return None

    def _nearest_neighbor_sort(self, nodes: List[int], start_node: int) -> List[int]:
        """使用最近邻策略排序节点

        Args:
            nodes: 待排序的节点列表
            start_node: 起始节点

        Returns:
            排序后的节点列表
        """
        if not nodes:
            return []

        distance_matrix = self.instance.distance_matrix
        sorted_nodes = []
        remaining = nodes.copy()
        current = start_node

        while remaining:
            # 找到距离当前节点最近的节点
            nearest = min(remaining, key=lambda n: distance_matrix[current][n])
            sorted_nodes.append(nearest)
            remaining.remove(nearest)
            current = nearest

        return sorted_nodes

    def _is_empty_solution(self, solution: Solution) -> bool:
        """检查解是否为空解（所有车辆均无路径）"""
        for veh_id in range(solution.num_vehicles):
            route = solution.routes[veh_id]
            if len(route) > 2:  # 起终点之外还有访问节点
                return False
        return True

    def _improve_coverage(
        self,
        solution: Solution,
        modified_veh: int,
    ) -> Solution:
        """basicNSGA-II 去除该操作"""

        return solution

    def _mutation(self, solution: Solution) -> Solution:
        """变异操作

        变异策略:
        1. 随机选择变异类型:
            - 单车辆变异(swap/inversion/insertion/delete)
            - 跨车辆变异(transfer/exchange)
        2. 对路径进行变异操作
        3. 同步更新覆盖信息
        4. 检查路径可行性,不可行则返回原解
        5. 调整覆盖信息以消除冲突
        """

        # 复制解
        mutated_solution = solution.copy()

        # 随机选择变异类型(添加跨车辆变异类型)
        mutate_types = [
            # 单车辆变异
            "swap",
            "inversion",
            "insertion",
            # # debug
            # # 跨车辆变异
            # "transfer",
            # "exchange",
            # # 任务级变异
            # "reselection",
        ]
        mutate_type = random.choice(mutate_types)

        # 根据变异类型选择不同的处理逻辑
        if mutate_type in ["swap", "inversion", "insertion", "delete"]:
            # 单车辆变异
            mutated_solution = self._single_vehicle_mutation(
                mutated_solution, mutate_type
            )
        elif mutate_type in ["transfer", "exchange"]:
            # 跨车辆变异
            mutated_solution = self._cross_vehicle_mutation(
                mutated_solution, mutate_type
            )
        elif mutate_type in ["reselection"]:
            # 任务级变异: 重新选择部分任务的覆盖节点
            mutated_solution = self._task_level_mutation(mutated_solution, mutate_type)

        mutated_solution._init_attrs()  # 重新初始化解属性

        # # debug: 检查解是否可行
        # self.population.evaluator.sol_feasible(mutated_solution)

        return mutated_solution

    def _single_vehicle_mutation(
        self, solution: Solution, mutate_type: str
    ) -> Solution:
        """单车辆变异操作(原有逻辑)"""

        # 随机选择一个车辆
        vehicle_ids = list(self.instance.ground_veh_ids) + list(self.instance.drone_ids)
        selected_veh = random.choice(vehicle_ids)

        # 提取路径(去除起终点)
        route = solution.routes[selected_veh][1:-1]

        # 路径长度不足无法变异
        if len(route) < 2:
            return solution  # 返回未变异的解

        # 保存起终点
        depot_start = solution.routes[selected_veh][0]
        depot_end = solution.routes[selected_veh][-1]

        # 随机选择变异类型
        mutate_types = [
            "swap",
            "inversion",
            "insertion",
            "delete",
        ]
        mutate_type = random.choice(mutate_types)

        # 执行变异操作
        if mutate_type == "swap":
            # 随机选择两个位置进行交换
            if len(route) >= 2:
                pos1, pos2 = random.sample(range(len(route)), 2)
                route[pos1], route[pos2] = route[pos2], route[pos1]

        elif mutate_type == "inversion":
            # 随机选择一个区间进行反转
            if len(route) >= 2:
                start, end = sorted(random.sample(range(len(route)), 2))
                if start < end:
                    route[start:end] = list(reversed(route[start:end]))

        elif mutate_type == "insertion":
            # 随机选择一个位置,移动到另一个位置
            if len(route) >= 2:
                from_pos = random.randint(0, len(route) - 1)
                to_pos = random.randint(0, len(route) - 1)
                if from_pos != to_pos:
                    node = route.pop(from_pos)
                    route.insert(to_pos, node)

        elif mutate_type == "delete":
            # 随机删除一个节点(如果路径长度允许)
            if len(route) > 1:
                pos = random.randint(0, len(route) - 1)
                deleted_node = route.pop(pos)
                # 删除该节点的覆盖信息
                if (selected_veh, deleted_node) in solution.coverage:
                    del solution.coverage[(selected_veh, deleted_node)]

        # 重新构建完整路径
        new_route = [depot_start] + route + [depot_end]
        solution.routes[selected_veh] = new_route

        # 检查可行性和调整覆盖信息
        evaluator = self.population.evaluator
        coverage = {k: v for k, v in solution.coverage.items() if k[0] == selected_veh}
        route_feasible, _ = evaluator.is_route_feasible(
            selected_veh, new_route, coverage
        )

        if not route_feasible:
            # debug: 打印不可行信息
            # print(
            #     f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            #     f" [warning] 单车辆变异后路径不可行，放弃变异操作 {mutate_type}"
            # )
            return solution  # 返回原解

        # 调整覆盖信息
        solution = self._improve_coverage(solution, selected_veh)

        return solution

    def _cross_vehicle_mutation(self, solution: Solution, mutate_type: str) -> Solution:
        """跨车辆变异操作

        变异类型:
        - transfer: 将一个节点从一个车辆转移到另一个车辆
        - exchange: 交换两个车辆之间的节点

        注意: 必须检查地面车辆的可达性约束
        """

        vehicle_ids = list(self.instance.ground_veh_ids) + list(self.instance.drone_ids)

        # 至少需要两个车辆
        if len(vehicle_ids) < 2:
            return solution

        # 随机选择两个不同的车辆
        veh1, veh2 = random.sample(vehicle_ids, 2)

        # 判断车辆类型
        is_veh1_ground = veh1 in self.instance.ground_veh_ids
        is_veh2_ground = veh2 in self.instance.ground_veh_ids

        # 提取路径(去除起终点) - 使用copy避免修改原解
        route1 = solution.routes[veh1][1:-1].copy()
        route2 = solution.routes[veh2][1:-1].copy()

        # 保存起终点
        depot_start1, depot_end1 = (
            solution.routes[veh1][0],
            solution.routes[veh1][-1],
        )
        depot_start2, depot_end2 = (
            solution.routes[veh2][0],
            solution.routes[veh2][-1],
        )

        # 保存原始路径和覆盖信息，用于变异失败时恢复
        original_route1 = solution.routes[veh1].copy()
        original_route2 = solution.routes[veh2].copy()
        original_coverage = solution.coverage.copy()

        if mutate_type == "transfer":
            # 节点转移: 从veh1转移一个节点到veh2
            if len(route1) < 1:
                return solution

            # 随机选择veh1的一个节点
            transfer_idx = random.randint(0, len(route1) - 1)
            transfer_node = route1[transfer_idx]

            # **关键检查**: 如果veh2是地面车辆,检查节点可达性
            if is_veh2_ground and self.instance.accessible[transfer_node] == 0:
                # 节点不可达,放弃变异
                return solution

            # 执行转移
            route1.pop(transfer_idx)

            # 随机插入到veh2的路径中
            insert_pos = random.randint(0, len(route2))
            route2.insert(insert_pos, transfer_node)

            # 更新覆盖信息: 删除veh1的,添加到veh2
            if (veh1, transfer_node) in solution.coverage:
                coverage_set = solution.coverage[(veh1, transfer_node)]
                del solution.coverage[(veh1, transfer_node)]
                solution.coverage[(veh2, transfer_node)] = coverage_set

        elif mutate_type == "exchange":
            # 节点交换: 交换两个车辆各一个节点
            if len(route1) < 1 or len(route2) < 1:
                return solution

            # 随机选择各自的一个节点
            idx1 = random.randint(0, len(route1) - 1)
            idx2 = random.randint(0, len(route2) - 1)

            node1 = route1[idx1]
            node2 = route2[idx2]

            # **关键检查**: 验证交换后的可达性
            # 如果veh1是地面车辆,检查node2是否可达
            if is_veh1_ground and self.instance.accessible[node2] == 0:
                return solution

            # 如果veh2是地面车辆,检查node1是否可达
            if is_veh2_ground and self.instance.accessible[node1] == 0:
                return solution

            # 交换节点
            route1[idx1] = node2
            route2[idx2] = node1

            # 交换覆盖信息
            coverage1 = solution.coverage.get((veh1, node1))
            coverage2 = solution.coverage.get((veh2, node2))

            if coverage1:
                del solution.coverage[(veh1, node1)]
                solution.coverage[(veh2, node1)] = coverage1

            if coverage2:
                del solution.coverage[(veh2, node2)]
                solution.coverage[(veh1, node2)] = coverage2

        # 重新构建完整路径
        solution.routes[veh1] = [depot_start1] + route1 + [depot_end1]
        solution.routes[veh2] = [depot_start2] + route2 + [depot_end2]

        # 检查两条路径的可行性
        evaluator = self.population.evaluator

        route_coverage1 = {k: v for k, v in solution.coverage.items() if k[0] == veh1}
        route_coverage2 = {k: v for k, v in solution.coverage.items() if k[0] == veh2}

        route_feasible1, _ = evaluator.is_route_feasible(
            veh1, solution.routes[veh1], route_coverage1
        )
        route_feasible2, _ = evaluator.is_route_feasible(
            veh2, solution.routes[veh2], route_coverage2
        )

        # 如果任一路径不可行,恢复原始状态并返回
        if not (route_feasible1 and route_feasible2):
            # debug: 打印不可行信息
            # print(
            #     f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            #     f" [warning] 跨车辆变异后路径不可行，放弃变异操作 {mutate_type}, "
            #     f"feasible: ({route_feasible1}, {route_feasible2})"
            # )
            # 恢复原始状态
            solution.routes[veh1] = original_route1
            solution.routes[veh2] = original_route2
            solution.coverage = original_coverage
            return solution

        # 调整两个车辆的覆盖信息
        solution = self._improve_coverage(solution, veh1)
        solution = self._improve_coverage(solution, veh2)

        return solution

    def _task_level_mutation(self, solution: Solution, mutate_type: str) -> Solution:
        """任务级变异: 重新选择部分任务的覆盖节点

        包含两种策略:
        1. 任务形式替换: 分析当前任务是否有更好的执行方法
        2. 任务对象替换: 如果任务附近有可执行的、距离更近的、优先级相同或更高的访问方案,则替换
        """

        # 复制解
        mutated_solution = solution.copy()

        # 随机选择一个有路径的车辆
        vehicle_ids = list(self.instance.ground_veh_ids) + list(self.instance.drone_ids)
        candidates = [v for v in vehicle_ids if len(mutated_solution.routes[v]) > 2]

        if not candidates:
            return solution

        selected_veh = random.choice(candidates)
        route = mutated_solution.routes[selected_veh]

        # 判断车辆类型
        is_ground_veh = selected_veh in self.instance.ground_veh_ids

        # 随机选择路径上的一个访问节点(排除起终点)
        if len(route) <= 2:
            return solution

        visit_idx = random.randint(1, len(route) - 2)
        visit_node = route[visit_idx]

        # 获取前后访问点
        prev_node = route[visit_idx - 1]
        next_node = route[visit_idx + 1]

        # 获取当前节点覆盖的需求集合
        current_coverage = mutated_solution.coverage.get(
            (selected_veh, visit_node), set()
        )

        if not current_coverage:
            return solution

        # 策略1: 任务形式替换 - 寻找能覆盖相同需求集合的更优节点
        better_node = self._find_better_coverage_node_vectorized(
            selected_veh,
            visit_node,
            current_coverage,
            prev_node,
            next_node,
            is_ground_veh,
        )

        # 策略2: 任务对象替换 - 寻找附近优先级相同或更高的替代任务
        if better_node is None:
            better_node = self._find_alternative_task_node_vectorized(
                selected_veh,
                visit_node,
                current_coverage,
                prev_node,
                next_node,
                is_ground_veh,
                mutated_solution,
            )

        # 检查可行性
        evaluator = self.population.evaluator

        # # debug: 检查解是否可行
        # self.population.evaluator.sol_feasible(mutated_solution)

        # 提取当前在route上的所有node
        for veh_id, node_id in mutated_solution.coverage.keys():
            if veh_id == selected_veh:
                continue
            if node_id == better_node:
                # 替换节点已被其他车辆覆盖，放弃变异
                return solution

        # 如果找到更优节点,执行替换
        if better_node is not None and better_node != visit_node:
            if better_node in mutated_solution.routes[selected_veh]:
                # 更好的节点本身就在路径上
                mutated_solution.routes[selected_veh].pop(visit_idx)
                if (selected_veh, visit_node) in mutated_solution.coverage:
                    # 删除原节点的覆盖信息
                    del mutated_solution.coverage[(selected_veh, visit_node)]

            else:
                # 保存原始状态用于回滚
                original_route = mutated_solution.routes[selected_veh].copy()
                original_coverage = mutated_solution.coverage.copy()

                # 更新路径
                mutated_solution.routes[selected_veh][visit_idx] = better_node

                # 更新覆盖信息
                if (selected_veh, visit_node) in mutated_solution.coverage:
                    old_coverage = mutated_solution.coverage[(selected_veh, visit_node)]
                    del mutated_solution.coverage[(selected_veh, visit_node)]

                    # 如果当前需要访问的点已经被覆盖，则需要从对应的覆盖信息中删除
                    included_tasks = mutated_solution.get_included_nodes()
                    if better_node in included_tasks:
                        for key, covered_set in mutated_solution.coverage.items():
                            if better_node in covered_set:
                                covered_set.remove(better_node)

                    if better_node in evaluator.id_demand:
                        better_node_coverage = old_coverage | {better_node}
                        mutated_solution.coverage[(selected_veh, better_node)] = (
                            better_node_coverage
                        )
                    elif better_node in evaluator.id_steiner:
                        mutated_solution.coverage[(selected_veh, better_node)] = (
                            old_coverage
                        )
                    else:
                        raise ValueError(
                            f"更优节点{better_node}既不是需求节点也不是Steiner节点"
                        )

            route_coverage = {
                k: v
                for k, v in mutated_solution.coverage.items()
                if k[0] == selected_veh
            }
            route_feasible, _ = evaluator.is_route_feasible(
                selected_veh, mutated_solution.routes[selected_veh], route_coverage
            )

            if not route_feasible:
                # 回滚
                mutated_solution.routes[selected_veh] = original_route
                mutated_solution.coverage = original_coverage
                return solution

            # # debug: 检查解是否可行
            # self.population.evaluator.sol_feasible(mutated_solution)

            # 调整覆盖信息
            mutated_solution = self._improve_coverage(mutated_solution, selected_veh)

        return mutated_solution

    def _find_better_coverage_node_vectorized(
        self,
        veh_id: int,
        current_node: int,
        target_coverage: set,
        prev_node: int,
        next_node: int,
        is_ground_veh: bool,
    ) -> int | None:
        """策略1: 向量化版本 - 寻找能覆盖相同需求的更优节点"""

        comm_coverage = self.instance.comm_coverage_matrix[veh_id]
        distance_matrix = self.instance.distance_matrix

        # 计算当前节点的总距离
        current_dist = (
            distance_matrix[prev_node][current_node]
            + distance_matrix[current_node][next_node]
        )

        # 构建候选节点mask
        total_node_num = self.instance.total_node_num
        candidate_mask = np.ones(total_node_num, dtype=bool)

        # 排除base和当前节点
        candidate_mask[list(self.instance.base_ids)] = False
        candidate_mask[current_node] = False

        # 地面车辆检查可达性
        if is_ground_veh:
            accessible_array = np.array(self.instance.accessible)
            candidate_mask &= accessible_array == 1

        # 获取候选节点索引
        candidate_nodes = np.where(candidate_mask)[0]

        if len(candidate_nodes) == 0:
            return None

        # 向量化检查覆盖能力
        # comm_coverage shape: (total_node_num, total_node_num) -> (n_candidates, total_node_num)
        candidate_coverage = comm_coverage[candidate_nodes]

        # 筛选需求节点
        demand_mask = np.zeros(total_node_num, dtype=bool)
        demand_mask[list(self.instance.demand_ids)] = True

        # 对每个候选节点,检查是否覆盖所有目标需求
        target_coverage_array = np.array(list(target_coverage))

        # 检查每个候选节点是否覆盖所有目标需求
        # candidate_coverage[:, target_coverage_array] shape: (n_candidates, n_targets)
        coverage_check = np.all(candidate_coverage[:, target_coverage_array], axis=1)

        # 筛选满足覆盖条件的候选节点
        valid_candidates = candidate_nodes[coverage_check]

        if len(valid_candidates) == 0:
            return None

        # 向量化计算距离
        # distance_matrix[prev_node, valid_candidates] + distance_matrix[valid_candidates, next_node]
        candidate_dists = (
            distance_matrix[prev_node, valid_candidates]
            + distance_matrix[valid_candidates, next_node]
        )

        # 找到距离最小且小于当前距离的节点
        best_idx = np.argmin(candidate_dists)
        best_dist = candidate_dists[best_idx]

        if best_dist < current_dist:
            return int(valid_candidates[best_idx])

        return None

    def _find_alternative_task_node_vectorized(
        self,
        veh_id: int,
        current_node: int,
        current_coverage: set,
        prev_node: int,
        next_node: int,
        is_ground_veh: bool,
        solution: Solution,
        search_radius_ratio: float = 1.5,
    ) -> int | None:
        """策略2: 向量化版本 - 寻找附近优先级相同或更高的替代任务"""

        comm_coverage = self.instance.comm_coverage_matrix[veh_id]
        distance_matrix = self.instance.distance_matrix

        # 计算当前节点的总距离和搜索半径
        current_dist = (
            distance_matrix[prev_node][current_node]
            + distance_matrix[current_node][next_node]
        )
        search_radius = current_dist * search_radius_ratio

        # 收集其他车辆已覆盖的需求 - 向量化
        covered_by_others = set()
        for (other_veh, node), covered_set in solution.coverage.items():
            if other_veh != veh_id:
                covered_by_others.update(covered_set)

        # 构建候选节点mask
        total_node_num = self.instance.total_node_num
        candidate_mask = np.ones(total_node_num, dtype=bool)

        # 排除base和当前节点
        candidate_mask[list(self.instance.base_ids)] = False
        candidate_mask[current_node] = False

        # 地面车辆检查可达性
        if is_ground_veh:
            accessible_array = np.array(self.instance.accessible)
            candidate_mask &= accessible_array == 1

        # 获取候选节点
        candidate_nodes = np.where(candidate_mask)[0]

        if len(candidate_nodes) == 0:
            return None

        # 向量化计算距离约束
        candidate_dists = (
            distance_matrix[prev_node, candidate_nodes]
            + distance_matrix[candidate_nodes, next_node]
        )

        # 筛选在搜索半径内的节点
        within_radius = candidate_dists <= search_radius
        candidate_nodes = candidate_nodes[within_radius]
        candidate_dists = candidate_dists[within_radius]

        if len(candidate_nodes) == 0:
            return None

        # 向量化计算覆盖信息
        candidate_coverage_matrix = comm_coverage[candidate_nodes]

        # 需求节点mask
        demand_mask = np.zeros(total_node_num, dtype=bool)
        demand_mask[list(self.instance.demand_ids)] = True

        # 对每个候选节点,计算其覆盖的需求数量(排除已被其他车辆覆盖的)
        candidate_priorities = np.zeros(len(candidate_nodes), dtype=int)

        for i, node in enumerate(candidate_nodes):
            # 获取该节点能覆盖的需求
            covered_demands = set(
                np.where(candidate_coverage_matrix[i] & demand_mask)[0]
            )
            # 排除已被其他车辆覆盖的
            covered_demands -= covered_by_others
            candidate_priorities[i] = len(covered_demands)

        # 筛选优先级 >= 当前优先级的节点
        current_priority = len(current_coverage)
        valid_priority_mask = candidate_priorities >= current_priority

        if not np.any(valid_priority_mask):
            return None

        # 筛选有效节点
        valid_nodes = candidate_nodes[valid_priority_mask]
        valid_dists = candidate_dists[valid_priority_mask]
        valid_priorities = candidate_priorities[valid_priority_mask]

        # 向量化计算综合得分
        distance_improvement = (current_dist - valid_dists) / (current_dist + 1e-6)
        scores = 0.6 * valid_priorities + 0.4 * distance_improvement * 10

        # 选择得分最高的节点
        best_idx = np.argmax(scores)

        if scores[best_idx] > 0:
            return int(valid_nodes[best_idx])

        return None

    def _elitist_selection(
        self, current_pop: Population, child_pop: Population, progress: float = 0.0
    ) -> Population:
        """精英选择操作 - 支持选择是否使用小生境技术

        Args:
            current_pop: 当前种群
            child_pop: 子代种群
            progress: 当前进化进度(0-1),用于自适应小生境

        Returns:
            选择后的新种群
        """

        # 合并父代和子代种群
        combined_pop = current_pop.merge(child_pop)

        # 进行非支配排序
        combined_pop.fast_non_dominated_sort_vectorized()

        if self.use_niching:
            # 【新增】计算自适应多样性阈值：早期严格(大阈值)，后期宽松(小阈值)
            # 使用非线性衰减：前期快速下降，后期缓慢下降
            adaptive_threshold = self.max_diversity_threshold - (
                self.max_diversity_threshold - self.min_diversity_threshold
            ) * (
                progress**0.5
            )  # 平方根衰减

            # 使用小生境选择 - 移除重复解和相似解
            selected_solutions = self._niching_selection(
                combined_pop,
                target_size=self.pop_size,
                diversity_threshold=adaptive_threshold,
                niche_capacity=self.max_niche_capacity,
            )

            # 创建新种群
            new_pop = Population(
                size=len(selected_solutions),
                solver_name=current_pop.solver_name,
                evaluator=current_pop.evaluator,
            )
            new_pop.solutions = selected_solutions
        else:
            # 使用原始的select_best方法（基于rank和拥挤度）
            selected_solutions = combined_pop.select_best(self.pop_size)

            # 创建新种群
            new_pop = Population(
                size=len(selected_solutions),
                solver_name=current_pop.solver_name,
                evaluator=current_pop.evaluator,
            )
            new_pop.solutions = selected_solutions

        # 清理临时的合并种群
        del combined_pop

        return new_pop

    def _niching_selection(
        self,
        population: Population,
        target_size: int,
        diversity_threshold: float = 0.05,
        niche_capacity: int = 3,
    ) -> List[Solution]:
        """小生境选择 - 保持种群多样性(支持自适应)"""

        selected_solutions: List[Solution] = []
        degenerate_count = 0
        duplicate_count = 0

        # 按非支配等级分组
        rank_groups: dict[int, List[Solution]] = {}
        for sol in population.solutions:
            if sol.rank not in rank_groups:
                rank_groups[sol.rank] = []
            rank_groups[sol.rank].append(sol)

        # 按等级从低到高处理
        for rank in sorted(rank_groups.keys()):
            if len(selected_solutions) >= target_size:
                break

            rank_solutions = rank_groups[rank]

            # 【新增】为当前rank的解重新计算拥挤度距离
            self._update_crowding_distance(rank_solutions)

            # **步骤1: 过滤退化解 - 保留至少一个退化解**
            non_degenerate = []
            degenerate_solutions = []

            for sol in rank_solutions:
                if sol.objectives[0] == 0 and sol.objectives[1] == 0:
                    degenerate_count += 1
                    degenerate_solutions.append(sol)
                else:
                    non_degenerate.append(sol)

            # 如果存在退化解,保留一个
            if degenerate_solutions:
                best_degenerate = degenerate_solutions[0]
                non_degenerate.append(best_degenerate)

            if not non_degenerate:
                continue

            # **步骤2: 移除重复和相似解(支持小生境容量限制)**
            diverse_solutions = self._remove_similar_solutions_with_capacity(
                non_degenerate, diversity_threshold, niche_capacity
            )

            duplicate_count += len(non_degenerate) - len(diverse_solutions)

            # **步骤3: 根据剩余容量选择解**
            remaining_slots = target_size - len(selected_solutions)

            if len(diverse_solutions) <= remaining_slots:
                selected_solutions.extend(diverse_solutions)
            else:
                # 【修改】需要进一步筛选 - 按拥挤度距离排序(已更新)
                diverse_solutions.sort(key=lambda x: x.crowding_distance, reverse=True)
                selected_solutions.extend(diverse_solutions[:remaining_slots])

        # debug：打印niching统计信息(包含自适应阈值)
        # if degenerate_count > 0 or duplicate_count > 0:
        #     print(
        #         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        #         f"[Niching] 阈值={diversity_threshold:.4f}, 容量={niche_capacity}, "
        #         f"过滤 {degenerate_count} 个退化解, "
        #         f"{duplicate_count} 个重复/相似解, "
        #         f"最终选择 {len(selected_solutions)} 个解"
        #     )

        return selected_solutions

    def _remove_similar_solutions(
        self,
        solutions: List[Solution],
        threshold: float,
    ) -> List[Solution]:
        """移除相似解 - 基于目标空间欧氏距离(原始版本,保留用于兼容)

        Args:
            solutions: 待处理的解列表
            threshold: 距离阈值(小于此值认为相似)

        Returns:
            去重后的解列表
        """

        if len(solutions) <= 1:
            return solutions

        # 提取目标函数值
        objectives = np.array([sol.objectives for sol in solutions])

        # 归一化目标函数值(避免量纲影响)
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min

        # 防止除零
        obj_range[obj_range < 1e-10] = 1.0

        normalized_obj = (objectives - obj_min) / obj_range

        # 使用贪心策略选择多样化的解
        selected_indices = [0]  # 首先选择第一个解
        selected_objectives = [normalized_obj[0]]

        for i in range(1, len(solutions)):
            current_obj = normalized_obj[i]

            # 计算与已选解的最小距离
            min_distance = min(
                np.linalg.norm(current_obj - selected_obj)
                for selected_obj in selected_objectives
            )

            # 如果距离足够大，则添加
            if min_distance > threshold:
                selected_indices.append(i)
                selected_objectives.append(current_obj)

        return [solutions[i] for i in selected_indices]

    def _remove_similar_solutions_with_capacity(
        self,
        solutions: List[Solution],
        threshold: float,
        niche_capacity: int = 3,
    ) -> List[Solution]:
        """移除相似解 - 支持小生境容量限制(自适应版本)

        策略:
        1. 基于目标空间欧氏距离判断相似性
        2. 每个小生境允许保留最多niche_capacity个个体(允许有限堆积)
        3. 同一小生境内,按拥挤度距离排序保留最优的个体

        Args:
            solutions: 待处理的解列表
            threshold: 距离阈值(小于此值认为同一小生境)
            niche_capacity: 每个小生境的最大容量

        Returns:
            去重后的解列表
        """

        if len(solutions) <= 1:
            return solutions

        # 提取目标函数值
        objectives = np.array([sol.objectives for sol in solutions])

        # 归一化目标函数值(避免量纲影响)
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min

        # 防止除零
        obj_range[obj_range < 1e-10] = 1.0

        normalized_obj = (objectives - obj_min) / obj_range

        # 使用字典存储小生境: {niche_id: [解的索引列表]}
        niches: dict[int, List[int]] = {}
        niche_centers: List[np.ndarray] = []  # 小生境中心点

        # 为每个解分配小生境
        for i in range(len(solutions)):
            current_obj = normalized_obj[i]

            # 查找最近的小生境中心
            min_distance = float("inf")
            closest_niche_id = -1

            for niche_id, center in enumerate(niche_centers):
                distance = float(np.linalg.norm(current_obj - center))
                if distance < min_distance:
                    min_distance = distance
                    closest_niche_id = niche_id

            # 判断是否归属于现有小生境
            if min_distance <= threshold and closest_niche_id != -1:
                # 归属于现有小生境
                niches[closest_niche_id].append(i)
            else:
                # 创建新小生境
                new_niche_id = len(niche_centers)
                niches[new_niche_id] = [i]
                niche_centers.append(current_obj.copy())

        # 从每个小生境中选择最多niche_capacity个解
        selected_indices = []

        for niche_id, indices in niches.items():
            if len(indices) <= niche_capacity:
                # 小生境内个体数不超过容量,全部保留
                selected_indices.extend(indices)
            else:
                # 【修改】超过容量,先更新拥挤度再排序选择
                niche_solutions = [solutions[idx] for idx in indices]
                self._update_crowding_distance(niche_solutions)

                # 按拥挤度距离排序
                sorted_pairs = sorted(
                    zip(indices, niche_solutions),
                    key=lambda x: x[1].crowding_distance,
                    reverse=True,
                )

                selected_in_niche = [idx for idx, _ in sorted_pairs[:niche_capacity]]
                selected_indices.extend(selected_in_niche)

        return [solutions[i] for i in selected_indices]

    # 【新增】独立的拥挤度计算方法
    def _calculate_crowding_distance(self, solutions: List[Solution]) -> List[float]:
        """计算解集的拥挤度距离

        Args:
            solutions: 解的列表

        Returns:
            每个解的拥挤度距离列表
        """
        n = len(solutions)
        if n <= 2:
            return [float("inf")] * n

        # 提取目标函数值
        objectives = np.array([sol.objectives for sol in solutions])
        n_obj = objectives.shape[1]

        # 初始化拥挤度距离
        crowding_distances = np.zeros(n)

        # 对每个目标函数计算拥挤度
        for m in range(n_obj):
            # 按第m个目标排序
            sorted_indices = np.argsort(objectives[:, m])

            # 边界解的拥挤度设为无穷大
            crowding_distances[sorted_indices[0]] = float("inf")
            crowding_distances[sorted_indices[-1]] = float("inf")

            # 计算目标函数的范围
            obj_range = (
                objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            )

            # 避免除零
            if obj_range < 1e-10:
                continue

            # 计算中间解的拥挤度
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                crowding_distances[idx] += (
                    objectives[sorted_indices[i + 1], m]
                    - objectives[sorted_indices[i - 1], m]
                ) / obj_range

        return crowding_distances.tolist()

    def _update_crowding_distance(self, solutions: List[Solution]) -> None:
        """更新解集的拥挤度距离(直接修改Solution对象)

        Args:
            solutions: 解的列表
        """
        distances = self._calculate_crowding_distance(solutions)
        for sol, dist in zip(solutions, distances):
            sol.crowding_distance = dist

    def plot_generation_objectives_gif(
        self,
        save_path: str | None = None,
        fps: int = 2,
        dpi: int = 100,
        figsize: tuple = (10, 8),
        show_pareto_front: bool = True,
    ) -> None:
        """生成每代目标函数值的动画GIF

        Args:
            save_path: GIF保存路径,默认为工作目录下的'nsga2_evolution.gif'
            fps: 帧率,默认2帧/秒
            dpi: 图像分辨率
            figsize: 图像大小
            show_pareto_front: 是否高亮显示Pareto前沿
        """

        if not self.generation_objectives:
            print("没有记录的代际数据,请先运行solve()方法")
            return

        if save_path is None:
            save_path = "nsga2_evolution.gif"

        # 创建保存目录
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )

        # 计算所有代的目标函数范围(用于统一坐标轴)
        all_objectives = np.vstack(self.generation_objectives)
        obj1_min, obj1_max = all_objectives[:, 0].min(), all_objectives[:, 0].max()
        obj2_min, obj2_max = all_objectives[:, 1].min(), all_objectives[:, 1].max()

        # 添加边距
        obj1_margin = (obj1_max - obj1_min) * 0.1
        obj2_margin = (obj2_max - obj2_min) * 0.1

        fig, ax = plt.subplots(figsize=figsize)

        def update(frame):
            # 使用nonlocal声明外部变量
            nonlocal obj1_min, obj1_max, obj2_min, obj2_max, obj1_margin, obj2_margin

            ax.clear()

            # 当前代的目标函数
            objectives = self.generation_objectives[frame]

            # 用于收集返回的Artist对象
            artists = []

            # 1. 绘制上一代的点(如果存在)
            if frame > 0:
                prev_objectives = self.generation_objectives[frame - 1]

                # 统计上一代的重复点
                prev_unique_coords, prev_counts = self._count_duplicate_points(
                    prev_objectives
                )

                # 绘制上一代的点(灰色)
                scatter_prev = ax.scatter(
                    prev_unique_coords[:, 0],
                    prev_unique_coords[:, 1],
                    c="lightgray",
                    alpha=0.3,
                    s=50,
                    label="Previous Generation",
                    zorder=1,
                )
                artists.append(scatter_prev)

                # 标注上一代重复点数量
                for coord, count in zip(prev_unique_coords, prev_counts):
                    if count > 1:
                        ax.annotate(
                            f"{count}",
                            xy=(coord[0], coord[1]),
                            xytext=(8, -8),  # 右下位置
                            textcoords="offset points",
                            fontsize=8,
                            color="gray",
                            alpha=0.6,
                            zorder=2,
                        )

            # 2. 统计当前代的重复点
            unique_coords, counts = self._count_duplicate_points(objectives)

            # 3. 绘制当前代的点
            if show_pareto_front and hasattr(self, "generation_ranks"):
                # 获取当前代的rank信息
                ranks = self.generation_ranks[frame]  # type: ignore

                # 为每个唯一坐标确定是否属于Pareto前沿
                # 通过检查该坐标对应的任意一个解的rank
                pareto_flags_list = []
                for coord in unique_coords:
                    # 找到该坐标对应的第一个解的索引
                    mask = np.all(objectives == coord, axis=1)
                    idx = np.where(mask)[0][0]
                    pareto_flags_list.append(ranks[idx] == 0)

                pareto_flags = np.array(pareto_flags_list)

                # 绘制非Pareto解
                non_pareto_coords = unique_coords[~pareto_flags]
                non_pareto_counts = counts[~pareto_flags]

                if len(non_pareto_coords) > 0:
                    scatter1 = ax.scatter(
                        non_pareto_coords[:, 0],
                        non_pareto_coords[:, 1],
                        c="lightblue",
                        alpha=0.6,
                        s=50,
                        label="Non-Pareto Solutions",
                        zorder=3,
                    )
                    artists.append(scatter1)

                    # 标注非Pareto重复点数量
                    for coord, count in zip(non_pareto_coords, non_pareto_counts):
                        if count > 1:
                            ax.annotate(
                                f"{count}",
                                xy=(coord[0], coord[1]),
                                xytext=(8, 8),  # 右上位置,与上一代的右下位置区分
                                textcoords="offset points",
                                fontsize=9,
                                color="blue",
                                fontweight="bold",
                                zorder=4,
                            )

                # 绘制Pareto前沿
                pareto_coords = unique_coords[pareto_flags]
                pareto_counts = counts[pareto_flags]

                if len(pareto_coords) > 0:
                    scatter2 = ax.scatter(
                        pareto_coords[:, 0],
                        pareto_coords[:, 1],
                        c="red",
                        alpha=0.8,
                        s=80,
                        marker="*",
                        label="Pareto Front",
                        zorder=5,
                    )
                    artists.append(scatter2)

                    # 标注Pareto重复点数量
                    for coord, count in zip(pareto_coords, pareto_counts):
                        if count > 1:
                            ax.annotate(
                                f"{count}",
                                xy=(coord[0], coord[1]),
                                xytext=(8, 8),  # 右上位置,与上一代的右下位置区分
                                textcoords="offset points",
                                fontsize=10,
                                color="red",
                                fontweight="bold",
                                zorder=6,
                            )
            else:
                # 只绘制所有解(蓝色)
                scatter = ax.scatter(
                    unique_coords[:, 0],
                    unique_coords[:, 1],
                    c="blue",
                    alpha=0.6,
                    s=50,
                    label="Current Generation",
                    zorder=3,
                )
                artists.append(scatter)

                # 标注重复点数量
                for coord, count in zip(unique_coords, counts):
                    if count > 1:
                        ax.annotate(
                            f"{count}",
                            xy=(coord[0], coord[1]),
                            xytext=(8, 8),  # 右上位置,与上一代的右下位置区分
                            textcoords="offset points",
                            fontsize=9,
                            color="blue",
                            fontweight="bold",
                            zorder=4,
                        )

            # 设置坐标轴范围
            ax.set_xlim(obj1_min - obj1_margin, obj1_max + obj1_margin)
            ax.set_ylim(obj2_min - obj2_margin, obj2_max + obj2_margin)

            # 设置标签和标题
            ax.set_xlabel("Objective 1 (Total Cost)", fontsize=12)
            ax.set_ylabel("Objective 2 (Coverage)", fontsize=12)
            ax.set_title(
                f"Generation {frame}/{len(self.generation_objectives)-1}", fontsize=14
            )
            ax.grid(True, alpha=0.3)

            # 显示图例
            ax.legend(loc="best", fontsize=10)

            # 返回Artist对象列表
            return artists

        # 创建动画
        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.generation_objectives),
            interval=1000 // fps,
            repeat=True,
        )

        # 保存为GIF
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        plt.close()

        print(f"GIF已保存至: {save_path}")

    def _count_duplicate_points(self, objectives: np.ndarray) -> tuple:
        """统计重复的目标函数值点

        Args:
            objectives: 目标函数值数组,形状为(n_solutions, n_objectives)

        Returns:
            unique_coords: 唯一坐标数组
            counts: 每个唯一坐标的重复次数
        """
        # 使用字典统计每个坐标的出现次数
        coord_dict: dict[tuple, int] = {}
        for obj in objectives:
            coord_tuple = tuple(obj)
            coord_dict[coord_tuple] = coord_dict.get(coord_tuple, 0) + 1

        # 转换为数组
        unique_coords = np.array(list(coord_dict.keys()))
        counts = np.array(list(coord_dict.values()))

        return unique_coords, counts


def base_nsgaii_solver(
    population: "Population",
    instance: "InstanceClass",
    use_niching: bool = True,
    use_adaptive_selection: bool = True,
    plot_gif: bool = False,
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    solver = BASICNSGAII(
        population,
        instance,
        use_niching=use_niching,
        use_adaptive_selection=use_adaptive_selection,
    )
    final_population = solver.solve()

    # debug: 生成目标函数进化GIF
    if plot_gif:
        solver.plot_generation_objectives_gif(
            save_path="nsgaii_evolution.gif",
            fps=2,
            dpi=100,
            figsize=(10, 8),
            show_pareto_front=True,
        )
    return final_population


def basic_nsgaii_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    final_population = base_nsgaii_solver(
        population,
        instance,
        use_niching=False,
        use_adaptive_selection=False,
    )
    return final_population


def nsgaii_niching_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    """ALNS-NSGA-II求解器入口函数"""
    final_population = base_nsgaii_solver(
        population,
        instance,
        use_niching=True,
        use_adaptive_selection=False,
    )
    return final_population


def nsgaii_adaptive_select_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    """ALNS-NSGA-II求解器入口函数"""
    final_population = base_nsgaii_solver(
        population,
        instance,
        use_niching=False,
        use_adaptive_selection=True,
    )
    return final_population


def nsgaii_niching_adaptive_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """ALNS-NSGA-II求解器入口函数"""
    final_population = base_nsgaii_solver(
        population,
        instance,
        use_niching=True,
        use_adaptive_selection=True,
    )
    return final_population
