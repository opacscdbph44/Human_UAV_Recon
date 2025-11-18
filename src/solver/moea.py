import random
import time
from datetime import datetime
from typing import List

import numpy as np

from src.element import (
    InstanceClass,
)
from src.element.solution import Solution
from src.evaluator import Population
from src.solver.nsgaii import NSGAII


class MOEAD(NSGAII):
    """基于NSGAII改造一个MOEAD"""

    def __init__(
        self,
        population: "Population",
        instance: "InstanceClass",
        use_niching: bool = False,
    ):
        """初始化MOEAD算法实例

        Args:
            population (list): 初始种群
            instance (_type_): 问题实例
            use_niching (bool): 是否使用小生境技术进行精英选择
        """
        super().__init__(
            population,
            instance,
            use_niching,
        )

        self.neighborhood_size = 20  # 邻域大小

        # MOEA/D特有: 生成权重向量和邻域结构
        self.weight_vectors = self._generate_weight_vectors()  # 生成均匀分布的权重向量
        self.neighborhoods = self._initialize_neighborhoods()  # 基于权重向量计算邻域

    def solve(self) -> Population:
        """执行MOEAD算法求解过程

        Returns:
            Population: 最终种群
        """

        # 开始计时
        start_time = time.perf_counter()
        # 初始化种群
        current_pop = self._init_population()

        # 初始化HV记录列表
        hv_history = []

        # 迭代进化
        for iteration in range(self.max_iter):
            # 记录当前代的HV值 - 使用Population类的内置方法
            current_hv = current_pop.calculate_hypervolume()
            current_hv_info = {
                "iteration": iteration,
                "hypervolume": current_hv,
            }

            hv_history.append(current_hv_info)

            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"Iteration {iteration:3d}: Hypervolume = {current_hv:.4f}"
            )

            # 生成子代
            child_pop = self._generate_offspring(current_pop)

            # 评估子代
            child_pop.evaluate_all()

            # 计算进化进度
            progress = iteration / self.max_iter

            # 使用切比雪夫精英选择(MOEA/D特有)
            new_pop = self.Chebyshev_elitist_selection(current_pop, child_pop, progress)

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

        return current_pop

    def basic_generate_offspring(self, parent_pop: Population) -> Population:
        """基于MOEAD的邻域交叉变异生成子代种群

        Args:
            parent_pop (Population): 父代种群

        Returns:
            Population: 子代种群
        """
        child_size = 0
        child_pop = Population(
            size=0,
            solver_name=parent_pop.solver_name,
            evaluator=parent_pop.evaluator,
        )

        # 为每个子问题生成一个子代
        while child_size < self.pop_size:
            for i in range(self.pop_size):
                # 获取当前子问题的邻域索引
                neighbor_indices = self.neighborhoods[i]

                # 从邻域中随机选择两个个体作为父母
                p1_idx, p2_idx = np.random.choice(
                    neighbor_indices, size=2, replace=False
                )
                parent1 = parent_pop.solutions[p1_idx]
                parent2 = parent_pop.solutions[p2_idx]

                # 交叉操作生成子代
                child1, child2 = self._crossover(parent1, parent2)

                # 变异操作
                mutated_child1 = self.basic_mutate(child1)
                mutated_child2 = self.basic_mutate(child2)
                # 记录新获得的解
                child_pop.add_solution(mutated_child1)
                child_pop.add_solution(mutated_child2)
                child_size += 2

        return child_pop

    def basic_mutate(self, solution: Solution) -> Solution:
        # 复制解
        mutated_solution = solution.copy()

        # 随机选择变异类型(添加跨车辆变异类型)
        mutate_types = [
            # 单车辆变异
            "swap",
            "inversion",
            "insertion",
        ]
        mutate_type = random.choice(mutate_types)

        # 单车辆变异
        mutated_solution = self._single_vehicle_mutation(mutated_solution, mutate_type)

        mutated_solution._init_attrs()  # 重新初始化解属性

        return mutated_solution

    def _generate_weight_vectors(self) -> np.ndarray:
        """生成均匀分布的权重向量(使用简单线性方法)

        MOEA/D将多目标优化问题分解为多个单目标子问题,
        每个子问题由一个权重向量定义

        Returns:
            np.ndarray: 权重向量矩阵,形状为(pop_size, n_objectives)
        """
        # 假设是双目标问题
        n_objectives = 2
        weight_vectors = np.zeros((self.pop_size, n_objectives))

        # 生成均匀分布的权重向量
        for i in range(self.pop_size):
            # 第一个目标的权重从0到1均匀分布
            w1 = i / (self.pop_size - 1)
            w2 = 1 - w1
            weight_vectors[i] = [w1, w2]

        return weight_vectors

    def _initialize_neighborhoods(self) -> List[List[int]]:
        """基于权重向量的欧氏距离计算邻域结构

        在MOEA/D中,邻域是在权重向量空间中定义的,
        相近的权重向量对应相似的子问题

        Returns:
            List[List[int]]: 每个子问题的邻域索引列表
        """
        neighborhoods = []

        # 计算所有权重向量对之间的欧氏距离
        for i in range(self.pop_size):
            # 计算当前权重向量与所有其他权重向量的距离
            distances = np.linalg.norm(
                self.weight_vectors - self.weight_vectors[i], axis=1
            )

            # 获取距离最近的T个邻居的索引(包括自己)
            neighbor_indices = np.argsort(distances)[: self.neighborhood_size]
            neighborhoods.append(neighbor_indices.tolist())

        return neighborhoods

    def Chebyshev_elitist_selection(
        self, current_pop: Population, child_pop: Population, progress: float = 0.0
    ) -> Population:
        """使用切比雪夫分解方法进行精英选择

        MOEA/D的核心选择策略:对每个子问题,使用切比雪夫聚合函数
        比较当前解和新生成的解,保留更优的解

        Args:
            current_pop: 当前种群
            child_pop: 子代种群
            progress: 当前进化进度(0-1)

        Returns:
            Population: 更新后的种群
        """
        # 合并当前种群和子代种群
        combined_solutions = current_pop.solutions + child_pop.solutions

        # 计算理想点(根据优化方向确定)
        all_objectives = np.array([sol.objectives for sol in combined_solutions])
        ideal_point = np.zeros(all_objectives.shape[1])

        # 获取目标函数的优化方向
        senses = current_pop.evaluator.obj_config.sense

        # 根据每个目标的优化方向计算理想点
        for i, sense in enumerate(senses):
            if sense.value == "minimize":
                # 最小化目标: 理想点是最小值
                ideal_point[i] = np.min(all_objectives[:, i])
            else:  # maximize
                # 最大化目标: 理想点是最大值
                ideal_point[i] = np.max(all_objectives[:, i])

        # 为每个子问题选择最优解
        selected_solutions = []

        for i in range(self.pop_size):
            # 获取当前子问题的权重向量
            weight = self.weight_vectors[i]

            # 获取邻域内的解(从合并的种群中)
            neighbor_indices = self.neighborhoods[i]

            # 计算邻域内所有解的切比雪夫函数值
            best_solution = None
            best_tchebycheff = float("inf")

            # 考虑邻域内的解 + 当前子问题对应的子代解
            candidate_indices = set(neighbor_indices)
            # 添加当前子问题对应的子代(如果存在)
            if i < len(current_pop.solutions):
                candidate_indices.add(i)
            if i < len(child_pop.solutions):
                candidate_indices.add(len(current_pop.solutions) + i)

            for idx in candidate_indices:
                if idx < len(combined_solutions):
                    solution = combined_solutions[idx]
                    # 计算切比雪夫函数值
                    tchebycheff_value = self._tchebycheff(
                        np.array(solution.objectives), weight, ideal_point
                    )

                    if tchebycheff_value < best_tchebycheff:
                        best_tchebycheff = tchebycheff_value
                        best_solution = solution

            if best_solution is not None:
                selected_solutions.append(best_solution)

        # 创建新种群
        new_pop = Population(
            size=len(selected_solutions),
            solver_name=current_pop.solver_name,
            evaluator=current_pop.evaluator,
        )

        # 添加选中的解
        for sol in selected_solutions:
            new_pop.add_solution(sol.copy())

        return new_pop

    def _tchebycheff(
        self, objectives: np.ndarray, weight: np.ndarray, ideal_point: np.ndarray
    ) -> float:
        """计算切比雪夫聚合函数值

        切比雪夫函数: g(x|λ,z*) = max{λ_i * |f_i(x) - z_i*|}
        其中:
        - λ是权重向量
        - z*是理想点
        - f_i(x)是目标函数值

        Args:
            objectives: 解的目标函数值
            weight: 权重向量
            ideal_point: 理想点

        Returns:
            float: 切比雪夫函数值(越小越好)
        """
        # 避免权重为0导致的问题
        adjusted_weight = np.where(weight < 1e-6, 1e-6, weight)

        # 计算加权的目标值偏差
        weighted_diff = adjusted_weight * np.abs(objectives - ideal_point)

        # 返回最大值(切比雪夫距离)
        return np.max(weighted_diff)


def moead_solver(
    population: "Population",
    instance: "InstanceClass",
) -> "Population":
    """MOEA/D求解器入口函数"""
    solver = MOEAD(
        population,
        instance,
    )
    final_population = solver.solve()
    return final_population
