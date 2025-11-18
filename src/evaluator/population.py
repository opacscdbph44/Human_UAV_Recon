from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import numpy as np

from src.config.prob_config import OptimizeSense
from src.element import Solution
from src.evaluator.evaluate import Evaluator


@dataclass
class Population:
    """
    种群类,管理一群解,提供NSGA-II所需的操作
    """

    size: int
    solver_name: str
    evaluator: Evaluator
    solve_time: float = -1.0
    solutions: List[Solution] = field(default_factory=list)
    sorted: bool = False
    fronts_idx: List[List[int]] = field(default_factory=list)
    elite_size: int = -1
    elite_idx: List[int] = field(default_factory=list)
    hv_calculated: bool = False
    hv_value: float = -1.0
    hv_history: List[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Population(size={self.size}, elite_size={self.elite_size})"

    __repr__ = __str__

    def copy(self) -> "Population":
        """只拷贝种群中的必要信息，避免耗时"""
        new_population = Population(self.size, self.solver_name, self.evaluator)
        new_population.solutions = [sol.copy() for sol in self.solutions]
        return new_population

    def add_solution(self, solution: Solution) -> None:
        """添加解到种群"""
        self.solutions.append(solution)

    def evaluate_all(self) -> None:
        """评估所有解"""
        for sol in self.solutions:
            self.evaluator.sol_evaluate(sol)

    def fast_non_dominated_sort_vectorized(
        self,
        tolerance: float = 1e-6,
    ) -> List[List[int]]:
        """快速非支配排序(NumPy向量化版本)

        Args:
            tolerance: 判断目标函数值是否相同的容差值

        Returns:
            List[List[int]]: 每层front的解索引列表
        """
        n = len(self.solutions)
        if n == 0:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[warning] 当前种群解数量为{n}！"
                "返回空列表'[[]]'"
            )
            return [[]]

        # 将所有目标值转换为NumPy数组 (n_solutions × n_objectives)
        objectives: np.ndarray = np.array([sol.objectives for sol in self.solutions])

        # 检查无穷大
        if np.any(np.isinf(objectives)):
            raise ValueError("解的目标函数值包含无穷大")

        # 向量化计算支配关系矩阵
        dominates = self._compute_domination_matrix(
            objectives,
            tolerance,
        )

        # 计算每个解被支配的次数
        domination_count = dominates.sum(axis=0)

        # 找出第一层(未被支配的解)
        fronts_idx: List[List[int]] = [[]]  # 存储索引
        current_front_indices = np.where(domination_count == 0)[0]

        # 对第一层解进行目标函数值去重（基于容差）
        unique_elite_indices: list = []
        seen_objectives: list = []

        for idx in current_front_indices:
            obj_current = objectives[idx]  # 直接使用numpy数组,更快
            is_duplicate = False

            for obj_seen in seen_objectives:
                if np.all(np.abs(obj_current - obj_seen) <= tolerance):
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_objectives.append(obj_current)
                unique_elite_indices.append(int(idx))  # 转为Python int

        # 更新精英解信息
        self.elite_idx = unique_elite_indices
        self.elite_size = len(unique_elite_indices)

        # 将所有第一层的索引添加到front,并更新rank
        for idx in current_front_indices:
            self.solutions[idx].rank = 0
            fronts_idx[0].append(int(idx))

        # 构建后续层级
        rank = 0
        while len(current_front_indices) > 0:
            dominated_by_current = dominates[current_front_indices, :].any(axis=0)
            domination_count -= dominates[current_front_indices, :].sum(axis=0)

            next_front_indices = np.where(
                (domination_count == 0) & dominated_by_current
            )[0]

            if len(next_front_indices) == 0:
                break

            rank += 1
            next_front = []
            for idx in next_front_indices:
                self.solutions[idx].rank = rank
                next_front.append(int(idx))

            fronts_idx.append(next_front)
            current_front_indices = next_front_indices
        # 把fronts_idx存在self中，防止反复计算支配关系
        self.fronts_idx = fronts_idx
        self.sorted = True
        return fronts_idx

    def _compute_domination_matrix(
        self,
        objectives: np.ndarray,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        """
        向量化计算支配关系矩阵(考虑优化方向和容差)

        Args:
            objectives: 目标值矩阵 (n_solutions × n_objectives)
            tolerance: 判断目标函数值是否相同的容差值

        返回: dominates[i, j] = True 表示解i支配解j
        时间复杂度: O(n² × m) 但利用NumPy向量化,实际速度快很多
        """

        senses = self.evaluator.obj_config.sense

        # 扩展维度用于广播比较
        obj_i = objectives[:, np.newaxis, :]  # (n, 1, m)
        obj_j = objectives[np.newaxis, :, :]  # (1, n, m)

        # 计算差值
        diff = obj_i - obj_j  # (n, n, m)

        # 根据优化方向构建比较矩阵 (n, n, m)
        better = np.zeros_like(diff, dtype=bool)
        worse = np.zeros_like(diff, dtype=bool)

        for m, sense in enumerate(senses):
            if sense == OptimizeSense.MINIMIZE:
                # 最小化: i比j小超过容差才算更好
                better[:, :, m] = diff[:, :, m] < -tolerance
                worse[:, :, m] = diff[:, :, m] > tolerance
            else:  # OptimizeSense.MAXIMIZE
                # 最大化: i比j大超过容差才算更好
                better[:, :, m] = diff[:, :, m] > tolerance
                worse[:, :, m] = diff[:, :, m] < -tolerance

        # i支配j的条件:
        # 1. i在所有目标上都不比j差: ~worse.any(axis=2)
        # 2. i在至少一个目标上比j好: better.any(axis=2)
        not_worse = ~worse.any(axis=2)
        at_least_one_better = better.any(axis=2)
        dominates: np.ndarray = np.asarray(not_worse & at_least_one_better)

        return dominates

    def _dominates_vectorized(self, sol1: Solution, sol2: Solution) -> np.bool_:
        """单个解对的支配判断(向量化版本,考虑优化方向)"""

        obj1 = np.array(sol1.objectives)
        obj2 = np.array(sol2.objectives)

        # 检查无穷大
        if np.any(np.isinf(obj1)) or np.any(np.isinf(obj2)):
            raise ValueError(
                f"解的目标函数值无穷大: sol1={sol1.objectives}, sol2={sol2.objectives}"
            )

        senses = self.evaluator.obj_config.sense

        # 根据优化方向比较
        better = np.zeros(len(obj1), dtype=bool)
        worse = np.zeros(len(obj1), dtype=bool)

        for m, sense in enumerate(senses):
            if sense == OptimizeSense.MINIMIZE:
                better[m] = obj1[m] < obj2[m]
                worse[m] = obj1[m] > obj2[m]
            else:  # OptimizeSense.MAXIMIZE
                better[m] = obj1[m] > obj2[m]
                worse[m] = obj1[m] < obj2[m]

        # sol1支配sol2: 所有目标不更差 且 至少一个目标更好
        dominate = (~worse.any()) and (better.any())
        return dominate

    def calculate_crowding_distance(self, front: List[Solution]):
        """计算拥挤度距离"""
        if len(front) == 0:
            raise ValueError("前沿解集不能为空")

        num_objectives = len(front[0].objectives)

        # 初始化拥挤度距离
        for sol in front:
            sol.crowding_distance = 0.0

        # 对每个目标计算拥挤度
        for m in range(num_objectives):
            # 按第m个目标排序
            front.sort(key=lambda x: x.objectives[m])

            # 边界解设为无穷大
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            obj_min = front[0].objectives[m]
            obj_max = front[-1].objectives[m]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # 中间解的拥挤度
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[m] - front[i - 1].objectives[m]
                ) / obj_range

    def select_best(self, n: int) -> List[Solution]:
        """
        选择最优的n个解(基于rank和拥挤度)

        Args:
            n: 需要选择的解数量

        Returns:
            选择的解列表
        """
        if not self.sorted:
            self.fast_non_dominated_sort_vectorized()

        all_front_idx = self.fronts_idx

        selected: List[Solution] = []
        for current_front_idx in all_front_idx:
            # 根据索引提取当前层的解
            current_front_solutions = self.get_solutions_by_indices(current_front_idx)

            if len(selected) + len(current_front_solutions) <= n:
                selected.extend(current_front_solutions)
            else:
                # 当前层不能全部加入,按拥挤度选择
                self.calculate_crowding_distance(current_front_solutions)
                current_front_solutions.sort(
                    key=lambda x: x.crowding_distance, reverse=True
                )
                selected.extend(current_front_solutions[: n - len(selected)])
                break

        return selected

    def get_solutions_by_indices(self, indices: List[int]) -> List[Solution]:
        """根据索引列表提取对应的解

        Args:
            indices: 解的索引列表

        Returns:
            对应的解列表
        """
        return [self.solutions[idx] for idx in indices]

    def get_pareto_front(self) -> List[Solution]:
        """获取Pareto前沿(rank=0的解)"""
        # 如果没有排序，则排序
        if not self.sorted:
            self.fast_non_dominated_sort_vectorized()

        # 提取每个前沿面的索引列表
        fronts_idx = self.fronts_idx

        # 如果第一个前沿面上存在至少一个解
        if fronts_idx[0]:
            # 返回rank=0的解列表
            return self.get_solutions_by_indices(fronts_idx[0])
        else:
            # 报错信息
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[warning] 当前种群无Pareto前沿解！返回空列表'[]'"
            )
            # 返回空列表
            return []

    def get_elite_solutions(self) -> List[Solution]:
        """获取精英解个体(去重后的rank=0解)"""
        if not self.sorted:
            self.fast_non_dominated_sort_vectorized()

        elite_solutions = self.get_solutions_by_indices(self.elite_idx)
        return elite_solutions

    def calculate_hypervolume(
        self,
        reference_point: List[float] = [2.0, 0.0],
    ) -> float:
        """计算2D超体积，支持混合优化方向"""
        if self.hv_calculated:
            return self.hv_value

        pareto_front = self.get_pareto_front()
        if len(pareto_front) == 0:
            raise ValueError("种群中没有Pareto前沿解，无法计算超体积")

        # 提取目标值
        objs = np.array([sol.objectives for sol in pareto_front])
        senses = self.evaluator.obj_config.sense
        ref = np.array(reference_point, dtype=float)

        # 统一转换为最小化问题
        for i in range(2):
            if senses[i] == OptimizeSense.MAXIMIZE:
                objs[:, i] = -objs[:, i]
                ref[i] = -ref[i]

        # 过滤:只保留被参考点支配的解
        diff = ref - objs  # 参考点 - 目标值
        valid_mask = np.all(diff >= 0, axis=1)  # 参考点在所有维度上都大于目标值
        invalid_count = np.sum(~valid_mask)

        if invalid_count > 0:
            invalid_objs = objs[~valid_mask]
            raise ValueError(
                f"发现 {invalid_count} 个解比负理想点还差！\n"
                f"参考点（负理想点）: {reference_point}\n"
                f"问题解的目标值:\n{invalid_objs}"
            )

        valid = objs[valid_mask]

        # 按第一个目标排序
        valid = valid[np.argsort(valid[:, 0])]

        # 计算超体积
        hv = 0.0
        for i in range(len(valid)):
            x, y = valid[i]
            # 确定矩形的宽度：到下一个解或到参考点
            if i < len(valid) - 1:
                width = valid[i + 1, 0] - x
            else:
                width = ref[0] - x

            # 矩形高度：参考点到当前解的距离
            height = ref[1] - y

            # 累加矩形面积
            hv += width * height

        self.hv_value = float(hv)
        self.hv_calculated = True
        return self.hv_value

    def merge(self, other: "Population") -> "Population":
        """
        合并当前种群与另一个种群

        Args:
            other: 另一个种群

        Returns:
            新的合并种群，保留目标函数值，重置种群相关属性
        """
        if self.solver_name != other.solver_name:
            raise ValueError(
                f"两个种群的solver_name不一致: {self.solver_name} vs {other.solver_name}"
            )

        new_size = self.size + other.size
        new_population = Population(
            size=new_size, solver_name=self.solver_name, evaluator=self.evaluator
        )

        # copy()会保留objectives，但重置rank和crowding_distance
        new_population.solutions = [sol.copy() for sol in self.solutions] + [
            sol.copy() for sol in other.solutions
        ]

        return new_population

    @classmethod
    def merge_populations(cls, pop1: "Population", pop2: "Population") -> "Population":
        """
        合并两个种群（类方法）

        Args:
            pop1: 第一个种群
            pop2: 第二个种群

        Returns:
            新的合并种群，保留目标函数值，重置种群相关属性
        """
        if pop1.solver_name != pop2.solver_name:
            raise ValueError(
                f"两个种群的solver_name不一致: {pop1.solver_name} vs {pop2.solver_name}"
            )

        new_size = pop1.size + pop2.size
        new_population = cls(
            size=new_size, solver_name=pop1.solver_name, evaluator=pop1.evaluator
        )

        # copy()会保留objectives，但重置rank和crowding_distance
        new_population.solutions = [sol.copy() for sol in pop1.solutions] + [
            sol.copy() for sol in pop2.solutions
        ]

        return new_population
