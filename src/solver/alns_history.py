"""ALNS算法历史记录类"""

from typing import Any


class ALNSHistory:
    """ALNS算法迭代历史记录器"""

    def __init__(self):
        """初始化历史记录器"""
        self.iteration_history: list[dict[str, Any]] = []

    def record_initial(
        self,
        best_objective: float,
        current_objective: float,
    ):
        """
        记录初始解信息

        Args:
            best_objective: 最优目标函数值
            current_objective: 当前解目标函数值
        """
        self.iteration_history.append(
            {
                "iteration": 0,
                "best_objective": best_objective,
                "current_objective": current_objective,
                "candidate_objective": current_objective,
                "destroy_operator": None,
                "repair_operator": None,
                "accepted": True,
            }
        )

    def record_iteration(
        self,
        iteration: int,
        best_objective: float,
        current_objective: float,
        candidate_objective: float,
        destroy_operator: str,
        repair_operator: str,
        accepted: bool,
        destroy_degree: float | None = None,
        temperature: float | None = None,
        **kwargs,
    ):
        """
        记录单次迭代信息

        Args:
            iteration: 当前迭代次数
            best_objective: 当前最优目标函数值
            current_objective: 当前解目标函数值
            candidate_objective: 候选解(新解)目标函数值
            destroy_operator: 使用的破坏算子名称
            repair_operator: 使用的修复算子名称
            accepted: 是否接受新解
            destroy_degree: 破坏程度(可选)
            temperature: 模拟退火温度(可选)
            **kwargs: 其他自定义记录字段
        """
        record = {
            "iteration": iteration,
            "best_objective": best_objective,
            "current_objective": current_objective,
            "candidate_objective": candidate_objective,
            "destroy_operator": destroy_operator,
            "repair_operator": repair_operator,
            "accepted": accepted,
        }

        # 添加可选字段
        if destroy_degree is not None:
            record["destroy_degree"] = destroy_degree

        if temperature is not None:
            record["temperature"] = temperature

        # 添加额外自定义字段
        record.update(kwargs)

        self.iteration_history.append(record)

    def get_history(self) -> list[dict[str, Any]]:
        """
        获取完整历史记录

        Returns:
            历史记录列表
        """
        return self.iteration_history

    def get_best_objectives(self) -> list[float]:
        """
        获取历代最优目标函数值序列

        Returns:
            最优目标函数值列表
        """
        return [record["best_objective"] for record in self.iteration_history]

    def get_current_objectives(self) -> list[float]:
        """
        获取历代当前解目标函数值序列

        Returns:
            当前解目标函数值列表
        """
        return [record["current_objective"] for record in self.iteration_history]

    def get_candidate_objectives(self) -> list[float]:
        """
        获取历代候选解目标函数值序列

        Returns:
            候选解目标函数值列表
        """
        return [record["candidate_objective"] for record in self.iteration_history]

    def get_acceptance_rate(self) -> float:
        """
        计算总体接受率

        Returns:
            接受率(0-1之间)
        """
        if len(self.iteration_history) <= 1:
            return 1.0

        # 排除初始解记录
        iterations = self.iteration_history[1:]
        accepted_count = sum(1 for record in iterations if record["accepted"])
        return accepted_count / len(iterations) if iterations else 0.0

    def get_operator_usage(self) -> dict[str, dict[str, int]]:
        """
        统计各操作算子使用次数

        Returns:
            {
                "destroy": {"operator1": count1, ...},
                "repair": {"operator1": count1, ...}
            }
        """
        destroy_usage = {}
        repair_usage = {}

        for record in self.iteration_history[1:]:  # 排除初始解
            destroy_op = record.get("destroy_operator")
            repair_op = record.get("repair_operator")

            if destroy_op:
                destroy_usage[destroy_op] = destroy_usage.get(destroy_op, 0) + 1

            if repair_op:
                repair_usage[repair_op] = repair_usage.get(repair_op, 0) + 1

        return {"destroy": destroy_usage, "repair": repair_usage}

    def clear(self):
        """清空历史记录"""
        self.iteration_history.clear()
