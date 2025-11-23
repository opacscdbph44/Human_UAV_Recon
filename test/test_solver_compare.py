import random
from datetime import datetime

import numpy as np

from src.config import (
    AlgorithmConfig,
    Config,
    InstanceParam,
)
from src.solver import Solver_Name
from src.utils import compare_solvers


def main(
    instance_name: str = "a280.tsp",
    demand_nums: list[int] | None = None,
    solver_names: list[Solver_Name] | None = None,
    random_seed: int = 42,
):
    """比较不同求解器在不同算例规模下的求解结果

    Args:
        instance_name (str): 算例名称，默认为"a280.tsp"
        demand_nums (list[int]): 算例规模列表，默认为[5, 6, 7, 8, 9, 10]
        solver_names (list[str]): 求解器名称列表，默认为["alns", "gurobi_single_obj_solver"]
        random_seed (int): 随机种子，默认为42
    """
    # 主函数起点，声明程序开始
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: " "开始运行求解器比较程序！"
    )

    # 设置默认值
    if demand_nums is None:
        demand_nums = [5, 6, 7, 8, 9, 10]

    if solver_names is None:
        solver_names = [
            Solver_Name.GUROBI_SINGLE_OBJ,
            Solver_Name.ALNS,
            Solver_Name.MS_ALNS,
        ]

    # 固定随机种子，保证结果可复现
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"设置随机种子: {random_seed}"
    )

    # 创建配置模板
    prob_config_template = Config(
        instance_param=InstanceParam(
            name=instance_name,
            demand_num=5,  # 这个值会在compare_solvers中被覆盖
        ),
        algorithm_config=AlgorithmConfig(
            max_iter=1000,
        ),
        random_seed=random_seed,
    )

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: " f"配置模板创建成功")

    # 执行比较
    compare_solvers(
        instance_name=instance_name,
        demand_nums=demand_nums,
        solver_names=solver_names,
        prob_config_template=prob_config_template,
    )

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: " "求解器比较程序执行完毕！"
    )


if __name__ == "__main__":
    # 示例1: 使用默认参数
    main(
        instance_name="a280.tsp",
        demand_nums=[
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            30,
            40,
            50,
            100,
        ],
        solver_names=[
            # Solver_Name.GUROBI_SINGLE_OBJ,
            Solver_Name.ALNS,
            Solver_Name.MS_ALNS,
            # Solver_Name.RANDOMIZED_GREEDY,
            # Solver_Name.EFFICIENT_GREEDY,
            # Solver_Name.GROUND_FIRST_GREEDY,
            # Solver_Name.DRONE_FIRST_GREEDY,
        ],
    )
