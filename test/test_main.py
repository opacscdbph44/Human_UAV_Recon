import random
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import (
    AlgorithmConfig,
    Config,
    InstanceParam,
)
from src.element import Solution
from src.evaluator import Population
from src.solver import Solver_Name, solve
from src.utils import (
    load_instance,
    plot_instance,
    query_result_folder,
    save_population_results,
    save_solution,
)

PLOT_INSTANCE_FIGURE = False
SAVE_RESULT_JSON = True


def main(
    solver_name: Solver_Name,
    demand_num: int = 5,
    random_seed: int = 42,
    json_folder: str | None = None,
    figure_folder: str | None = None,
):
    # 主函数起点，声明程序开始
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:"
        "开始运行 多车辆通信覆盖路径规划 主程序！"
    )

    # =============== 初始化配置 ===============
    prob_config = Config(
        instance_param=InstanceParam(
            name="a280.tsp",
            demand_num=demand_num,
        ),
        algorithm_config=AlgorithmConfig(
            max_iter=200,
        ),
        random_seed=random_seed,
    )
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"加载配置成功！ {prob_config}"
    )

    # 固定随机种子，保证结果可复现
    seed = prob_config.random_seed
    random.seed(seed)
    np.random.seed(seed)
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"为保证实验结果复现，设置随机种子: {seed}"
    )

    # 读取算例文件
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"读取算例文件: {prob_config.instance_param.name}"
    )
    instance = load_instance(prob_config)

    if PLOT_INSTANCE_FIGURE:
        if figure_folder is not None:
            figure_folder_path = Path(figure_folder)
        else:
            figure_folder_path = None
        _ = plot_instance(instance=instance, figure_folder_path=figure_folder_path)

    # ====================================
    #             进入solver
    # ====================================
    # 打印开始求解
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"使用求解器 {solver_name.value} 进行求解..."
    )
    solve_result = solve(
        solver_name=solver_name,
        instance=instance,
    )

    # ====================================
    #           获得solution
    # ====================================
    if isinstance(solve_result, Solution):
        solution = solve_result
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"获得单解求解结果: {solution}"
        )
        if SAVE_RESULT_JSON:
            # 查找结果存储文件夹
            if json_folder is not None:
                result_folder = Path(json_folder)
            else:
                result_folder = query_result_folder(prob_config, solver_name.value)

            # 存储结果
            save_solution(
                solution,
                folder_path=result_folder,
            )

    elif isinstance(solve_result, Population):
        population = solve_result
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"获得群解求解结果,种群规模: {population.size}"
        )
        # 查找结果存储文件夹
        if SAVE_RESULT_JSON:
            if json_folder is not None:
                result_folder = Path(json_folder)
            else:
                result_folder = query_result_folder(prob_config, solver_name.value)

            # 存储结果
            save_population_results(population, result_folder)
    else:
        raise TypeError(f"求解结果类型错误！{type(solve_result)}")


if __name__ == "__main__":
    for random_seed in [
        42,
        # 13,
        # 78,
        # 99,
        # 6,
        # 72,
        # 53,
        # 30,
        # 32,
        # 2025,
    ]:
        for solver_name in [
            # Solver_Name.GUROBI_EPSILON_CONS,
            # Solver_Name.GREEDY_RANDOMIZED_HEURI,
            # Solver_Name.BASIC_NSGA_II,
            Solver_Name.NSGA_II,
            Solver_Name.NSGA_II_n,
            Solver_Name.ALNS_NSGA_II,
            Solver_Name.ALNS_NSGA_II_n,
            Solver_Name.MOEAD,
        ]:
            figure_folder = ""
            for scale in [
                5,
                6,
                7,
                8,
                9,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
            ]:
                main(
                    solver_name,
                    demand_num=scale,
                    random_seed=random_seed,
                )
