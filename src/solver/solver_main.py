from enum import Enum

from src.element import InstanceClass, Solution
from src.solver.alns import alns_solve
from src.solver.consr_heuristic import (
    drone_first_greedy_heuristic,
    efficient_greedy_heuristic,
    ground_first_greedy_heuristic,
    multi_start_heuristic,
    randomized_greedy_heuristic,
)
from src.solver.mip_gurobi import (
    gurobi_single_obj_solver,
)
from src.solver.ms_alns import ms_alns_solve

# 点搜索精确算法
single_point_exact_list = {
    # 精确求解去，使用Gurobi求解单目标ECCP问题
    "gurobi_single_obj_solver": gurobi_single_obj_solver,
}
# 点搜索启发式算法
single_point_heuri_list = {
    # 启发式算法
    # Construction-based
    "randomized_greedy": randomized_greedy_heuristic,
    "efficient_greedy": efficient_greedy_heuristic,
    "ground_first_greedy": ground_first_greedy_heuristic,
    "drone_first_greedy": drone_first_greedy_heuristic,
    # Multi-start Construction-based
    "multi_start_heuristic": multi_start_heuristic,
    # ALNS
    "ALNS": alns_solve,
    # MS-ALNS
    "MS_ALNS": ms_alns_solve,
}


class Solver_Name(str, Enum):
    GUROBI_SINGLE_OBJ = "gurobi_single_obj_solver"
    GUROBI_EPSILON_CONS = "gurobi_epsilon_constraint_solver"
    GREEDY_RANDOMIZED_HEURI = "greedy_randomized_heuristic"
    MULTI_GREEDY_RANDOMIZED_HEURI = "multi_randomized_greedy_heuristic"
    MULTI_START_HEURISTIC = "multi_start_heuristic"
    ALNS = "ALNS"
    MS_ALNS = "MS_ALNS"
    RANDOMIZED_GREEDY = "randomized_greedy"
    EFFICIENT_GREEDY = "efficient_greedy"
    GROUND_FIRST_GREEDY = "ground_first_greedy"
    DRONE_FIRST_GREEDY = "drone_first_greedy"


# 求解器入口，初始化初始解，根据输入参数设定求解器，将solution和求解器设定传入具体调用的求解器
def solve(
    solver_name: Solver_Name,
    instance: InstanceClass,
) -> Solution:

    # 根据solver_type选择相应的求解器
    if solver_name in single_point_exact_list:
        # 初始化一个解
        solution = Solution.new_solution(instance, solver_name=solver_name)
        # 获取求解器
        single_exact_solver = single_point_exact_list[solver_name]
        # 调用求解器，传入初始解、问题实例和算法参数
        solution = single_exact_solver(solution, instance)
        return solution
    elif solver_name in single_point_heuri_list:
        # 初始化一个解
        solution = Solution.new_solution(instance, solver_name=solver_name)
        # 获取求解器
        single_heuri_solver = single_point_heuri_list[solver_name]
        # 调用求解器，传入初始解、问题实例和算法参数
        solution = single_heuri_solver(solution, instance)
        return solution
    else:
        raise ValueError(f"未知的求解器类型: {solver_name}")
