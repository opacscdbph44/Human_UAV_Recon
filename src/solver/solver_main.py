from enum import Enum

from src.element import InstanceClass, Solution
from src.evaluator import Evaluator, Population
from src.solver.alns_nsgaii import (
    alns_nsgaii_adaptive_select_solver,
    alns_nsgaii_niching_adaptive_solver,
    alns_nsgaii_niching_solver,
    alns_nsgaii_solver,
)
from src.solver.basic_nsgaii import basic_nsgaii_solver
from src.solver.consr_heuristic import (
    multi_randomized_greedy_heuristic,
    randomized_greedy_heuristic,
)
from src.solver.mip_gurobi import (
    gurobi_epsilon_constraint_solver,
    gurobi_single_obj_solver,
)
from src.solver.moea import moead_solver
from src.solver.nsgaii import (
    nsgaii_adaptive_select_solver,
    nsgaii_niching_adaptive_solver,
    nsgaii_niching_solver,
    nsgaii_solver,
)

# 点搜索精确算法
single_point_exact_list = {
    # 精确求解去，使用Gurobi求解单目标ECCP问题
    "gurobi_single_obj_solver": gurobi_single_obj_solver,
}
# 点搜索启发式算法
single_point_heuri_list = {
    # 启发式算法
    # Construction-based
    "greedy_randomized_heuristic": randomized_greedy_heuristic,
}

# 群搜索精确算法
group_exact_list = {
    # 精确求解多目标ECCP问题，采用ε-约束法
    "gurobi_epsilon_constraint_solver": gurobi_epsilon_constraint_solver,
}

# 群搜索启发式算法
group_heuri_list = {
    # Population-based initialization
    "multi_randomized_greedy_heuristic": multi_randomized_greedy_heuristic,
    # Population-based meta-heuristic
    "BASIC_NSGA_II": basic_nsgaii_solver,
    "NSGA_II": nsgaii_solver,
    "NSGA_II_n": nsgaii_niching_solver,
    "NSGA_II_as": nsgaii_adaptive_select_solver,
    "NSGA_II_nas": nsgaii_niching_adaptive_solver,
    "ALNS_NSGA_II": alns_nsgaii_solver,
    "ALNS_NSGA_II_n": alns_nsgaii_niching_solver,
    "ALNS_NSGA_II_as": alns_nsgaii_adaptive_select_solver,
    "ALNS_NSGA_II_nas": alns_nsgaii_niching_adaptive_solver,
    "MOEAD": moead_solver,
}


class Solver_Name(str, Enum):
    GUROBI_SINGLE_OBJ = "gurobi_single_obj_solver"
    GUROBI_EPSILON_CONS = "gurobi_epsilon_constraint_solver"
    GREEDY_RANDOMIZED_HEURI = "greedy_randomized_heuristic"
    MULTI_GREEDY_RANDOMIZED_HEURI = "multi_randomized_greedy_heuristic"
    BASIC_NSGA_II = "BASIC_NSGA_II"
    NSGA_II = "NSGA_II"
    NSGA_II_n = "NSGA_II_n"
    NSGA_II_as = "NSGA_II_as"
    NSGA_II_nas = "NSGA_II_nas"
    ALNS_NSGA_II = "ALNS_NSGA_II"
    ALNS_NSGA_II_n = "ALNS_NSGA_II_n"
    ALNS_NSGA_II_as = "ALNS_NSGA_II_as"
    ALNS_NSGA_II_nas = "ALNS_NSGA_II_nas"
    MOEAD = "MOEAD"


# 求解器入口，初始化初始解，根据输入参数设定求解器，将solution和求解器设定传入具体调用的求解器
def solve(
    solver_name: Solver_Name,
    instance: InstanceClass,
) -> Solution | Population:

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
    elif solver_name in group_exact_list:
        algorithm_config = instance.prob_config.algorithm_config
        epsilon_num = algorithm_config.num_epsilon_points
        # 初始化解集合
        population = Population(
            size=epsilon_num,
            solver_name=solver_name.value,
            evaluator=Evaluator(instance),
        )
        # 获取求解器
        group_exact_solver = group_exact_list[solver_name]
        # 调用求解器，传入初始解、问题实例和算法参数
        population = group_exact_solver(population, instance)

        return population
    elif solver_name in group_heuri_list:
        algorithm_config = instance.prob_config.algorithm_config
        pop_size = algorithm_config.pop_size
        # 初始化解集合
        population = Population(
            size=pop_size,
            solver_name=solver_name.value,
            evaluator=Evaluator(instance),
        )
        # 获取求解器
        group_heuri_solver = group_heuri_list[solver_name]
        # 调用求解器,传入初始解、问题实例和算法参数
        population = group_heuri_solver(population, instance)
        return population
    else:
        raise ValueError(f"未知的求解器类型: {solver_name}")
