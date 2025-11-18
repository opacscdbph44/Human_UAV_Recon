import random
import time

import numpy as np

from src.element import InstanceClass, Solution
from src.evaluator import Evaluator, Population


def randomized_greedy_heuristic(
    solution: Solution,
    instance: InstanceClass,
    success_rate: float = 1.0,
) -> Solution:
    """随机化贪心启发式算法构建初始解

    Args:
        solution (Solution): 初始解
        instance (InstanceClass): 问题实例

    Returns:
        Solution: 构建的解
    """

    solution.init_all_routes()

    # 全取时不需要筛选，直接返回
    if success_rate == 1.0:
        num_tasks = instance.prob_config.instance_param.demand_num
        task_to_assign = instance.demand_ids[:]
    # 不取时，不需要筛选，返回空
    elif success_rate == 0.0:
        task_to_assign = []
    # 部分取时，随机选择任务（向上取整）
    elif success_rate > 0.0:
        num_tasks = int(
            np.ceil(instance.prob_config.instance_param.demand_num * success_rate)
        )
        task_to_assign = np.random.choice(
            instance.demand_ids,
            size=num_tasks,
            replace=False,
        ).tolist()
    else:
        raise ValueError(
            f"任务成功率 success_rate 应在 [0.0, 1.0] 范围内，实际: {success_rate}"
        )

    # 构建评价器
    evaluator = Evaluator(instance)

    # 提取距离矩阵、覆盖矩阵等
    comm_coverage_matrix = instance.comm_coverage_matrix
    priority_list = instance.priority

    # 计算每个待分配任务的可覆盖信息（数量、收益、具体任务列表）
    coverage_info = {}
    # 将 task_to_assign 转换为集合，方便快速查找
    task_set = set(task_to_assign)

    # 提取vehicle_ids
    vehicle_ids = instance.ground_veh_ids + instance.drone_ids

    # 提取accessible
    accessible_list = instance.accessible
    for veh_id in vehicle_ids:
        if veh_id in instance.ground_veh_ids:
            check_accessible = True
        else:
            check_accessible = False
        for task in task_to_assign:
            # 跳过地面车辆不可达点
            if check_accessible and accessible_list[task] == 0:
                continue

            # 找出该任务可以覆盖的其他任务（布尔索引）
            covered_mask = comm_coverage_matrix[veh_id][task]
            # 获取覆盖任务的索引列表
            all_covered_tasks = np.where(covered_mask)[0].tolist()

            # 只保留在 task_to_assign 中的任务
            covered_tasks = [t for t in all_covered_tasks if t in task_set]

            # 计算覆盖收益（只统计 task_to_assign 中任务的优先级）
            coverage_benefit = sum(priority_list[t] for t in covered_tasks)

            # 判断这个访问是否有价值，如果完全没价值直接跳过
            if coverage_benefit == 0:
                continue
            else:
                # 存储所有信息
                coverage_info[(veh_id, task)] = {
                    "count": len(covered_tasks),  # 覆盖数量
                    "benefit": coverage_benefit,  # 覆盖收益
                    "covered_tasks": covered_tasks,  # 覆盖的任务列表
                }

        # 计算Steiner节点信息
        steiner_ids = instance.steiner_ids
        for steiner in steiner_ids:
            # 找出该Steiner节点可以覆盖的任务（布尔索引）
            covered_mask = comm_coverage_matrix[veh_id][steiner]
            # 获取覆盖任务的索引列表
            all_covered_tasks = np.where(covered_mask)[0].tolist()
            # 只保留在 task_to_assign 中的任务
            covered_tasks = [t for t in all_covered_tasks if t in task_set]
            # 计算覆盖收益（只统计 task_to_assign 中任务的优先级）
            coverage_benefit = sum(priority_list[t] for t in covered_tasks)

            # 判断这个访问是否有价值，如果完全没价值直接跳过
            if coverage_benefit == 0:
                continue
            else:
                # 存储所有信息
                coverage_info[(veh_id, steiner)] = {
                    "count": len(covered_tasks),  # 覆盖数量
                    "benefit": coverage_benefit,  # 覆盖收益
                    "covered_tasks": covered_tasks,  # 覆盖的任务列表
                }

    while task_set:
        # 随机选择一个覆盖方案
        current_plan = random.choice(list(coverage_info.keys()))

        # 提取选择的信息
        veh_id, node_id = current_plan
        completed_tasks = coverage_info[current_plan]["covered_tasks"]

        # 更新解中的路径信息
        target_route = solution.routes[veh_id]

        # 组成临时插入后路径
        candidate_route = target_route[:-1] + [node_id] + target_route[-1:]

        # 提取当前解中的覆盖信息(只提取当前车辆相关的)
        current_coverage = {
            key: solution.get_coverage(key[0], key[1])
            for key in solution.coverage.keys()
            if key[0] == veh_id
        }
        current_coverage[(veh_id, node_id)] = set(completed_tasks)

        # 计算路径信息，判断是否可行
        route_feasible, feasible_reason = evaluator.is_route_feasible(
            veh_id,
            candidate_route,
            current_coverage,
        )

        if route_feasible:
            # 更新路径
            solution.routes[veh_id] = candidate_route
            # 更新覆盖信息
            solution.set_coverage(veh_id, node_id, set(completed_tasks))

            # 从待分配任务中移除已完成任务
            for t in completed_tasks:
                if t in task_set:
                    task_set.remove(t)
                else:
                    raise ValueError(f"任务 {t} 不在待分配任务集中,无法移除")

            # 更新全体coverage_info
            keys_to_remove = []
            for key in coverage_info.keys():
                veh_id_key, node_id_key = key
                covered_tasks_key = coverage_info[key]["covered_tasks"]

                # 情况1: 如果访问节点已被覆盖(不在task_set中),删除该键
                if (node_id_key in instance.demand_ids) and (
                    node_id_key not in task_set
                ):
                    keys_to_remove.append(key)
                # 情况2: 如果访问的是Steiner节点,删除所有车辆对该Steiner节点的访问记录
                elif (node_id in instance.steiner_ids) and (node_id_key == node_id):
                    keys_to_remove.append(key)
                # 情况3: 访问节点未被覆盖,但其覆盖的任务发生变化
                else:
                    # 更新covered_tasks,只保留仍在task_set中的任务
                    updated_covered_tasks = [
                        t for t in covered_tasks_key if t in task_set
                    ]

                    # 如果没有可覆盖的任务了,标记删除
                    if not updated_covered_tasks:
                        keys_to_remove.append(key)
                    else:
                        # 更新覆盖信息
                        coverage_info[key]["covered_tasks"] = updated_covered_tasks
                        coverage_info[key]["count"] = len(updated_covered_tasks)
                        coverage_info[key]["benefit"] = sum(
                            priority_list[t] for t in updated_covered_tasks
                        )

            # 删除标记的键
            for key in keys_to_remove:
                del coverage_info[key]

    # 评估解，计算目标函数
    evaluator.sol_evaluate(solution)

    # 返回构建的解
    return solution


def efficient_greedy_heuristic(
    solution: Solution,
    instance: InstanceClass,
    success_rate: float = 1.0,
) -> Solution:
    """随机化贪心启发式算法构建初始解

    Args:
        solution (Solution): 初始解
        instance (InstanceClass): 问题实例

    Returns:
        Solution: 构建的解
    """

    solution.init_all_routes()

    # 全取时不需要筛选，直接返回
    if success_rate == 1.0:
        num_tasks = instance.prob_config.instance_param.demand_num
        task_to_assign = instance.demand_ids[:]
    # 不取时，不需要筛选，返回空
    elif success_rate == 0.0:
        task_to_assign = []
    # 部分取时，随机选择任务（向上取整）
    elif success_rate > 0.0:
        num_tasks = int(
            np.ceil(instance.prob_config.instance_param.demand_num * success_rate)
        )
        task_to_assign = np.random.choice(
            instance.demand_ids,
            size=num_tasks,
            replace=False,
        ).tolist()
    else:
        raise ValueError(
            f"任务成功率 success_rate 应在 [0.0, 1.0] 范围内，实际: {success_rate}"
        )

    # 构建评价器
    evaluator = Evaluator(instance)

    # 提取距离矩阵、覆盖矩阵等
    comm_coverage_matrix = instance.comm_coverage_matrix
    priority_list = instance.priority

    # 计算每个待分配任务的可覆盖信息（数量、收益、具体任务列表）
    coverage_info = {}
    # 将 task_to_assign 转换为集合，方便快速查找
    task_set = set(task_to_assign)

    # 提取vehicle_ids
    vehicle_ids = instance.ground_veh_ids + instance.drone_ids

    # 提取accessible
    accessible_list = instance.accessible
    for veh_id in vehicle_ids:
        if veh_id in instance.ground_veh_ids:
            check_accessible = True
        else:
            check_accessible = False
        for task in task_to_assign:
            # 跳过地面车辆不可达点
            if check_accessible and accessible_list[task] == 0:
                continue

            # 找出该任务可以覆盖的其他任务（布尔索引）
            covered_mask = comm_coverage_matrix[veh_id][task]
            # 获取覆盖任务的索引列表
            all_covered_tasks = np.where(covered_mask)[0].tolist()

            # 只保留在 task_to_assign 中的任务
            covered_tasks = [t for t in all_covered_tasks if t in task_set]

            # 计算覆盖收益（只统计 task_to_assign 中任务的优先级）
            coverage_benefit = sum(priority_list[t] for t in covered_tasks)

            # 判断这个访问是否有价值，如果完全没价值直接跳过
            if coverage_benefit == 0:
                continue
            else:
                # 存储所有信息
                coverage_info[(veh_id, task)] = {
                    "count": len(covered_tasks),  # 覆盖数量
                    "benefit": coverage_benefit,  # 覆盖收益
                    "covered_tasks": covered_tasks,  # 覆盖的任务列表
                }

        # 计算Steiner节点信息
        steiner_ids = instance.steiner_ids
        for steiner in steiner_ids:
            # 找出该Steiner节点可以覆盖的任务（布尔索引）
            covered_mask = comm_coverage_matrix[veh_id][steiner]
            # 获取覆盖任务的索引列表
            all_covered_tasks = np.where(covered_mask)[0].tolist()
            # 只保留在 task_to_assign 中的任务
            covered_tasks = [t for t in all_covered_tasks if t in task_set]
            # 计算覆盖收益（只统计 task_to_assign 中任务的优先级）
            coverage_benefit = sum(priority_list[t] for t in covered_tasks)

            # 判断这个访问是否有价值，如果完全没价值直接跳过
            if coverage_benefit == 0:
                continue
            else:
                # 存储所有信息
                coverage_info[(veh_id, steiner)] = {
                    "count": len(covered_tasks),  # 覆盖数量
                    "benefit": coverage_benefit,  # 覆盖收益
                    "covered_tasks": covered_tasks,  # 覆盖的任务列表
                }

    while task_set:
        # 随机选择一个覆盖方案
        current_plan = max(
            coverage_info.keys(),
            key=lambda k: coverage_info[k]["benefit"],
        )

        # 提取选择的信息
        veh_id, node_id = current_plan
        completed_tasks = coverage_info[current_plan]["covered_tasks"]

        # 更新解中的路径信息
        target_route = solution.routes[veh_id]

        # 组成临时插入后路径
        candidate_route = target_route[:-1] + [node_id] + target_route[-1:]

        # 提取当前解中的覆盖信息(只提取当前车辆相关的)
        current_coverage = {
            key: solution.get_coverage(key[0], key[1])
            for key in solution.coverage.keys()
            if key[0] == veh_id
        }
        current_coverage[(veh_id, node_id)] = set(completed_tasks)

        # 计算路径信息，判断是否可行
        route_feasible, feasible_reason = evaluator.is_route_feasible(
            veh_id,
            candidate_route,
            current_coverage,
        )

        if route_feasible:
            # 更新路径
            solution.routes[veh_id] = candidate_route
            # 更新覆盖信息
            solution.set_coverage(veh_id, node_id, set(completed_tasks))

            # 从待分配任务中移除已完成任务
            for t in completed_tasks:
                if t in task_set:
                    task_set.remove(t)
                else:
                    raise ValueError(f"任务 {t} 不在待分配任务集中,无法移除")

            # 更新全体coverage_info
            keys_to_remove = []
            for key in coverage_info.keys():
                veh_id_key, node_id_key = key
                covered_tasks_key = coverage_info[key]["covered_tasks"]

                # 情况1: 如果访问节点已被覆盖(不在task_set中),删除该键
                if (node_id_key in instance.demand_ids) and (
                    node_id_key not in task_set
                ):
                    keys_to_remove.append(key)
                # 情况2: 如果访问的是Steiner节点,删除所有车辆对该Steiner节点的访问记录
                elif (node_id in instance.steiner_ids) and (node_id_key == node_id):
                    keys_to_remove.append(key)
                # 情况3: 访问节点未被覆盖,但其覆盖的任务发生变化
                else:
                    # 更新covered_tasks,只保留仍在task_set中的任务
                    updated_covered_tasks = [
                        t for t in covered_tasks_key if t in task_set
                    ]

                    # 如果没有可覆盖的任务了,标记删除
                    if not updated_covered_tasks:
                        keys_to_remove.append(key)
                    else:
                        # 更新覆盖信息
                        coverage_info[key]["covered_tasks"] = updated_covered_tasks
                        coverage_info[key]["count"] = len(updated_covered_tasks)
                        coverage_info[key]["benefit"] = sum(
                            priority_list[t] for t in updated_covered_tasks
                        )

            # 删除标记的键
            for key in keys_to_remove:
                del coverage_info[key]

    # 评估解，计算目标函数
    _, sol_feasible = evaluator.sol_evaluate(solution)

    if sol_feasible:
        solution.status = "已评估, 当前解可行"
    else:
        solution.status = "已评估, 当前解不可行"

    # 返回构建的解
    return solution


def multi_randomized_greedy_heuristic(
    population: "Population",
    instance: "InstanceClass",
) -> Population:
    """初始种群生成函数"""
    # 提取算法参数
    algorithm_config = instance.prob_config.algorithm_config
    pop_size = algorithm_config.pop_size

    # 检查pop规模
    if population.size != pop_size:
        raise ValueError(f"种群大小不匹配，期望: {pop_size}, 实际: {population.size}")

    # 一次性生成 pop_size 个非重复的成功率
    # 确保包含 0 和 1，其余在 (0, 1) 之间均匀分布
    if pop_size < 2:
        raise ValueError(f"种群大小至少需要2，实际: {pop_size}")

    # 开始计时
    start_time = time.perf_counter()

    # 生成成功率列表
    success_rates = [0.0, 1.0]  # 确保包含边界值

    # 在 (0, 1) 之间生成 pop_size - 2 个随机值
    if pop_size > 2:
        middle_rates = np.random.uniform(0, 1, pop_size - 2)
        success_rates.extend(middle_rates.tolist())

    # 打乱顺序，避免总是先处理 0 和 1
    np.random.shuffle(success_rates)

    # 初始化个体idx
    pop_idx = 0
    while pop_idx < pop_size:
        # 生成初始解
        solution = Solution.new_solution(
            instance,
            solver_name="randomized_greedy_heuristic",
        )
        # 使用预生成的成功率
        success_rate = success_rates[pop_idx]
        # # 调用随机化贪心启发式算法
        solution = randomized_greedy_heuristic(
            solution,
            instance,
            success_rate=success_rate,
        )

        # 将解添加到种群中
        population.add_solution(solution)
        pop_idx += 1
    # 结束计时
    end_time = time.perf_counter()
    population.solve_time = end_time - start_time
    return population
