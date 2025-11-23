import random
from typing import List

import numpy as np

from src.element import InstanceClass, Solution
from src.evaluator import Evaluator


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

    while task_set and coverage_info:
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
        else:
            # 如果不可行，从候选列表中移除该方案
            del coverage_info[current_plan]

    # 评估解，计算目标函数
    _, sol_feasible = evaluator.sol_evaluate(solution)

    if sol_feasible:
        solution.status = "已评估, 当前解可行"
    else:
        solution.status = "已评估, 当前解不可行"

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

    while task_set and coverage_info:
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
        else:
            # 如果不可行，从候选列表中移除该方案
            del coverage_info[current_plan]

    # 评估解，计算目标函数
    _, sol_feasible = evaluator.sol_evaluate(solution)

    if sol_feasible:
        solution.status = "已评估, 当前解可行"
    else:
        solution.status = "已评估, 当前解不可行"

    # 返回构建的解
    return solution


def ground_first_greedy_heuristic(
    solution: Solution,
    instance: InstanceClass,
    success_rate: float = 1.0,
) -> Solution:
    """地面编队优先的随机贪婪启发式算法构建初始解

    优先使用地面编队，当地面编队队伍满了之后，考虑无人机

    Args:
        solution (Solution): 初始解
        instance (InstanceClass): 问题实例
        success_rate (float): 任务成功率

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

    # 提取accessible
    accessible_list = instance.accessible

    # 优先考虑地面车辆，然后是无人机
    vehicle_ids = instance.ground_veh_ids + instance.drone_ids

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

    while task_set and coverage_info:
        # 筛选出地面车辆的覆盖方案
        ground_plans = [
            key for key in coverage_info.keys() if key[0] in instance.ground_veh_ids
        ]

        # 如果有地面车辆的方案，优先从中选择
        if ground_plans:
            current_plan = random.choice(ground_plans)
        else:
            # 否则从无人机方案中选择
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
        else:
            # 如果不可行，从候选列表中移除该方案
            del coverage_info[current_plan]

    # 评估解，计算目标函数
    _, sol_feasible = evaluator.sol_evaluate(solution)

    if sol_feasible:
        solution.status = "已评估, 当前解可行"
    else:
        solution.status = "已评估, 当前解不可行"

    # 返回构建的解
    return solution


def drone_first_greedy_heuristic(
    solution: Solution,
    instance: InstanceClass,
    success_rate: float = 1.0,
) -> Solution:
    """低空编队优先的随机贪婪启发式算法构建初始解

    优先使用无人机，当无人机编队满了之后，考虑地面编队

    Args:
        solution (Solution): 初始解
        instance (InstanceClass): 问题实例
        success_rate (float): 任务成功率

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

    # 提取accessible
    accessible_list = instance.accessible

    # 优先考虑无人机，然后是地面车辆
    vehicle_ids = instance.drone_ids + instance.ground_veh_ids

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

    while task_set and coverage_info:
        # 筛选出无人机的覆盖方案
        drone_plans = [
            key for key in coverage_info.keys() if key[0] in instance.drone_ids
        ]

        # 如果有无人机的方案，优先从中选择
        if drone_plans:
            current_plan = random.choice(drone_plans)
        else:
            # 否则从地面车辆方案中选择
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
        else:
            # 如果不可行，从候选列表中移除该方案
            del coverage_info[current_plan]

    # 评估解，计算目标函数
    _, sol_feasible = evaluator.sol_evaluate(solution)

    if sol_feasible:
        solution.status = "已评估, 当前解可行"
    else:
        solution.status = "已评估, 当前解不可行"

    # 返回构建的解
    return solution


def multi_start_initial_solution(
    instance: InstanceClass,
    num_starts: int = 10,
    success_rate: float = 1.0,
) -> List[Solution]:
    """多起点初始解生成

    Args:
        instance (InstanceClass): 问题实例
        num_starts (int): 起点数量
        success_rate (float): 任务成功率

    Returns:
        List[Solution]: 初始解列表
    """

    # 构造算法映射
    heuristic_map = {
        "randomized_greedy": randomized_greedy_heuristic,
        "efficient_greedy": efficient_greedy_heuristic,
        "ground_first_greedy": ground_first_greedy_heuristic,
        "drone_first_greedy": drone_first_greedy_heuristic,
    }

    # 获取算法列表
    heuristic_list = list(heuristic_map.values())
    num_heuristics = len(heuristic_list)

    init_starts = []

    for i in range(num_starts):
        # 创建空解
        initial_solution = Solution.new_solution(instance)

        # 循环使用不同的算法
        heuristic_func = heuristic_list[i % num_heuristics]

        # 使用选中的启发式算法构建初始解
        constructed_solution = heuristic_func(
            initial_solution,
            instance,
            success_rate,
        )

        # 将构建的解添加到列表中
        init_starts.append(constructed_solution)

    return init_starts


def multi_start_heuristic(
    solution: Solution,
    instance: InstanceClass,
    num_starts: int = 10,
    success_rate: float = 1.0,
) -> List[Solution]:
    """多起点初始解生成

    Args:
        instance (InstanceClass): 问题实例
        num_starts (int): 起点数量
        success_rate (float): 任务成功率

    Returns:
        Solution: 初始解列表
    """

    # 构造算法映射
    heuristic_map = {
        "randomized_greedy": randomized_greedy_heuristic,
        "efficient_greedy": efficient_greedy_heuristic,
        "ground_first_greedy": ground_first_greedy_heuristic,
        "drone_first_greedy": drone_first_greedy_heuristic,
    }

    # 获取算法列表
    heuristic_list = list(heuristic_map.values())
    num_heuristics = len(heuristic_list)

    init_starts = []

    for i in range(num_starts):
        # 创建空解
        initial_solution = Solution.new_solution(instance)

        # 循环使用不同的算法
        heuristic_func = heuristic_list[i % num_heuristics]

        # 使用选中的启发式算法构建初始解
        constructed_solution = heuristic_func(
            initial_solution,
            instance,
            success_rate,
        )

        # 将构建的解添加到列表中
        init_starts.append(constructed_solution)
    best_objective = 0.0

    for idx, current_solution in enumerate(init_starts):
        if current_solution.objectives[0] > best_objective:
            best_objective = current_solution.objectives[0]
            best_solution = current_solution

    return best_solution
