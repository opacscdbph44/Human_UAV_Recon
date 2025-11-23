"""测试新的启发式算法"""

import random
from datetime import datetime

import numpy as np

from src.config import (
    AlgorithmConfig,
    Config,
    InstanceParam,
)
from src.element import Solution
from src.solver.consr_heuristic import (
    drone_first_greedy_heuristic,
    efficient_greedy_heuristic,
    ground_first_greedy_heuristic,
    randomized_greedy_heuristic,
)
from src.utils import load_instance


def test_new_heuristics():
    """测试五个启发式算法"""

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    demand_num = 5

    # 创建配置
    prob_config = Config(
        instance_param=InstanceParam(
            name="a280.tsp",
            demand_num=demand_num,
        ),
        algorithm_config=AlgorithmConfig(
            max_iter=200,
        ),
        random_seed=42,
    )
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"加载配置成功！ {prob_config}"
    )

    # 加载实例
    instance = load_instance(prob_config)

    print(f"\n{'='*60}")
    print(f"测试实例: {instance.name}")
    print(f"需求点数量: {instance.demand_num}")
    print(f"地面车辆数量: {instance.ground_veh_num}")
    print(f"无人机数量: {instance.drone_num}")
    print(f"{'='*60}\n")

    # 测试1: 随机化贪心启发式
    print("测试1: 随机化贪心启发式")
    print("-" * 60)
    solution1 = Solution.new_solution(instance, solver_name="randomized_greedy")
    solution1 = randomized_greedy_heuristic(solution1, instance, success_rate=1.0)
    # 计算覆盖的任务数
    covered_tasks1 = set()
    for covered_set in solution1.coverage.values():
        covered_tasks1.update(covered_set)
    print(f"解状态: {solution1.status}")
    print(f"目标函数值: {solution1.objectives}")
    print(f"包含的节点数: {len(solution1.get_included_nodes())}")
    print(f"覆盖的任务数: {len(covered_tasks1)}")
    for veh_id in instance.ground_veh_ids + instance.drone_ids:
        route = solution1.routes[veh_id]
        veh_type = "地面车辆" if veh_id in instance.ground_veh_ids else "无人机"
        print(f"  {veh_type} {veh_id}: 路径长度={len(route)-2}, 路径={route}")

    # 测试2: 高效贪心启发式
    print(f"\n{'='*60}")
    print("测试2: 高效贪心启发式")
    print("-" * 60)
    solution2 = Solution.new_solution(instance, solver_name="efficient_greedy")
    solution2 = efficient_greedy_heuristic(solution2, instance, success_rate=1.0)
    # 计算覆盖的任务数
    covered_tasks2 = set()
    for covered_set in solution2.coverage.values():
        covered_tasks2.update(covered_set)
    print(f"解状态: {solution2.status}")
    print(f"目标函数值: {solution2.objectives}")
    print(f"包含的节点数: {len(solution2.get_included_nodes())}")
    print(f"覆盖的任务数: {len(covered_tasks2)}")
    for veh_id in instance.ground_veh_ids + instance.drone_ids:
        route = solution2.routes[veh_id]
        veh_type = "地面车辆" if veh_id in instance.ground_veh_ids else "无人机"
        print(f"  {veh_type} {veh_id}: 路径长度={len(route)-2}, 路径={route}")

    # 测试3: 地面编队优先
    print(f"\n{'='*60}")
    print("测试3: 地面编队优先的随机贪婪启发式")
    print("-" * 60)
    solution3 = Solution.new_solution(instance, solver_name="ground_first_greedy")
    solution3 = ground_first_greedy_heuristic(solution3, instance, success_rate=1.0)
    # 计算覆盖的任务数
    covered_tasks3 = set()
    for covered_set in solution3.coverage.values():
        covered_tasks3.update(covered_set)
    print(f"解状态: {solution3.status}")
    print(f"目标函数值: {solution3.objectives}")
    print(f"包含的节点数: {len(solution3.get_included_nodes())}")
    print(f"覆盖的任务数: {len(covered_tasks3)}")
    for veh_id in instance.ground_veh_ids + instance.drone_ids:
        route = solution3.routes[veh_id]
        veh_type = "地面车辆" if veh_id in instance.ground_veh_ids else "无人机"
        print(f"  {veh_type} {veh_id}: 路径长度={len(route)-2}, 路径={route}")

    # 测试4: 无人机编队优先
    print(f"\n{'='*60}")
    print("测试4: 低空编队优先的随机贪婪启发式")
    print("-" * 60)
    solution4 = Solution.new_solution(instance, solver_name="drone_first_greedy")
    solution4 = drone_first_greedy_heuristic(solution4, instance, success_rate=1.0)
    # 计算覆盖的任务数
    covered_tasks4 = set()
    for covered_set in solution4.coverage.values():
        covered_tasks4.update(covered_set)
    print(f"解状态: {solution4.status}")
    print(f"目标函数值: {solution4.objectives}")
    print(f"包含的节点数: {len(solution4.get_included_nodes())}")
    print(f"覆盖的任务数: {len(covered_tasks4)}")
    for veh_id in instance.ground_veh_ids + instance.drone_ids:
        route = solution4.routes[veh_id]
        veh_type = "地面车辆" if veh_id in instance.ground_veh_ids else "无人机"
        print(f"  {veh_type} {veh_id}: 路径长度={len(route)-2}, 路径={route}")

    # 比较结果 - 表格格式
    print(f"\n{'='*120}")
    print("结果比较")
    print("=" * 120)

    # 提取目标函数值
    obj1 = solution1.objectives[0] if solution1.objectives else float("inf")
    obj2 = solution2.objectives[0] if solution2.objectives else float("inf")
    obj3 = solution3.objectives[0] if solution3.objectives else float("inf")
    obj4 = solution4.objectives[0] if solution4.objectives else float("inf")

    # 准备所有解的详细信息
    solutions = [
        ("随机化贪心", solution1, obj1, covered_tasks1),
        ("高效贪心", solution2, obj2, covered_tasks2),
        ("地面编队优先", solution3, obj3, covered_tasks3),
        ("无人机编队优先", solution4, obj4, covered_tasks4),
    ]

    for algo_name, solution, obj, covered_tasks in solutions:
        print(f"\n算法: {algo_name}")
        print("-" * 120)
        print(f"  目标函数值: {obj:.2f}")
        print(f"  包含节点数: {len(solution.get_included_nodes())}")
        print(f"  覆盖任务数: {len(covered_tasks)}")
        print(f"  解状态: {solution.status}")
        print("  路径详情:")

        for veh_id in instance.ground_veh_ids + instance.drone_ids:
            route = solution.routes[veh_id]
            veh_type = "地面车辆" if veh_id in instance.ground_veh_ids else "无人机"
            if len(route) > 2:  # 只显示有访问节点的路径
                # 获取该车辆的覆盖信息
                veh_coverage = []
                for (v_id, visit_node), covered_set in solution.coverage.items():
                    if v_id == veh_id:
                        veh_coverage.append(
                            f"访问{visit_node}->覆盖{sorted(list(covered_set))}"
                        )

                print(f"    {veh_type} {veh_id}: 路径={route}")
                if veh_coverage:
                    for coverage_info in veh_coverage:
                        print(f"      {coverage_info}")

    # 汇总统计表格
    print(f"\n{'='*120}")
    print("汇总统计")
    print("=" * 120)

    # 表格表头 - 使用中文字符宽度计算（中文字符占2个宽度）
    header = f"{'算法名称':<18}{'目标函数值':>12}{'包含节点':>10}{'覆盖任务':>10}{'解状态':<20}"
    print(header)
    print("-" * 120)

    # 表格内容
    for algo_name, solution, obj, covered_tasks in solutions:
        # 计算中文字符串的显示宽度
        status_str = solution.status[:18]  # 截断过长的状态
        row = f"{algo_name:<18}{obj:>12.2f}{len(solution.get_included_nodes()):>10}{len(covered_tasks):>10}{status_str:<20}"
        print(row)

    print("=" * 120)

    # 找出最优解
    best_obj = min(obj1, obj2, obj3, obj4, obj5)
    best_algos = []
    if obj1 == best_obj:
        best_algos.append("随机化贪心")
    if obj2 == best_obj:
        best_algos.append("高效贪心")
    if obj3 == best_obj:
        best_algos.append("地面编队优先")
    if obj4 == best_obj:
        best_algos.append("无人机编队优先")
    if obj5 == best_obj:
        best_algos.append("距离引导")

    print(f"\n最优目标函数值: {best_obj:.2f}")
    print(f"最优算法: {', '.join(best_algos)}")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    test_new_heuristics()
