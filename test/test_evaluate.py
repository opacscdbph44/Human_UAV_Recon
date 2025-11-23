"""
测试评估器功能
测试流程:
1. 根据算例参数和求解器名称查询已保存的解
2. 如果查不到解，直接报错停止
3. 如果查到解，加载算例并使用Evaluator重新计算目标函数值
4. 验证Evaluator计算的目标值与存储的目标值是否一致
5. 如果一致，检查解的可行性
6. 可行性也通过，则测试通过

使用方法:
    运行所有测试:
        python test/test_evaluate.py

    或在Python中运行单个测试:
        from test.test_evaluate import test_solution_evaluation
        test_solution_evaluation("a280.tsp", 5, "gurobi_single_obj_solver")
"""

from datetime import datetime

from src.config import AlgorithmConfig, Config, InstanceParam
from src.evaluator import Evaluator
from src.utils import load_instance
from src.utils.file_funcs import (
    load_solution,
    query_instance_folder,
    query_result_folder,
)


def test_solution_evaluation(
    instance_name: str,
    demand_num: int,
    solver_name: str,
    base_num: int = 1,
    ground_veh_num: int = 2,
    drone_num: int = 2,
    random_seed: int = 42,
) -> bool:
    """测试已保存解的评估器功能

    Args:
        instance_name: 算例名称，如 "a280.tsp"
        demand_num: 需求节点数量
        solver_name: 求解器名称
        base_num: 基地数量
        ground_veh_num: 地面车辆数量
        drone_num: 无人机数量
        random_seed: 随机种子

    Returns:
        bool: 测试是否通过
    """
    print("\n" + "=" * 80)
    print(f"测试解的评估: {instance_name} (D={demand_num}) - {solver_name}")
    print("=" * 80)

    # =============== 1. 创建配置 ===============
    config = Config(
        instance_param=InstanceParam(
            name=instance_name,
            demand_num=demand_num,
            base_num=base_num,
            ground_veh_num=ground_veh_num,
            drone_num=drone_num,
        ),
        algorithm_config=AlgorithmConfig(
            time_limit=60,
        ),
        random_seed=random_seed,
    )

    # =============== 2. 查询并加载解 ===============
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: 查询已保存的解...")

    result_folder = query_result_folder(config, solver_name)

    try:
        solution = load_solution(result_folder)
        print("✓ 成功加载解")
        print(f"  求解器: {solution.Solver_name}")
        print(f"  状态: {solution.status}")
        print(f"  求解时间: {solution.solve_time:.2f}秒")
        print(f"  存储的目标值: {solution.objectives}")
    except FileNotFoundError as e:
        print(f"× 未找到解文件: {e}")
        print(f"  请先使用求解器 '{solver_name}' 对算例进行求解并保存结果")
        return False
    except Exception as e:
        print(f"× 加载解时发生错误: {e}")
        return False

    # =============== 3. 加载算例 ===============
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: 加载算例...")

    try:
        instance_folder = query_instance_folder(config)
        instance = load_instance(config, str(instance_folder))
        print(f"✓ 成功加载算例: {instance.name}")
    except Exception as e:
        print(f"× 加载算例失败: {e}")
        return False

    # =============== 4. 使用Evaluator重新计算目标函数值 ===============
    print(
        f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: 使用Evaluator计算目标值..."
    )

    evaluator = Evaluator(instance)
    evaluator_objectives = evaluator.calculate_objectives(solution)

    print(f"  Evaluator目标值: {evaluator_objectives}")

    # =============== 5. 比较目标函数值 ===============
    print(f"\n{'-' * 80}")
    print("目标函数值一致性检查:")
    print(f"{'-' * 80}")

    tolerance = 1e-4  # 容差
    all_match = True
    stored_objectives = solution.objectives

    if len(stored_objectives) != len(evaluator_objectives):
        print("× 目标数量不一致!")
        print(f"  存储的目标数: {len(stored_objectives)}")
        print(f"  Evaluator目标数: {len(evaluator_objectives)}")
        return False

    for i, (stored_val, eval_val) in enumerate(
        zip(stored_objectives, evaluator_objectives)
    ):
        diff = abs(stored_val - eval_val)
        relative_diff = diff / (abs(stored_val) + 1e-10) * 100

        match = diff < tolerance
        match_symbol = "✓" if match else "×"

        print(f"  目标 {i+1}:")
        print(f"    存储值:       {stored_val:.6f}")
        print(f"    Evaluator值:  {eval_val:.6f}")
        print(f"    绝对差值:     {diff:.6e}")
        print(f"    相对差值:     {relative_diff:.4f}%")
        print(f"    匹配状态:     {match_symbol}")

        if not match:
            all_match = False

    if not all_match:
        print(f"\n{'-' * 80}")
        print("××× 目标函数值不一致！")
        return False

    print("\n✓ 目标函数值一致性检查通过")

    # =============== 6. 检查解的可行性 ===============
    print(f"\n{'-' * 80}")
    print("解可行性检查:")
    print(f"{'-' * 80}")

    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: 开始可行性检查...")
    is_feasible = evaluator.sol_feasible(solution)

    if not is_feasible:
        print("××× 解不可行！")
        return False

    print("✓ 解可行性检查通过")

    # =============== 7. 打印解的详细信息 ===============
    print(f"\n{'-' * 80}")
    print("解的详细信息:")
    print(f"{'-' * 80}")

    print("\n车辆路径:")
    for veh_id, route in enumerate(solution.routes):
        if len(route) > 2:  # 只显示非空路径
            print(f"  车辆 {veh_id}: {route}")

    print("\n覆盖情况:")
    for (veh_id, visit_node), covered_nodes in solution.coverage.items():
        if len(covered_nodes) > 0:
            print(f"  车辆 {veh_id} 在节点 {visit_node} 覆盖: {sorted(covered_nodes)}")

    # 统计总覆盖节点
    all_covered = set()
    for covered_nodes in solution.coverage.values():
        all_covered.update(covered_nodes)

    print("\n覆盖统计:")
    print(f"  总需求节点数: {instance.demand_num}")
    print(f"  已覆盖节点数: {len(all_covered)}")
    print(f"  覆盖率: {len(all_covered) / instance.demand_num * 100:.1f}%")

    # =============== 8. 测试通过 ===============
    print(f"\n{'=' * 80}")
    print("✓✓✓ 所有测试通过！")
    print(f"{'=' * 80}")

    return True


def test_all():
    """运行所有预定义的测试案例"""
    print("\n" + "=" * 80)
    print("开始运行评估器测试")
    print("=" * 80)

    # 定义测试案例
    test_cases = [
        {
            "instance_name": "a280.tsp",
            "demand_num": 5,
            "solver_name": "gurobi_single_obj_solver",
            "description": "小规模测试 (5个需求节点)",
        },
        {
            "instance_name": "a280.tsp",
            "demand_num": 10,
            "solver_name": "gurobi_single_obj_solver",
            "description": "中等规模测试 (10个需求节点)",
        },
    ]

    # 运行所有测试
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"测试案例 {i}/{len(test_cases)}: {test_case['description']}")
        print(f"{'=' * 80}")

        passed = test_solution_evaluation(
            instance_name=test_case["instance_name"],
            demand_num=test_case["demand_num"],
            solver_name=test_case["solver_name"],
        )

        results.append((test_case["description"], passed))

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    for desc, passed in results:
        status = "✓ 通过" if passed else "× 失败"
        print(f"{desc}: {status}")

    all_passed = all(passed for _, passed in results)

    print(f"\n{'=' * 80}")
    if all_passed:
        print("✓✓✓ 所有测试通过！")
    else:
        print("××× 部分测试失败！")
    print(f"{'=' * 80}")

    return all_passed


if __name__ == "__main__":
    all_passed_list = []
    # 运行单个测试示例
    for demand_num in [
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
    ]:
        all_passed_list.append(
            test_solution_evaluation(
                instance_name="a280.tsp",
                demand_num=demand_num,
                solver_name="gurobi_single_obj_solver",
            )
        )
    print("\n" + "=" * 80)
    if all(all_passed_list):
        print("✓✓✓ 所有单独测试通过！")
    else:
        print("××× 部分单独测试失败！")
    # test_solution_evaluation("a280.tsp", 5, "gurobi_single_obj_solver")

    # # 运行所有测试
    # test_all()
