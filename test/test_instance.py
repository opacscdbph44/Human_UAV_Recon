"""
测试算例的生成、保存、读取和绘制功能
"""

from pathlib import Path

from src.config import Config
from src.element import InstanceClass
from src.utils.display import plot_instance
from src.utils.file_funcs import load_instance, query_instance_folder
from src.utils.ins_create import create_by_tsp


def test_instance_creation_basic():
    """测试基本算例生成"""
    print("\n" + "=" * 80)
    print("测试1: 基本算例生成（不分组）")
    print("=" * 80)

    # 创建配置
    config = Config.from_dict(
        {
            "instance_param": {
                "name": "berlin52.tsp",
                "base_num": 1,
                "base_select_mode": "assign",
                "base_select_param": [[0.5, 0.5]],
                "demand_num": 20,
                "demand_selection": "Random",
                "distrib_type": "Random",
                "isolate_ratio": 0.3,
                "ground_veh_num": 2,
                "drone_num": 2,
                "grouping_method": "none",  # 不分组
                "vehicle_config": {
                    "ground_veh_speed": 30.0,
                    "ground_veh_range": 200.0,
                    "ground_veh_comm_radius": 0.0,
                    "drone_speed": 50.0,
                    "drone_range": 150.0,
                    "drone_comm_radius": 80.0,
                },
                "min_visit_time": 10.0,
            },
            "algorithm_config": {
                "big_m": 1e6,
                "time_limit": 300,
                "max_iter": 100,
            },
        }
    )

    # 生成算例
    save_folder = query_instance_folder(config)
    instance = create_by_tsp(config, save_folder, save_file=True)

    # 验证分组信息
    print(f"\n算例名称: {instance.name}")
    print(f"基地数量: {instance.base_num}")
    print(f"需求点数量: {instance.demand_num}")
    print(f"总节点数: {instance.total_node_num}")
    print(f"\n分组信息 (group_id): {instance.group_id}")
    print(f"分组集合 (group_sets): {instance.group_sets}")

    # 绘制算例
    figure_folder = save_folder / "figures"
    plot_instance(instance, figure_folder_path=figure_folder)

    print(f"\n✓ 测试1通过：算例已生成并保存到 {save_folder}")
    return instance, save_folder


def test_instance_creation_kmeans():
    """测试K-means分组"""
    print("\n" + "=" * 80)
    print("测试2: K-means聚类分组")
    print("=" * 80)

    config = Config.from_dict(
        {
            "instance_param": {
                "name": "berlin52.tsp",
                "base_num": 1,
                "base_select_mode": "assign",
                "base_select_param": [[0.5, 0.5]],
                "demand_num": 30,
                "demand_selection": "Random",
                "distrib_type": "Random",
                "isolate_ratio": 0.3,
                "ground_veh_num": 2,
                "drone_num": 2,
                "grouping_method": "kmeans",
                "num_groups": 5,  # 分为5组
                "vehicle_config": {
                    "ground_veh_speed": 30.0,
                    "ground_veh_range": 200.0,
                    "ground_veh_comm_radius": 0.0,
                    "drone_speed": 50.0,
                    "drone_range": 150.0,
                    "drone_comm_radius": 80.0,
                },
                "min_visit_time": 10.0,
            },
            "algorithm_config": {
                "big_m": 1e6,
                "time_limit": 300,
                "max_iter": 100,
            },
        }
    )

    save_folder = query_instance_folder(config)
    instance = create_by_tsp(config, save_folder, save_file=True)

    print(f"\n算例名称: {instance.name}")
    print("分组方式: K-means")
    print(f"分组数量: {len(instance.group_sets)}")
    print("分组集合:")
    for group_id, nodes in instance.group_sets.items():
        print(f"  组 {group_id}: {sorted(nodes)} ({len(nodes)}个节点)")

    # 绘制算例
    figure_folder = save_folder / "figures"
    plot_instance(instance, figure_folder_path=figure_folder)

    print("\n✓ 测试2通过：K-means分组算例已生成")
    return instance, save_folder


def test_instance_creation_grid():
    """测试网格分组"""
    print("\n" + "=" * 80)
    print("测试3: 网格分组")
    print("=" * 80)

    config = Config.from_dict(
        {
            "instance_param": {
                "name": "berlin52.tsp",
                "base_num": 1,
                "base_select_mode": "assign",
                "base_select_param": [[0.5, 0.5]],
                "demand_num": 36,
                "demand_selection": "Top_k",
                "distrib_type": "Random",
                "isolate_ratio": 0.3,
                "ground_veh_num": 2,
                "drone_num": 2,
                "grouping_method": "grid",
                "grid_size": [3, 3],  # 3x3网格
                "vehicle_config": {
                    "ground_veh_speed": 30.0,
                    "ground_veh_range": 200.0,
                    "ground_veh_comm_radius": 0.0,
                    "drone_speed": 50.0,
                    "drone_range": 150.0,
                    "drone_comm_radius": 80.0,
                },
                "min_visit_time": 10.0,
            },
            "algorithm_config": {
                "big_m": 1e6,
                "time_limit": 300,
                "max_iter": 100,
            },
        }
    )

    save_folder = query_instance_folder(config)
    instance = create_by_tsp(config, save_folder, save_file=True)

    print(f"\n算例名称: {instance.name}")
    print("分组方式: 网格分组")
    print("网格尺寸: 3x3")
    print(f"实际分组数: {len(instance.group_sets)}")
    print("分组集合:")
    for group_id in sorted(instance.group_sets.keys()):
        nodes = instance.group_sets[group_id]
        print(f"  组 {group_id}: {len(nodes)}个节点")

    # 绘制算例
    figure_folder = save_folder / "figures"
    plot_instance(instance, figure_folder_path=figure_folder)

    print("\n✓ 测试3通过：网格分组算例已生成")
    return instance, save_folder


def test_instance_creation_distance():
    """测试距离阈值分组"""
    print("\n" + "=" * 80)
    print("测试4: 距离阈值分组")
    print("=" * 80)

    config = Config.from_dict(
        {
            "instance_param": {
                "name": "berlin52.tsp",
                "base_num": 1,
                "base_select_mode": "assign",
                "base_select_param": [[0.5, 0.5]],
                "demand_num": 25,
                "demand_selection": "Random",
                "distrib_type": "Random",
                "isolate_ratio": 0.3,
                "ground_veh_num": 2,
                "drone_num": 2,
                "grouping_method": "distance",
                "distance_threshold": 200.0,  # 距离阈值
                "vehicle_config": {
                    "ground_veh_speed": 30.0,
                    "ground_veh_range": 200.0,
                    "ground_veh_comm_radius": 0.0,
                    "drone_speed": 50.0,
                    "drone_range": 150.0,
                    "drone_comm_radius": 80.0,
                },
                "min_visit_time": 10.0,
            },
            "algorithm_config": {
                "big_m": 1e6,
                "time_limit": 300,
                "max_iter": 100,
            },
        }
    )

    save_folder = query_instance_folder(config)
    instance = create_by_tsp(config, save_folder, save_file=True)

    print(f"\n算例名称: {instance.name}")
    print("分组方式: 距离阈值")
    print("距离阈值: 200.0")
    print(f"实际分组数: {len(instance.group_sets)}")
    print("分组集合:")
    for group_id in sorted(instance.group_sets.keys()):
        nodes = instance.group_sets[group_id]
        print(f"  组 {group_id}: {sorted(nodes)[:5]}... ({len(nodes)}个节点)")

    # 绘制算例
    figure_folder = save_folder / "figures"
    plot_instance(instance, figure_folder_path=figure_folder)

    print("\n✓ 测试4通过：距离阈值分组算例已生成")
    return instance, save_folder


def test_instance_save_and_load(instance: InstanceClass, save_folder: Path):
    """测试算例的保存和读取"""
    print("\n" + "=" * 80)
    print("测试5: 算例保存和读取")
    print("=" * 80)

    # 使用原始配置重新加载算例
    prob_config = instance.prob_config
    print("使用配置重新加载算例...")

    # 使用load_instance加载（它会根据配置查找匹配的算例）
    loaded_instance = load_instance(prob_config)

    # 验证数据一致性
    print("\n原始算例:")
    print(f"  名称: {instance.name}")
    print(f"  需求点数: {instance.demand_num}")
    print(f"  分组数: {len(instance.group_sets)}")

    print("\n加载算例:")
    print(f"  名称: {loaded_instance.name}")
    print(f"  需求点数: {loaded_instance.demand_num}")
    print(f"  分组数: {len(loaded_instance.group_sets)}")

    # 验证关键数据
    assert instance.name == loaded_instance.name, "名称不匹配"
    assert instance.demand_num == loaded_instance.demand_num, "需求点数不匹配"
    assert instance.base_num == loaded_instance.base_num, "基地数不匹配"
    assert len(instance.coords) == len(loaded_instance.coords), "坐标数不匹配"
    assert instance.group_id == loaded_instance.group_id, "分组ID不匹配"
    assert instance.group_sets == loaded_instance.group_sets, "分组集合不匹配"

    print("\n✓ 测试5通过：算例保存和读取正确")

    # 绘制加载的算例
    figure_folder = save_folder / "figures_loaded"
    plot_instance(loaded_instance, figure_folder_path=figure_folder)
    print(f"✓ 加载算例的图形已保存到: {figure_folder}")

    return loaded_instance


def test_instance_properties():
    """测试算例属性和方法"""
    print("\n" + "=" * 80)
    print("测试6: 算例属性和方法")
    print("=" * 80)

    config = Config.from_dict(
        {
            "instance_param": {
                "name": "berlin52.tsp",
                "base_num": 1,
                "base_select_mode": "assign",
                "base_select_param": [[0.5, 0.5]],
                "demand_num": 15,
                "demand_selection": "Random",
                "distrib_type": "Linear",
                "distrib_center": [0.5, 0.5],
                "isolate_ratio": 0.4,
                "ground_veh_num": 2,
                "drone_num": 2,
                "grouping_method": "kmeans",
                "num_groups": 3,
                "vehicle_config": {
                    "ground_veh_speed": 30.0,
                    "ground_veh_range": 200.0,
                    "ground_veh_comm_radius": 0.0,
                    "drone_speed": 50.0,
                    "drone_range": 150.0,
                    "drone_comm_radius": 80.0,
                },
                "min_visit_time": 10.0,
            },
            "algorithm_config": {
                "big_m": 1e6,
                "time_limit": 300,
                "max_iter": 100,
            },
        }
    )

    save_folder = query_instance_folder(config)
    instance = create_by_tsp(config, save_folder, save_file=False)

    print("\n基本信息:")
    print(f"  {instance}")
    print(f"  总节点数: {instance.total_node_num}")
    print(f"  总车辆数: {instance.total_veh_num}")

    print("\n节点ID列表:")
    print(f"  基地ID: {instance.base_ids}")
    print(f"  需求点ID: {instance.demand_ids}")

    print("\n车辆ID列表:")
    print(f"  地面车ID: {instance.ground_veh_ids}")
    print(f"  无人机ID: {instance.drone_ids}")

    print("\n矩阵形状:")
    print(f"  坐标数组: {instance.np_coords.shape}")
    print(f"  距离矩阵: {instance.distance_matrix.shape}")
    print(f"  通信覆盖矩阵: {instance.comm_coverage_matrix.shape}")
    print(f"  行驶时间矩阵: {instance.travel_time_matrix.shape}")

    print("\n容量数组:")
    print(f"  {instance.capacity_array}")

    print("\n通信半径数组:")
    print(f"  {instance.radius_array}")

    print("\n分组详情:")
    for gid, nodes in sorted(instance.group_sets.items()):
        print(f"  组 {gid}: {sorted(nodes)}")
        # 计算组内节点的平均坐标
        group_coords = [instance.coords[n] for n in nodes]
        avg_x = sum(c[0] for c in group_coords) / len(group_coords)
        avg_y = sum(c[1] for c in group_coords) / len(group_coords)
        print(f"         中心坐标: ({avg_x:.2f}, {avg_y:.2f})")

    # 测试to_dict方法
    instance_dict = instance.to_dict()
    print(f"\nto_dict()键: {list(instance_dict.keys())}")

    print("\n✓ 测试6通过：算例属性访问正常")
    return instance


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#" * 80)
    print("#" + " " * 30 + "算例测试套件" + " " * 30 + "#")
    print("#" * 80)

    try:
        # 测试1: 基本算例
        instance1, folder1 = test_instance_creation_basic()

        # 测试2: K-means分组
        instance2, folder2 = test_instance_creation_kmeans()

        # 测试3: 网格分组
        instance3, folder3 = test_instance_creation_grid()

        # 测试4: 距离阈值分组
        instance4, folder4 = test_instance_creation_distance()

        # 测试5: 保存和读取
        loaded_instance = test_instance_save_and_load(instance2, folder2)

        # 测试6: 属性和方法
        instance6 = test_instance_properties()

        # 总结
        print("\n" + "=" * 80)
        print("测试总结")
        print("=" * 80)
        print("✓ 所有测试通过！")
        print("\n测试的算例保存在:")
        print(f"  - {folder1}")
        print(f"  - {folder2}")
        print(f"  - {folder3}")
        print(f"  - {folder4}")

        print("\n分组功能验证:")
        print(f"  ✓ 不分组 (none): {len(instance1.group_sets)}个组")
        print(f"  ✓ K-means分组: {len(instance2.group_sets)}个组")
        print(f"  ✓ 网格分组: {len(instance3.group_sets)}个组")
        print(f"  ✓ 距离阈值分组: {len(instance4.group_sets)}个组")

        print("\n核心功能验证:")
        print("  ✓ 算例生成")
        print("  ✓ 算例保存")
        print("  ✓ 算例读取")
        print("  ✓ 算例绘制")
        print("  ✓ 分组功能")
        print("  ✓ 数据一致性")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
