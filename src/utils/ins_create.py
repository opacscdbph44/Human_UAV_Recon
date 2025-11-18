import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tsplib95  # type: ignore
from sklearn.cluster import KMeans

from src.config import Config
from src.element import InstanceClass


def create_by_tsp(
    prob_config: Config,
    save_folder: Path,
    save_file: bool = True,
) -> InstanceClass:
    """根据.tsp格式的算例文件，生成算例实例对象

    Args:
        prob_config (Config): 问题求解的所有参数配置，可以读取节点、车辆等信息

    Returns:
        InstanceClass: 生成的算例实例对象
    """

    instance_name = prob_config.instance_param.name

    # 读取.tsp文件，生成算例对象
    tsp_problem = read_tsp_file(instance_name)

    # 根据要求，是否保存生成的新算例

    #  提取需求节点坐标信息
    demand_coords = demand_selection(
        tsp_problem,
        prob_config,
    )

    #  确定需求点连通性和优先级
    demand_priority, demand_accessible = demand_distrib(
        demand_coords,
        prob_config,
    )

    base_num = prob_config.instance_param.base_num
    steiner_num = prob_config.instance_param.steiner_num
    priority = [0] * base_num + demand_priority + [0] * steiner_num
    accessible = [1] * base_num + demand_accessible + [1] * steiner_num

    #  确定基地坐标
    base_coords = base_selection(prob_config, demand_coords)

    #  初始化Stiner点信息
    stiner_coords = stiner_selection(prob_config, demand_coords)

    # 合并全体坐标信息
    all_coords = base_coords + demand_coords + stiner_coords

    # 建立新的instance
    new_instance = InstanceClass.new_instance(
        prob_config,
        all_coords,
        priority,
        accessible,
    )
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:"
        f"成功生成算例对象: {new_instance}"
    )
    if save_file:
        save_instance(
            prob_config,
            new_instance,
            save_folder,
        )

    return new_instance


def read_tsp_file(
    prob_name: str, folder_path: Path = Path("data/ALL_tsp_file")
) -> "tsplib95.models.StandardProblem":
    """读取.tsp格式的算例文件，生成算例实例对象

    Args:
        file_path (Path): .tsp格式的算例文件路径

    Returns:
        InstanceClass: 生成的算例实例对象
    """

    # 检查prob_name是否以.tsp结尾
    if not prob_name.endswith(".tsp"):
        error_msg = f"当前仅支持读取.tsp格式的算例文件，请检查算例文件名: {prob_name}！"
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
        raise ValueError(error_msg)

    # 生成tsp文件完整路径
    folder_path = folder_path / prob_name

    if not folder_path.exists():
        error_msg = f"指定的.tsp算例文件不存在，请检查路径: {folder_path}！"
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
        raise FileNotFoundError(error_msg)

    # 读取.tsp文件
    tsp_problem = tsplib95.load(folder_path)

    # 检查是不是EUC_2D类型
    if tsp_problem.edge_weight_type != "EUC_2D":
        error_msg = (
            f"当前仅支持EUC_2D类型的.tsp算例文件，"
            f"请检查算例文件: {prob_name} 的edge_weight_type属性！"
        )
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
        raise ValueError(error_msg)

    return tsp_problem


def demand_selection(
    tsp_problem: "tsplib95.models.StandardProblem",
    prob_config: Config,
) -> list:
    all_nodes = list(tsp_problem.get_nodes())

    # 提取需求点数量
    demand_num = prob_config.instance_param.demand_num
    # 提取需求点选择方式
    demand_selection_method = prob_config.instance_param.demand_selection

    if demand_selection_method == "Top_k":
        selected_nodes = all_nodes[:demand_num]
    elif demand_selection_method == "Random":
        selected_nodes = random.sample(all_nodes, demand_num)
    else:
        error_msg = (
            f"不支持的需求点选择方式: {demand_selection_method}，"
            f"请检查prob_config.instance_param.demand_selection参数！"
        )
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
        raise ValueError(error_msg)

    demand_coords = [tsp_problem.node_coords[node] for node in selected_nodes]  # type: ignore
    return demand_coords


def demand_distrib(
    demand_coords: list,
    prob_config: Config,
):
    demand_num = prob_config.instance_param.demand_num

    distrib_type = prob_config.instance_param.distrib_type
    distrib_center = prob_config.instance_param.distrib_center
    isolate_rate = prob_config.instance_param.isolate_ratio

    total = len(demand_coords)
    # 初始化可达性和优先级向量
    priority = [0 for _ in range(demand_num)]
    accessible = [1 for _ in range(demand_num)]
    # 随机分配
    if distrib_type == "Random":
        # 计算需要设为 isolated 的数量
        total_points = len(demand_coords)
        cnt_isolated = round(isolate_rate * total_points)

        # 随机选择cnt_isolated个点作为isolated
        all_indices = list(range(len(demand_coords)))
        isolated_indices = random.sample(all_indices, cnt_isolated)
        isolated_set = set(isolated_indices)

        for idx, coord in enumerate(demand_coords):
            is_accessible = idx not in isolated_set
            accessible[idx] = 1 if is_accessible else 0
            priority[idx] = random.choice([1, 2, 3, 4, 5])
    # 线性中心分配
    elif distrib_type == "Linear":
        a, b = distrib_center
        distance_list: list = []
        epsilon = 0.01  # 用于确保权重为正

        if total > 0:
            # 计算每个需求点到分布中心的距离
            for idx, coord in enumerate(demand_coords):
                x, y = coord
                distance = ((x - a) ** 2 + (y - b) ** 2) ** 0.5
                distance_list.append((idx, distance))

            all_distance = [dist for _, dist in distance_list]
            max_distance = max(all_distance)
            min_distance = min(all_distance)
            delta_d_linear = max_distance - min_distance

            distance_list.sort(key=lambda x: x[1])
            cnt_isolated = round(isolate_rate * total)
            isolated_indices = [idx for idx, _ in distance_list[-cnt_isolated:]]
            isolated_set = set(isolated_indices)

            for idx, dist in distance_list:
                is_accessible = idx not in isolated_set
                accessible[idx] = 1 if is_accessible else 0

                # 根据距离生成随机权重，然后给点指定权重
                norm_dist = (
                    (dist - min_distance) / delta_d_linear if delta_d_linear > 0 else 0
                )
                d = norm_dist  # 归一化距离映射到0-4区间
                # 确定权重
                weights = [
                    d**2 + epsilon,  # Priority 1
                    d * 0.5 + epsilon,  # Priority 2
                    (1 - 2 * abs(d - 0.5)) * 0.5
                    + epsilon,  # Priority 3 (peaks at d=0.5)
                    (1 - d) * 0.5 + epsilon,  # Priority 4
                    (1 - d) ** 2 + epsilon,  # Priority 5
                ]
                priority[idx] = random.choices([1, 2, 3, 4, 5], weights=weights, k=1)[0]
        # 径向中心分配
        elif distrib_type == "Radial":
            a, b = distrib_center
            epsilon = 0.01  # 用于确保权重为正

            if total > 0:
                # 获取所有坐标的x和y值用于归一化
                xs = [coord[0] for coord in demand_coords]
                ys = [coord[1] for coord in demand_coords]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # 计算归一化后的距离
                normalized_coords_with_dist = []  # (idx, dist_to_center_after_norm)
                for idx, coord in enumerate(demand_coords):
                    x, y = coord
                    norm_x = (
                        (x - min_x) / (max_x - min_x) if max_x - min_x != 0 else 0.0
                    )
                    norm_y = (
                        (y - min_y) / (max_y - min_y) if max_y - min_y != 0 else 0.0
                    )
                    dx = norm_x - a
                    dy = norm_y - b
                    dist = (dx * dx + dy * dy) ** 0.5
                    normalized_coords_with_dist.append((idx, dist))

                # 获取距离范围
                all_radial_dists = [dist for _, dist in normalized_coords_with_dist]
                min_d_radial = min(all_radial_dists)
                max_d_radial = max(all_radial_dists)
                delta_d_radial = max_d_radial - min_d_radial

                # 按距离排序并选择最近的点作为isolated
                normalized_coords_with_dist.sort(key=lambda item: item[1])
                cnt_isolated = round(isolate_rate * total)
                isolated_indices = [
                    idx for idx, _ in normalized_coords_with_dist[:cnt_isolated]
                ]
                isolated_set = set(isolated_indices)

                # 分配可达性和优先级
                for idx, dist_val in normalized_coords_with_dist:
                    is_accessible = idx not in isolated_set
                    accessible[idx] = 1 if is_accessible else 0

                    # 根据归一化距离生成随机权重
                    norm_dist = (
                        (dist_val - min_d_radial) / delta_d_radial
                        if delta_d_radial > 0
                        else 0
                    )
                    d = norm_dist  # d=0 closest, d=1 farthest

                    weights = [
                        d**2 + epsilon,  # Priority 1
                        d * 0.5 + epsilon,  # Priority 2
                        (1 - 2 * abs(d - 0.5)) * 0.5 + epsilon,  # Priority 3
                        (1 - d) * 0.5 + epsilon,  # Priority 4
                        (1 - d) ** 2 + epsilon,  # Priority 5
                    ]
                    priority[idx] = random.choices(
                        [1, 2, 3, 4, 5], weights=weights, k=1
                    )[0]
    else:
        error_msg = (
            f"不支持的需求点分布方式: {distrib_type}，"
            f"请检查prob_config.instance_param.distrib_type参数！"
        )
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
        raise ValueError(error_msg)

    return priority, accessible


def base_selection(
    prob_config: Config,
    demand_coords: list,
) -> list:
    base_coords: list = []

    base_num = prob_config.instance_param.base_num
    base_select_mode = prob_config.instance_param.base_select_mode
    base_select_param = prob_config.instance_param.base_select_param

    coord_xs = [coord[0] for coord in demand_coords]
    coord_ys = [coord[1] for coord in demand_coords]
    min_x, max_x = min(coord_xs), max(coord_xs)
    min_y, max_y = min(coord_ys), max(coord_ys)

    if base_select_mode == "assign":
        if len(base_select_param) != base_num:
            error_msg = (
                f"数量不符（{len(base_select_param)} != {base_num}），"
                "请检查prob_config.instance_param.base_select_param参数！"
            )
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
            raise ValueError(error_msg)

        for i in range(base_num):
            rel_x, rel_y = base_select_param[i]
            # 将0-1范围内的相对坐标转换为实际坐标
            abs_x = min_x + rel_x * (max_x - min_x)
            abs_y = min_y + rel_y * (max_y - min_y)
            base_coords.append((abs_x, abs_y))

    elif base_select_mode == "random":
        # random模式：在坐标范围内随机生成指定数量的基站
        for i in range(base_num):
            # 随机生成坐标
            rand_x = min_x + random.random() * (max_x - min_x)
            rand_y = min_y + random.random() * (max_y - min_y)
            base_coords.append((rand_x, rand_y))
    else:
        error_msg = (
            f"不支持的基地点选择方式: {base_select_mode}，"
            f"请检查prob_config.instance_param.base_select_mode参数！"
        )
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
        raise ValueError(error_msg)

    return base_coords


def stiner_selection(
    prob_config: Config,
    demand_coords: list,
) -> list:
    stiner_coords: list = []

    steiner_num = prob_config.instance_param.steiner_num
    steiner_mode = prob_config.instance_param.steiner_generation_mode

    coord_xs = [coord[0] for coord in demand_coords]
    coord_ys = [coord[1] for coord in demand_coords]
    min_x, max_x = min(coord_xs), max(coord_xs)
    min_y, max_y = min(coord_ys), max(coord_ys)

    if steiner_mode == "grid":
        grid_size = prob_config.instance_param.steiner_grid_size

        # 计算网格间隔
        x_step = (max_x - min_x) / (grid_size[0] + 1)
        y_step = (max_y - min_y) / (grid_size[1] + 1)

        # 生成网格点，保留2位小数
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = round(min_x + (i + 1) * x_step, 2)
                y = round(min_y + (j + 1) * y_step, 2)
                stiner_coords.append((x, y))
    elif steiner_mode == "assign":
        assigned_coords = prob_config.instance_param.steiner_coords
        if len(assigned_coords) != steiner_num:
            error_msg = (
                f"数量不符（{len(assigned_coords)} != {steiner_num}），"
                "请检查prob_config.instance_param.steiner_coords参数！"
            )
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
            raise ValueError(error_msg)
        for coord in assigned_coords:
            rel_x, rel_y = coord
            # 将0-1范围内的相对坐标转换为实际坐标
            abs_x = min_x + rel_x * (max_x - min_x)
            abs_y = min_y + rel_y * (max_y - min_y)
            stiner_coords.append((abs_x, abs_y))
    elif steiner_mode == "kmeans":
        demand_coords_np = np.array(demand_coords)
        kmeans = KMeans(n_clusters=steiner_num, random_state=42)
        centroids = kmeans.fit(demand_coords_np).cluster_centers_
        stiner_coords = centroids.tolist()
        stiner_coords = [tuple(coord) for coord in stiner_coords]
    else:
        error_msg = (
            f"不支持的Steiner点生成方式: {steiner_mode}，"
            f"请检查prob_config.instance_param.steiner_generation_mode参数！"
        )
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}")
        raise ValueError(error_msg)

    return stiner_coords


def save_instance(
    prob_config: Config,
    instance: InstanceClass,
    instance_folder: Path,
) -> str:
    """将参数配置、算例详情存入制定文件夹,存储格式为json"""

    # 提取时间戳,作为文件唯一标识
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{time_str}.json"
    file_path = instance_folder / file_name
    
    # 确保父文件夹存在,不存在则自动创建
    instance_folder.mkdir(parents=True, exist_ok=True)

    # 将参数配置、算例详情存入json文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            {"instance": instance.to_dict()},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"成功保存算例文件: {file_path}"
    )

    return str(file_path)
