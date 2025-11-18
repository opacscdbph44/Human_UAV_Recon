from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle

from src.element import (
    InstanceClass,
    Solution,
)
from src.evaluator import Population
from src.utils.file_funcs import (
    query_instance_folder,
    query_result_folder,
)


def plot_instance(
    instance: InstanceClass,
    figsize: tuple = (10, 8),
    with_labels: bool = True,
    node_size: int = 150,
    font_size: int = 8,
    node_edge_width: float = 0.5,
    transparency: float = 0.75,
    figure_folder_path: Path | None = None,
):
    """
    绘制InstanceClass实例中的所有节点，包括基站、需求点和Steiner点

    Args:
        instance (InstanceClass): 要绘制的实例对象
        figsize (tuple): 图形尺寸，默认为(10, 8)
        with_labels (bool): 是否显示节点标签，默认为True
        node_size (int): 节点大小，默认为150
        font_size (int): 标签字体大小，默认为8
        node_edge_width (float): 节点边框宽度，默认为0.5
        transparency (float): 节点透明度，默认为0.75
        figure_folder_path (str): 保存图形的路径，如果为None则不保存

    Returns:
        matplotlib.figure.Figure: 生成的图形对象
    """
    # 设置全局字体为宋体
    plt.rcParams["font.family"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建宋体字体属性对象
    simSun = FontProperties(family="SimSun")

    # 创建无向图
    G: nx.Graph = nx.Graph()

    # 节点坐标字典
    pos = {}

    # 添加基站节点
    base_nodes = []
    for base_id in instance.base_ids:
        G.add_node(base_id)
        pos[base_id] = instance.coords[base_id]
        base_nodes.append(base_id)

    # 添加需求点，按照accessible分类
    demand_nodes: dict[str, list[int]] = {
        "accessible": [],  # 地面可达
        "inaccessible": [],  # 地面不可达
    }
    for demand_id in instance.demand_ids:
        G.add_node(demand_id)
        pos[demand_id] = instance.coords[demand_id]
        if instance.accessible[demand_id] == 1:
            demand_nodes["accessible"].append(demand_id)
        else:
            demand_nodes["inaccessible"].append(demand_id)

    # 添加Steiner点
    steiner_nodes = []
    for steiner_id in instance.steiner_ids:
        G.add_node(steiner_id)
        pos[steiner_id] = instance.coords[steiner_id]
        steiner_nodes.append(steiner_id)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制基站节点（星形）
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=base_nodes,
        node_color="#3498db",
        node_size=node_size + 100,
        node_shape="*",
        label="基地",
        edgecolors="black",
        linewidths=node_edge_width,
        ax=ax,
        alpha=transparency,
    )

    # 绘制地面可达需求点
    if demand_nodes["accessible"]:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=demand_nodes["accessible"],
            node_color="#2ecc71",
            node_size=node_size,
            node_shape="o",
            label="地面可达节点",
            edgecolors="black",
            linewidths=node_edge_width,
            ax=ax,
            alpha=transparency,
        )

    # 绘制地面不可达需求点
    if demand_nodes["inaccessible"]:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=demand_nodes["inaccessible"],
            node_color="#e74c3c",
            node_size=node_size,
            node_shape="o",
            label="地面不可达节点",
            edgecolors="black",
            linewidths=node_edge_width,
            ax=ax,
            alpha=transparency,
        )

    # 绘制Steiner点
    if steiner_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=steiner_nodes,
            node_color="#f39c12",
            node_size=node_size,
            node_shape="s",
            label="候选部署点",
            edgecolors="black",
            linewidths=node_edge_width,
            ax=ax,
            alpha=transparency,
        )

    # 添加节点标签
    if with_labels:
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=font_size,
            font_family="SimSun",
            font_weight="bold",
            ax=ax,
        )

    # 设置图形标题和图例
    plt.title(f"算例: {instance.name}", fontproperties=simSun, fontsize=12)
    plt.legend(prop=simSun, loc="best")

    # 设置坐标轴标签
    ax.set_xlabel("X", fontproperties=simSun, fontsize=10)
    ax.set_ylabel("Y", fontproperties=simSun, fontsize=10)

    # 确保x和y轴比例相同
    ax.set_aspect("equal")

    # 计算坐标范围
    coords_array = instance.np_coords
    x_values = coords_array[:, 0]
    y_values = coords_array[:, 1]

    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    # 设置显示范围，留出10%边距
    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    ax.set_xlim((x_min - x_margin, x_max + x_margin))
    ax.set_ylim((y_min - y_margin, y_max + y_margin))

    # 打开坐标轴和网格
    plt.axis("on")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 确保刻度可见
    plt.tick_params(
        axis="both",
        which="both",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
    )

    # 调整布局
    plt.tight_layout()

    # 保存图形
    if figure_folder_path is not None:
        # 确保目录存在
        figure_folder_path.mkdir(parents=True, exist_ok=True)
        save_path = (
            figure_folder_path
            / f"instance_{instance.name}_{instance.demand_num}figure.jpg"
        )
        plt.savefig(
            save_path,
            dpi=600,
            bbox_inches="tight",
        )
    else:
        instance_folder = query_instance_folder(instance.prob_config)
        # 确保目录存在
        instance_folder.mkdir(parents=True, exist_ok=True)
        default_save_path = (
            instance_folder
            / f"instance_{instance.name}_{instance.demand_num}figure.jpg"
        )
        plt.savefig(
            default_save_path,
            dpi=600,
            bbox_inches="tight",
        )
        save_path = default_save_path

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"算例图形已保存至: {save_path}"
    )

    # 显示图片（如果在交互环境中）
    # plt.show()
    plt.close(fig)


def plot_solution(
    instance: InstanceClass,
    solution: Solution,
    figsize: tuple = (12, 10),
    with_labels: bool = True,
    node_size: int = 150,
    font_size: int = 8,
    node_edge_width: float = 0.5,
    transparency: float = 0.75,
    arrow_width: float = 2.0,
    arrow_head_width: float = 10.0,
    arrow_head_length: float = 15.0,
    coverage_line_style: str = "--",
    coverage_line_width: float = 1.0,
    figure_folder_path: Path | None = None,
):
    """
    绘制解决方案,包括车辆路径、覆盖关系和通信半径

    Args:
        instance (InstanceClass): 算例对象
        solution: 解决方案对象
        figsize (tuple): 图形尺寸,默认为(12, 10)
        with_labels (bool): 是否显示节点标签,默认为True
        node_size (int): 节点大小,默认为150
        font_size (int): 标签字体大小,默认为8
        node_edge_width (float): 节点边框宽度,默认为0.5
        transparency (float): 节点透明度,默认为0.75
        arrow_width (float): 箭头线宽,默认为2.0
        arrow_head_width (float): 箭头头部宽度,默认为10.0
        arrow_head_length (float): 箭头头部长度,默认为15.0
        coverage_line_style (str): 覆盖线样式,默认为"--"
        coverage_line_width (float): 覆盖线宽度,默认为1.0
        figure_folder_path (Path): 保存图形的路径,如果为None则使用默认路径

    Returns:
        matplotlib.figure.Figure: 生成的图形对象
    """
    # 设置全局字体为宋体
    plt.rcParams["font.family"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建宋体字体属性对象
    simSun = FontProperties(family="SimSun")

    # 创建无向图
    G: nx.Graph = nx.Graph()

    # 节点坐标字典
    pos = {}

    # 添加所有节点
    for i, coord in enumerate(instance.coords):
        G.add_node(i)
        pos[i] = coord

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 定义车辆颜色方案(使用更多颜色)
    vehicle_colors = [
        "#e74c3c",  # 红色
        "#3498db",  # 蓝色
        "#2ecc71",  # 绿色
        "#f39c12",  # 橙色
        "#9b59b6",  # 紫色
        "#1abc9c",  # 青色
        "#e67e22",  # 深橙
        "#34495e",  # 深灰
        "#16a085",  # 深青
        "#c0392b",  # 深红
    ]

    # 绘制车辆路径和覆盖关系
    for veh_id, route in enumerate(solution.routes):
        if not route:
            continue

        # 获取车辆颜色
        color = vehicle_colors[veh_id % len(vehicle_colors)]

        # 绘制路径箭头
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]

            # 获取起点和终点坐标
            x1, y1 = pos[from_node]
            x2, y2 = pos[to_node]

            # 绘制箭头
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=f"->,head_width={arrow_head_width/30},head_length={arrow_head_length/30}",
                    color=color,
                    lw=arrow_width,
                    alpha=0.8,
                ),
            )

        # 绘制覆盖关系和通信半径
        visited_nodes_with_coverage = set()  # 记录有覆盖的访问节点

        for (v_id, visit_node), covered_nodes in solution.coverage.items():
            if v_id != veh_id or not covered_nodes:
                continue

            visited_nodes_with_coverage.add(visit_node)

            # 获取访问节点坐标
            x_visit, y_visit = pos[visit_node]

            # 检查是否存在被覆盖点不是visit_node的情况
            has_other_covered_nodes = any(
                covered_node != visit_node for covered_node in covered_nodes
            )

            # 只有存在其他被覆盖节点时才绘制通信半径圆
            if has_other_covered_nodes:
                # 绘制通信半径圆
                radius = instance.comm_radius_array[veh_id]
                circle = Circle(
                    (x_visit, y_visit),
                    radius,
                    facecolor=color,
                    fill=True,
                    alpha=0.15,
                    linestyle="-",
                    linewidth=1.5,
                    edgecolor=color,
                )
                ax.add_patch(circle)

            # 绘制覆盖虚线
            for covered_node in covered_nodes:
                x_covered, y_covered = pos[covered_node]
                ax.plot(
                    [x_visit, x_covered],
                    [y_visit, y_covered],
                    linestyle=coverage_line_style,
                    color=color,
                    linewidth=coverage_line_width,
                    alpha=0.6,
                )

    # 绘制基站节点(星形)
    base_nodes = instance.base_ids
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=base_nodes,
        node_color="#3498db",
        node_size=node_size + 100,
        node_shape="*",
        label="基地",
        edgecolors="black",
        linewidths=node_edge_width,
        ax=ax,
        alpha=transparency,
    )

    # 绘制需求点
    demand_nodes: dict = {"accessible": [], "inaccessible": []}
    for demand_id in instance.demand_ids:
        if instance.accessible[demand_id] == 1:
            demand_nodes["accessible"].append(demand_id)
        else:
            demand_nodes["inaccessible"].append(demand_id)

    if demand_nodes["accessible"]:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=demand_nodes["accessible"],
            node_color="#2ecc71",
            node_size=node_size,
            node_shape="o",
            label="地面可达节点",
            edgecolors="black",
            linewidths=node_edge_width,
            ax=ax,
            alpha=transparency,
        )

    if demand_nodes["inaccessible"]:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=demand_nodes["inaccessible"],
            node_color="#e74c3c",
            node_size=node_size,
            node_shape="o",
            label="地面不可达节点",
            edgecolors="black",
            linewidths=node_edge_width,
            ax=ax,
            alpha=transparency,
        )

    # 绘制Steiner点
    steiner_nodes = instance.steiner_ids
    if steiner_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=steiner_nodes,
            node_color="#f39c12",
            node_size=node_size,
            node_shape="s",
            label="候选部署点",
            edgecolors="black",
            linewidths=node_edge_width,
            ax=ax,
            alpha=transparency,
        )

    # 添加节点标签
    if with_labels:
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=font_size,
            font_family="SimSun",
            font_weight="bold",
            ax=ax,
        )

    # 设置图形标题
    objectives_str = ", ".join([f"{obj:.4f}" for obj in solution.objectives])
    title_text = (
        f"算例: {instance.name} | 规模: N = {instance.demand_num} | "
        f"算法：{solution.Solver_name} | "
        f"目标值: [{objectives_str}] | "
        f"状态: {solution.status}"
    )
    plt.title(title_text, fontproperties=simSun, fontsize=12)
    plt.legend(prop=simSun, loc="best")

    # 设置坐标轴
    ax.set_xlabel("X", fontproperties=simSun, fontsize=10)
    ax.set_ylabel("Y", fontproperties=simSun, fontsize=10)
    ax.set_aspect("equal")

    # 设置显示范围
    coords_array = instance.np_coords
    x_values = coords_array[:, 0]
    y_values = coords_array[:, 1]

    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    ax.set_xlim((x_min - x_margin, x_max + x_margin))
    ax.set_ylim((y_min - y_margin, y_max + y_margin))

    plt.axis("on")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tick_params(
        axis="both",
        which="both",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
    )

    # 紧凑布局
    plt.tight_layout()

    # 保存图形
    if figure_folder_path is not None:
        # 确保目录存在
        figure_folder_path.mkdir(parents=True, exist_ok=True)
        save_path = (
            figure_folder_path / f"solution_{instance.name}_{solution.Solver_name}.jpg"
        )
    else:
        result_folder = query_result_folder(
            instance.prob_config,
            solution.Solver_name,
        )
        figure_folder_path = result_folder / "figures"
        # 确保目录存在
        figure_folder_path.mkdir(parents=True, exist_ok=True)
        save_path = (
            figure_folder_path / f"solution_{instance.name}_{solution.Solver_name}.jpg"
        )

    plt.savefig(save_path, dpi=600, bbox_inches="tight")

    # 显示图片（如果在交互环境中）
    # plt.show()
    plt.close(fig)

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:"
        f"解决方案图形已保存至: {save_path}"
    )


def plot_pareto_front(
    population: Population,
    figure_folder_path: Path | str | None = None,
    figsize: tuple = (10, 8),
    dpi: int = 300,
) -> None:
    """
    绘制Pareto前沿面

    Args:
        population: 种群对象
        figure_folder_path: 图片保存路径
        figsize: 图片大小
        dpi: 图片分辨率
        tolerance: 判断重复解的精度容差(已弃用,保留参数以兼容旧代码)
    """
    # 求解器名称映射字典
    solver_name_map = {
        "gurobi_epsilon_constraint_solver": r"$\epsilon$ - 约束法",
        # 在此添加更多映射
        # "nsga2": "NSGA-II",
        # "moead": "MOEA/D",
        # "spea2": "SPEA2",
    }

    # 设置全局字体为宋体
    plt.rcParams["font.family"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建宋体字体属性对象
    simSun = FontProperties(family="SimSun")

    # 提取Pareto前沿解
    pareto_elite_front: List[Solution] = population.get_elite_solutions()
    total_first_front = len(population.fronts_idx[0])

    # 将Pareto前沿按照obj1升序排序
    pareto_elite_front.sort(key=lambda sol: sol.objectives[0])

    if not pareto_elite_front:
        print("警告: 没有找到Pareto前沿解")
        return

    # 提取目标函数值
    pareto_obj1 = [sol.objectives[0] for sol in pareto_elite_front]
    pareto_obj2 = [sol.objectives[1] for sol in pareto_elite_front]

    # 创建图形
    plt.figure(figsize=figsize, dpi=dpi)

    # 绘制Pareto前沿解
    plt.scatter(
        pareto_obj1,
        pareto_obj2,
        c="red",
        marker="*",
        s=200,
        alpha=0.8,
        edgecolors="black",
        linewidths=1.5,
        label=f"非重复Pareto前沿解 (n={len(pareto_elite_front)}/{total_first_front})",
    )

    # 使用阶梯线连接Pareto前沿点，以更好地可视化支配区域
    if len(pareto_elite_front) > 1:
        step_x, step_y = [], []
        # 从第一个点开始
        step_x.append(pareto_obj1[0])
        step_y.append(pareto_obj2[0])
        # 依次连接后续点
        for i in range(len(pareto_obj1) - 1):
            # 水平线到下一个点的x坐标
            step_x.append(pareto_obj1[i + 1])
            step_y.append(pareto_obj2[i])
            # 垂直线到下一个点
            step_x.append(pareto_obj1[i + 1])
            step_y.append(pareto_obj2[i + 1])

        plt.plot(
            step_x,
            step_y,
            "r--",
            alpha=0.5,
            linewidth=1.5,
            label="Pareto前沿",
        )

    # 设置坐标轴标签
    plt.xlabel(r"$f_1$: 通信保障任务持续时长", fontproperties=simSun, fontsize=12)
    plt.ylabel(r"$f_2$: 应急通信收益", fontproperties=simSun, fontsize=12)

    # 获取显示用的求解器名称
    display_solver_name = solver_name_map.get(
        population.solver_name, population.solver_name
    )

    # 获取hv
    if not population.hv_calculated:
        population.calculate_hypervolume()
    # 设置标题
    title_text = (
        f"Pareto前沿面\n"
        f"求解器:{display_solver_name} | "
        f"非重复前沿解数量: {len(pareto_elite_front)}/{total_first_front} 个 | 超体积: HV = {population.hv_value:.4f}"
    )

    plt.title(
        title_text,
        fontproperties=simSun,
        fontsize=14,
        fontweight="bold",
    )

    # 添加网格
    plt.grid(True, linestyle="--", alpha=0.3)

    # 添加图例
    plt.legend(loc="best", prop=simSun, fontsize=10)

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    if figure_folder_path:
        figure_folder_path = Path(figure_folder_path)
        figure_folder_path.mkdir(parents=True, exist_ok=True)

        save_path = figure_folder_path / "pareto_front.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"非重复Pareto前沿图已保存至: {save_path} "
            f"(共{len(pareto_elite_front)}个解)"
        )

    # 显示图片（如果在交互环境中）
    # plt.show()
    plt.close()


def plot_pareto_front_comparison(
    population1: Population,
    population2: Population,
    population1_label: str = "种群1",
    population2_label: str = "种群2",
    figure_folder_path: Path | str | None = None,
    figsize: tuple = (12, 9),
    dpi: int = 300,
) -> None:
    """
    对比绘制两个种群的Pareto前沿面

    Args:
        population1: 第一个种群对象
        population2: 第二个种群对象
        population1_label: 第一个种群的标签名称
        population2_label: 第二个种群的标签名称
        figure_folder_path: 图片保存路径
        figsize: 图片大小
        dpi: 图片分辨率
    """
    # 求解器名称映射字典
    solver_name_map = {
        "gurobi_epsilon_constraint_solver": r"$\epsilon$ - 约束法",
        # 在此添加更多映射
    }

    # 设置全局字体为宋体
    plt.rcParams["font.family"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建宋体字体属性对象
    simSun = FontProperties(family="SimSun")

    # 提取两个种群的Pareto前沿解
    pareto_elite_front1: List[Solution] = population1.get_elite_solutions()
    pareto_elite_front2: List[Solution] = population2.get_elite_solutions()

    # 获取总的前沿解数量
    total_first_front1 = len(population1.fronts_idx[0])
    total_first_front2 = len(population2.fronts_idx[0])

    # 按照obj1升序排序
    pareto_elite_front1.sort(key=lambda sol: sol.objectives[0])
    pareto_elite_front2.sort(key=lambda sol: sol.objectives[0])

    if not pareto_elite_front1 and not pareto_elite_front2:
        print("警告: 两个种群都没有找到Pareto前沿解")
        return

    # 提取目标函数值
    pareto_obj1_pop1 = [sol.objectives[0] for sol in pareto_elite_front1]
    pareto_obj2_pop1 = [sol.objectives[1] for sol in pareto_elite_front1]

    pareto_obj1_pop2 = [sol.objectives[0] for sol in pareto_elite_front2]
    pareto_obj2_pop2 = [sol.objectives[1] for sol in pareto_elite_front2]

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 定义两个种群的颜色和标记
    color1 = "#e74c3c"  # 红色
    color2 = "#3498db"  # 蓝色
    marker1 = "*"
    marker2 = "o"

    # 计算合适的坐标轴范围
    all_obj1 = pareto_obj1_pop1 + pareto_obj1_pop2
    all_obj2 = pareto_obj2_pop1 + pareto_obj2_pop2

    x_min, x_max, y_min, y_max = 0.0, 0.0, 0.0, 0.0
    x_margin, y_margin = 0.0, 0.0
    if all_obj1 and all_obj2:
        # 强制从原点开始
        x_min = 0.0
        y_min = 0.0
        x_max = max(all_obj1)
        y_max = max(all_obj2)

        # 添加边距
        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0

    # 绘制第一个种群的Pareto前沿解
    if pareto_elite_front1:
        ax.scatter(
            pareto_obj1_pop1,
            pareto_obj2_pop1,
            c=color1,
            marker=marker1,
            s=200,
            alpha=0.8,
            edgecolors="black",
            linewidths=1.5,
            label=f"{population1_label} (n={len(pareto_elite_front1)}/{total_first_front1})",
            zorder=3,  # 确保点在线的上方
        )

        # 使用阶梯线连接第一个种群的Pareto前沿点
        if len(pareto_elite_front1) > 1:
            step_x1, step_y1 = [], []
            step_x1.append(pareto_obj1_pop1[0])
            step_y1.append(pareto_obj2_pop1[0])
            for i in range(len(pareto_obj1_pop1) - 1):
                step_x1.append(pareto_obj1_pop1[i + 1])
                step_y1.append(pareto_obj2_pop1[i])
                step_x1.append(pareto_obj1_pop1[i + 1])
                step_y1.append(pareto_obj2_pop1[i + 1])

            # 延伸到图表右边界
            if x_max > 0:
                step_x1.append(x_max + x_margin)
                step_y1.append(pareto_obj2_pop1[-1])

            ax.plot(
                step_x1,
                step_y1,
                color=color1,
                linestyle="--",
                alpha=0.5,
                linewidth=2.0,
                label=f"{population1_label} 前沿",
                zorder=2,
            )
            # 填充阶梯线下方的区域
            ax.fill_between(step_x1, step_y1, 0, color=color1, alpha=0.2, zorder=1)

    # 绘制第二个种群的Pareto前沿解
    if pareto_elite_front2:
        ax.scatter(
            pareto_obj1_pop2,
            pareto_obj2_pop2,
            c=color2,
            marker=marker2,
            s=150,
            alpha=0.7,
            edgecolors="black",
            linewidths=1.5,
            label=f"{population2_label} (n={len(pareto_elite_front2)}/{total_first_front2})",
            zorder=3,
        )

        # 使用阶梯线连接第二个种群的Pareto前沿点
        if len(pareto_elite_front2) > 1:
            step_x2, step_y2 = [], []
            step_x2.append(pareto_obj1_pop2[0])
            step_y2.append(pareto_obj2_pop2[0])
            for i in range(len(pareto_obj1_pop2) - 1):
                step_x2.append(pareto_obj1_pop2[i + 1])
                step_y2.append(pareto_obj2_pop2[i])
                step_x2.append(pareto_obj1_pop2[i + 1])
                step_y2.append(pareto_obj2_pop2[i + 1])

            # 延伸到图表右边界
            if x_max > 0:
                step_x2.append(x_max + x_margin)
                step_y2.append(pareto_obj2_pop2[-1])

            ax.plot(
                step_x2,
                step_y2,
                color=color2,
                linestyle="-.",
                alpha=0.5,
                linewidth=2.0,
                label=f"{population2_label} 前沿",
                zorder=2,
            )
            # 填充阶梯线下方的区域
            ax.fill_between(step_x2, step_y2, 0, color=color2, alpha=0.2, zorder=1)

    # 标记重叠的点（可选功能）
    # 找出在两个种群中目标值非常接近的解（容差为1e-6）
    tolerance = 1e-6
    overlapping_points = []
    for sol1 in pareto_elite_front1:
        for sol2 in pareto_elite_front2:
            if (
                abs(sol1.objectives[0] - sol2.objectives[0]) < tolerance
                and abs(sol1.objectives[1] - sol2.objectives[1]) < tolerance
            ):
                overlapping_points.append((sol1.objectives[0], sol1.objectives[1]))
                break

    # 用特殊标记标注重叠点
    if overlapping_points:
        overlap_x = [pt[0] for pt in overlapping_points]
        overlap_y = [pt[1] for pt in overlapping_points]
        ax.scatter(
            overlap_x,
            overlap_y,
            c="none",
            marker="o",
            s=400,
            edgecolors="green",
            linewidths=3.0,
            label=f"重叠解 (n={len(overlapping_points)})",
            zorder=4,
        )

    # 设置坐标轴标签
    ax.set_xlabel(r"$f_1$: 通信保障任务持续时长", fontproperties=simSun, fontsize=12)
    ax.set_ylabel(r"$f_2$: 应急通信收益", fontproperties=simSun, fontsize=12)

    # 获取显示用的求解器名称
    display_solver_name1 = solver_name_map.get(
        population1.solver_name, population1.solver_name
    )
    display_solver_name2 = solver_name_map.get(
        population2.solver_name, population2.solver_name
    )

    # 计算超体积
    if not population1.hv_calculated:
        hv1 = population1.calculate_hypervolume()
    else:
        hv1 = population1.hv_value

    if not population2.hv_calculated:
        hv2 = population2.calculate_hypervolume()
    else:
        hv2 = population2.hv_value

    # 设置标题
    title_text = (
        f"Pareto前沿面对比\n"
        f"{population1_label} ({display_solver_name1}): "
        f"{len(pareto_elite_front1)}/{total_first_front1} 个解, HV = {hv1:.4f} | "
        f"{population2_label} ({display_solver_name2}): "
        f"{len(pareto_elite_front2)}/{total_first_front2} 个解, HV = {hv2:.4f}"
    )

    if overlapping_points:
        title_text += f"\n重叠解: {len(overlapping_points)} 个"

    ax.set_title(
        title_text,
        fontproperties=simSun,
        fontsize=14,
        fontweight="bold",
    )

    # 添加网格
    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

    # 添加图例
    ax.legend(loc="best", prop=simSun, fontsize=10, framealpha=0.9)

    # 设置坐标轴范围
    if all_obj1 and all_obj2:
        # 确保从原点 (0, 0) 开始显示
        ax.set_xlim(0, x_max + x_margin)
        ax.set_ylim(0, y_max + y_margin)

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    if figure_folder_path:
        figure_folder_path = Path(figure_folder_path)
        figure_folder_path.mkdir(parents=True, exist_ok=True)

        save_path = figure_folder_path / "pareto_front_comparison.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"Pareto前沿对比图已保存至: {save_path}"
        )

    # 显示图片（如果在交互环境中）
    # plt.show()
    plt.close(fig)


def plot_hv_convergence(
    pop_list: list[Population],
    figure_folder_path: Path | str | None = None,
    figsize: tuple = (12, 9),
    dpi: int = 600,
):
    """
    绘制超体积收敛曲线

    Args:
        pop_list: 种群列表
        figure_folder_path: 图片保存路径
        figsize: 图片大小
        dpi: 图片分辨率
    """
    # 设置全局字体为宋体
    plt.rcParams["font.family"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建宋体字体属性对象（为图例指定字体大小）
    simSun = FontProperties(family="SimSun", size=16)

    hv_dict = {}
    for pop in pop_list:
        if not pop.hv_history:
            if pop.solver_name == "gurobi_epsilon_constraint_solver":
                constant_hv = pop.calculate_hypervolume()
                pop.hv_history = [{"hypervolume": constant_hv}] * 200
            else:
                raise ValueError(
                    f"种群 {pop.solver_name} 缺少 hv_history 数据，无法绘制收敛曲线。"
                )
        # 从 list of dict 中提取超体积值
        hv_values = [entry.get("hypervolume", 0.0) for entry in pop.hv_history]
        hv_dict[pop.solver_name] = hv_values

    solver_name_map = {
        "gurobi_epsilon_constraint_solver": r"$\epsilon$ - 约束法",
        "NSGA_II": "NSGA-II",
        "MOEA_D": "MOEA/D",
        "ALNS_NSGA_II_n": "I-ALNS+NSGA-II",
    }
    # 绘制收敛曲线
    plt.figure(figsize=figsize, dpi=dpi)
    for solver_name, hv_values in hv_dict.items():
        iterations = list(range(len(hv_values)))
        display_name = solver_name_map.get(solver_name, solver_name)
        plt.plot(iterations, hv_values, label=display_name, linewidth=2, alpha=0.8)

    plt.xlabel("迭代次数", fontproperties=simSun, fontsize=14)
    plt.ylabel("超体积 (HV)", fontproperties=simSun, fontsize=14)
    # plt.title("超体积收敛曲线", fontproperties=simSun, fontsize=14, fontweight="bold")
    plt.legend(prop=simSun, loc="best")  # 字体大小已在FontProperties中设置
    plt.grid(True, linestyle="--", alpha=0.3)

    # 紧凑布局
    plt.tight_layout()

    # 保存图像
    if figure_folder_path:
        figure_folder_path = Path(figure_folder_path)
        figure_folder_path.mkdir(parents=True, exist_ok=True)

        save_path = figure_folder_path / "hv_convergence.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"超体积收敛曲线已保存至: {save_path}"
        )

    plt.close()


def plot_pareto_front_grid(
    populations_data: List[dict],
    figure_folder_path: Path | str | None = None,
    figsize: tuple = (20, 24),
    dpi: int = 600,
    nrows: int = 4,
    ncols: int = 2,
) -> None:
    """
    在一个网格中绘制多个规模的Pareto前沿对比图

    Args:
        populations_data: 包含种群数据的列表，每个元素是一个字典，包含:
            - 'population1': 第一个种群对象
            - 'population2': 第二个种群对象
            - 'population1_label': 第一个种群的标签
            - 'population2_label': 第二个种群的标签
            - 'demand_num': 需求点数量（规模）
        figure_folder_path: 图片保存路径
        figsize: 整个图的大小
        dpi: 图片分辨率
        nrows: 子图行数
        ncols: 子图列数
    """
    # 设置全局字体为宋体
    plt.rcParams["font.family"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建宋体字体属性对象
    simSun = FontProperties(family="SimSun", size=14)

    # 创建子图网格，设置更合适的宽高比
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # 使用GridSpec来控制子图的宽高比（4:3）
    gs = fig.add_gridspec(nrows, ncols, hspace=0.3, wspace=0.15)
    axes = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

    # 定义两个种群的颜色和标记
    color1 = "#e74c3c"  # 红色
    color2 = "#3498db"  # 蓝色
    marker1 = "*"
    marker2 = "o"

    for idx, data in enumerate(populations_data):
        if idx >= len(axes):
            break

        ax = axes[idx]
        population1 = data["population1"]
        population2 = data["population2"]
        population1_label = data["population1_label"]
        population2_label = data["population2_label"]
        demand_num = data["demand_num"]

        # 提取两个种群的Pareto前沿解
        pareto_elite_front1: List[Solution] = population1.get_elite_solutions()
        pareto_elite_front2: List[Solution] = population2.get_elite_solutions()

        # 按照obj1升序排序
        pareto_elite_front1.sort(key=lambda sol: sol.objectives[0])
        pareto_elite_front2.sort(key=lambda sol: sol.objectives[0])

        if not pareto_elite_front1 and not pareto_elite_front2:
            ax.text(
                0.5,
                0.5,
                "无Pareto前沿解",
                ha="center",
                va="center",
                fontproperties=simSun,
            )
            ax.set_title(
                f"({idx+1}) 规模={demand_num}",
                fontproperties=simSun,
                fontsize=12,
                fontweight="bold",
            )
            continue

        # 提取目标函数值
        pareto_obj1_pop1 = [sol.objectives[0] for sol in pareto_elite_front1]
        pareto_obj2_pop1 = [sol.objectives[1] for sol in pareto_elite_front1]

        pareto_obj1_pop2 = [sol.objectives[0] for sol in pareto_elite_front2]
        pareto_obj2_pop2 = [sol.objectives[1] for sol in pareto_elite_front2]

        # 计算坐标轴范围（用于延伸阶梯线）
        all_obj1 = pareto_obj1_pop1 + pareto_obj1_pop2
        all_obj2 = pareto_obj2_pop1 + pareto_obj2_pop2

        x_max = max(all_obj1) if all_obj1 else 1.0
        y_max = max(all_obj2) if all_obj2 else 1.0
        x_margin = x_max * 0.1
        y_margin = y_max * 0.1

        # 绘制第一个种群
        if pareto_elite_front1:
            ax.scatter(
                pareto_obj1_pop1,
                pareto_obj2_pop1,
                c=color1,
                marker=marker1,
                s=120,
                alpha=0.8,
                edgecolors="black",
                linewidths=1.2,
                label=f"{population1_label} (n={len(pareto_elite_front1)})",
                zorder=3,
            )
            # 绘制阶梯线并填充覆盖范围
            if len(pareto_elite_front1) > 1:
                step_x, step_y = [], []
                step_x.append(pareto_obj1_pop1[0])
                step_y.append(pareto_obj2_pop1[0])
                for i in range(len(pareto_elite_front1) - 1):
                    step_x.append(pareto_obj1_pop1[i + 1])
                    step_y.append(pareto_obj2_pop1[i])
                    step_x.append(pareto_obj1_pop1[i + 1])
                    step_y.append(pareto_obj2_pop1[i + 1])

                # 延伸到图表右边界
                step_x.append(x_max + x_margin)
                step_y.append(pareto_obj2_pop1[-1])

                ax.plot(
                    step_x,
                    step_y,
                    color=color1,
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.5,
                    zorder=2,
                )
                # 填充阶梯线下方的区域
                ax.fill_between(step_x, step_y, 0, color=color1, alpha=0.15, zorder=1)

        # 绘制第二个种群
        if pareto_elite_front2:
            ax.scatter(
                pareto_obj1_pop2,
                pareto_obj2_pop2,
                c=color2,
                marker=marker2,
                s=80,
                alpha=0.7,
                edgecolors="black",
                linewidths=1.2,
                label=f"{population2_label} (n={len(pareto_elite_front2)})",
                zorder=3,
            )
            # 绘制阶梯线并填充覆盖范围
            if len(pareto_elite_front2) > 1:
                step_x, step_y = [], []
                step_x.append(pareto_obj1_pop2[0])
                step_y.append(pareto_obj2_pop2[0])
                for i in range(len(pareto_elite_front2) - 1):
                    step_x.append(pareto_obj1_pop2[i + 1])
                    step_y.append(pareto_obj2_pop2[i])
                    step_x.append(pareto_obj1_pop2[i + 1])
                    step_y.append(pareto_obj2_pop2[i + 1])

                # 延伸到图表右边界
                step_x.append(x_max + x_margin)
                step_y.append(pareto_obj2_pop2[-1])

                ax.plot(
                    step_x,
                    step_y,
                    color=color2,
                    linestyle="-.",
                    linewidth=2.0,
                    alpha=0.5,
                    zorder=2,
                )
                # 填充阶梯线下方的区域
                ax.fill_between(step_x, step_y, 0, color=color2, alpha=0.15, zorder=1)

        # 标记重叠的点
        # 找出在两个种群中目标值非常接近的解（容差为1e-6）
        tolerance = 1e-6
        overlapping_points = []
        for sol1 in pareto_elite_front1:
            for sol2 in pareto_elite_front2:
                if (
                    abs(sol1.objectives[0] - sol2.objectives[0]) < tolerance
                    and abs(sol1.objectives[1] - sol2.objectives[1]) < tolerance
                ):
                    overlapping_points.append((sol1.objectives[0], sol1.objectives[1]))
                    break

        # 用绿色圆圈标注重叠点
        if overlapping_points:
            overlap_x = [pt[0] for pt in overlapping_points]
            overlap_y = [pt[1] for pt in overlapping_points]
            # 绘制图中显示的大圆圈（不加入图例）
            ax.scatter(
                overlap_x,
                overlap_y,
                c="none",
                marker="o",
                s=250,
                edgecolors="green",
                linewidths=2.5,
                zorder=4,
            )
            # 绘制一个用于图例的小标记（不实际显示在图中）
            ax.scatter(
                [],
                [],
                c="none",
                marker="o",
                s=80,
                edgecolors="green",
                linewidths=2.5,
                label=f"重叠解 (n={len(overlapping_points)})",
                zorder=4,
            )

        # 设置标题（序号 + 规模）
        ax.set_title(
            f"({idx+1}) 算例规模={demand_num}",
            fontproperties=simSun,
            fontsize=16,
            fontweight="bold",
        )

        # 设置坐标轴标签
        ax.set_xlabel(
            r"$f_1$: 通信保障任务持续时长", fontproperties=simSun, fontsize=14
        )
        ax.set_ylabel(r"$f_2$: 应急通信收益", fontproperties=simSun, fontsize=14)

        # 设置坐标轴范围（从原点开始）
        if all_obj1 and all_obj2:
            ax.set_xlim(0, x_max + x_margin)
            ax.set_ylim(0, y_max + y_margin)

        # 添加图例
        ax.legend(prop=simSun, loc="best", framealpha=0.9)

        # 添加网格
        ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

        # 设置刻度标签字体大小
        ax.tick_params(labelsize=14)

        # 设置子图的宽高比为4:3
        ax.set_aspect("auto")

    # 隐藏多余的子图
    for idx in range(len(populations_data), len(axes)):
        axes[idx].axis("off")

    # 调整子图间距（不使用tight_layout，因为已经用GridSpec设置了间距）
    # plt.tight_layout()

    # 保存图像
    if figure_folder_path:
        figure_folder_path = Path(figure_folder_path)
        figure_folder_path.mkdir(parents=True, exist_ok=True)

        save_path = figure_folder_path / "pareto_front_grid_comparison.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"组合对比图已保存至: {save_path}"
        )

    plt.close()
