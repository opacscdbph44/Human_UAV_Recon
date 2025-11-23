from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle

from src.element import (
    InstanceClass,
    Solution,
)
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
    绘制InstanceClass实例中的所有节点，包括基站、需求点

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
                radius = instance.radius_array[veh_id]
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
