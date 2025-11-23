"""ALNS算法历史记录可视化工具"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.solver.alns_history import ALNSHistory


def plot_alns_convergence(
    history: ALNSHistory,
    save_path: str | Path | None = None,
    show_candidate: bool = True,
    figsize: tuple[int, int] = (12, 6),
):
    """
    绘制ALNS算法收敛曲线

    Args:
        history: ALNS历史记录器实例
        save_path: 图片保存路径(可选)
        show_candidate: 是否显示候选解曲线
        figsize: 图片尺寸
    """
    best_objectives = history.get_best_objectives()
    current_objectives = history.get_current_objectives()
    candidate_objectives = history.get_candidate_objectives()
    iterations = list(range(len(best_objectives)))

    plt.figure(figsize=figsize)

    # 绘制最优解曲线
    plt.plot(
        iterations,
        best_objectives,
        label="Best Solution",
        color="green",
        linewidth=2,
        marker="o",
        markersize=3,
    )

    # 绘制当前解曲线
    plt.plot(
        iterations,
        current_objectives,
        label="Current Solution",
        color="blue",
        linewidth=1.5,
        alpha=0.7,
    )

    # 绘制候选解曲线
    if show_candidate:
        plt.plot(
            iterations,
            candidate_objectives,
            label="Candidate Solution",
            color="orange",
            linewidth=1,
            alpha=0.5,
            linestyle="--",
        )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Objective Value", fontsize=12)
    plt.title("ALNS Algorithm Convergence", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存至: {save_path}")

    plt.show()


def plot_acceptance_analysis(
    history: ALNSHistory,
    save_path: str | Path | None = None,
    window_size: int = 50,
    figsize: tuple[int, int] = (12, 8),
):
    """
    绘制接受率分析图

    Args:
        history: ALNS历史记录器实例
        save_path: 图片保存路径(可选)
        window_size: 滑动窗口大小用于计算移动平均接受率
        figsize: 图片尺寸
    """
    records = history.get_history()[1:]  # 排除初始解
    if not records:
        print("没有足够的数据进行绘图")
        return

    iterations = [r["iteration"] for r in records]
    accepted = [1 if r["accepted"] else 0 for r in records]

    # 计算滑动窗口接受率
    moving_acceptance = []
    for i in range(len(accepted)):
        start = max(0, i - window_size + 1)
        window = accepted[start : i + 1]
        moving_acceptance.append(sum(window) / len(window))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # 子图1: 接受/拒绝散点图
    colors = ["green" if a else "red" for a in accepted]
    ax1.scatter(iterations, accepted, c=colors, alpha=0.5, s=20)
    ax1.set_ylabel("Accepted (1) / Rejected (0)", fontsize=11)
    ax1.set_title("Solution Acceptance Pattern", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # 子图2: 移动平均接受率
    ax2.plot(iterations, moving_acceptance, color="blue", linewidth=2)
    ax2.axhline(
        y=history.get_acceptance_rate(),
        color="red",
        linestyle="--",
        label=f"Overall Rate: {history.get_acceptance_rate():.2%}",
    )
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel(f"Acceptance Rate (Window={window_size})", fontsize=11)
    ax2.set_title("Moving Average Acceptance Rate", fontsize=12, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存至: {save_path}")

    plt.show()


def plot_operator_usage(
    history: ALNSHistory,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 6),
):
    """
    绘制操作算子使用统计图

    Args:
        history: ALNS历史记录器实例
        save_path: 图片保存路径(可选)
        figsize: 图片尺寸
    """
    usage = history.get_operator_usage()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 破坏算子使用统计
    if usage["destroy"]:
        destroy_ops = list(usage["destroy"].keys())
        destroy_counts = list(usage["destroy"].values())
        ax1.bar(destroy_ops, destroy_counts, color="coral", alpha=0.7)
        ax1.set_title("Destroy Operator Usage", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Usage Count", fontsize=11)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3, axis="y")

    # 修复算子使用统计
    if usage["repair"]:
        repair_ops = list(usage["repair"].keys())
        repair_counts = list(usage["repair"].values())
        ax2.bar(repair_ops, repair_counts, color="skyblue", alpha=0.7)
        ax2.set_title("Repair Operator Usage", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Usage Count", fontsize=11)
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存至: {save_path}")

    plt.show()


def plot_all_analysis(
    history: ALNSHistory,
    save_dir: str | Path | None = None,
    prefix: str = "alns",
):
    """
    绘制所有分析图表

    Args:
        history: ALNS历史记录器实例
        save_dir: 图片保存目录(可选)
        prefix: 文件名前缀
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 收敛曲线
    conv_path = save_dir / f"{prefix}_convergence.png" if save_dir else None
    plot_alns_convergence(history, save_path=conv_path)

    # 接受率分析
    acc_path = save_dir / f"{prefix}_acceptance.png" if save_dir else None
    plot_acceptance_analysis(history, save_path=acc_path)

    # 算子使用统计
    op_path = save_dir / f"{prefix}_operators.png" if save_dir else None
    plot_operator_usage(history, save_path=op_path)

    print("\n所有分析图表已生成完毕!")


# 使用示例
if __name__ == "__main__":
    # 示例:如何使用历史记录器和可视化功能
    # from src.solver.alns import ALNS
    # from src.element import InstanceClass
    #
    # # 创建实例并求解
    # instance = InstanceClass(...)
    # alns = ALNS(instance)
    # solution = alns.solve()
    #
    # # 获取历史记录并绘图
    # history = alns.history
    #
    # # 绘制单个图表
    # plot_alns_convergence(history, save_path="convergence.png")
    #
    # # 或者绘制所有图表
    # plot_all_analysis(history, save_dir="results/plots", prefix="test_run")
    #
    # # 获取统计信息
    # print(f"总体接受率: {history.get_acceptance_rate():.2%}")
    # print(f"算子使用情况: {history.get_operator_usage()}")

    pass
