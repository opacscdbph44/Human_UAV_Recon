import numpy as np

from src.evaluator import Population


def plot_front(population: Population) -> None:
    """使用Matplotlib绘制种群的非支配前沿"""
    import matplotlib.pyplot as plt

    # 提取目标值
    objectives = np.array([sol.objectives for sol in population.solutions])
    if objectives.shape[1] != 2:
        raise ValueError("仅支持二维目标值的绘制")

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(objectives[:, 0], objectives[:, 1], c="blue", label="Solutions")

    # 标记非支配前沿
    fronts_idx = population.fast_non_dominated_sort_vectorized()
    for i, front_idx in enumerate(fronts_idx):
        front_objectives = np.array([objectives[sol_idx] for sol_idx in front_idx])
        plt.scatter(
            front_objectives[:, 0], front_objectives[:, 1], label=f"Front {i+1}"
        )

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Non-dominated Fronts")
    plt.legend()
    plt.grid()
    plt.show()
