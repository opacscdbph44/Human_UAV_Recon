import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.config import Config
from src.element import Solution
from src.solver.solver_main import Solver_Name
from src.utils.file_funcs import query_result_folder


def extract_latest_solution(
    prob_config: Config,
    solver_name: Solver_Name,
) -> Solution | None:
    """从result文件夹中提取指定算法和算例的最新解

    Args:
        prob_config (Config): 问题配置
        solver_name (str): 求解器名称

    Returns:
        Solution | None: 最新的解对象，如果没找到则返回None
    """
    result_folder = query_result_folder(prob_config, solver_name.value)

    # 查找所有 JSON 文件
    json_files = list(result_folder.glob("solution_result_*.json"))

    if len(json_files) == 0:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"未找到求解结果: {result_folder}"
        )
        return None

    # 从文件名中提取时间戳并排序,选择最新的文件
    def extract_timestamp(file_path: Path) -> datetime:
        try:
            name = file_path.stem  # 去掉.json后缀
            timestamp_str = name.replace("solution_result_", "")
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except Exception:
            return datetime.min

    latest_file = max(json_files, key=extract_timestamp)

    # 读取并解析解文件
    with open(latest_file, "r", encoding="utf-8") as f:
        solution_data = json.load(f)

    solution = Solution.from_dict(solution_data)

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"提取最新解: {solver_name}, D={prob_config.instance_param.demand_num}, "
        f"文件={latest_file.name}"
    )

    return solution


def compare_solvers(
    instance_name: str,
    demand_nums: List[int],
    solver_names: List[Solver_Name],
    prob_config_template: Config | None = None,
) -> None:
    """比较不同求解器在不同算例规模下的求解结果，并生成Excel报告

    Args:
        instance_name (str): 算例名称
        demand_nums (List[int]): 算例规模列表
        solver_names (List[str]): 求解器名称列表
        prob_config_template (Config | None): 配置模板，如果为None则使用默认配置
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: " f"开始比较求解器结果...")
    print(f"算例: {instance_name}")
    print(f"规模: {demand_nums}")
    print(f"求解器: {solver_names}")

    # 收集所有解的数据
    results = []

    for demand_num in demand_nums:
        for solver_name in solver_names:
            # 创建配置
            if prob_config_template is not None:
                from dataclasses import replace

                prob_config = replace(
                    prob_config_template,
                    instance_param=replace(
                        prob_config_template.instance_param,
                        name=instance_name,
                        demand_num=demand_num,
                    ),
                )
            else:
                from src.config import AlgorithmConfig, InstanceParam

                prob_config = Config(
                    instance_param=InstanceParam(
                        name=instance_name,
                        demand_num=demand_num,
                    ),
                    algorithm_config=AlgorithmConfig(),
                )

            # 提取最新解
            solution = extract_latest_solution(prob_config, solver_name)

            if solution is not None:
                # 提取关键信息
                result_row = {
                    "solver_name": solution.Solver_name,
                    "instance_name": solution.instance_name,
                    "instance_scale": solution.instance_scale,
                    "objectives": (
                        solution.objectives[0]
                        if len(solution.objectives) == 1
                        else tuple(solution.objectives)
                    ),
                    "solve_time": solution.solve_time,
                }

                # 如果是gurobi求解器，添加mip_gap
                solver_name_str = (
                    solver_name.value
                    if hasattr(solver_name, "value")
                    else str(solver_name)
                )
                if "gurobi" in solver_name_str.lower():
                    result_row["mip_gap"] = solution.mip_gap

                results.append(result_row)

    if not results:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"未找到任何求解结果，退出比较"
        )
        return

    # 创建DataFrame
    df_raw = pd.DataFrame(results)

    # 确保列的顺序
    base_columns = [
        "solver_name",
        "instance_name",
        "instance_scale",
        "objectives",
        "solve_time",
    ]
    if "mip_gap" in df_raw.columns:
        base_columns.append("mip_gap")

    df_raw = df_raw[base_columns]

    # 注意: df_raw保持原始数值类型,不需要格式化的DataFrame
    # Excel会根据单元格格式自动显示合适的精度

    # 为sheet2创建重整后的表格结构
    # 按算例名和算例规模分组，将不同求解器的结果展开到列
    sheet2_data = []

    # 获取唯一的算例规模
    unique_scales = df_raw["instance_scale"].unique()
    unique_scales = sorted(unique_scales)

    # 调试：打印所有求解器名称
    # print(f"DEBUG: DataFrame中的所有solver_name值: {df_raw['solver_name'].unique()}")
    # print(
    #     f"DEBUG: 传入的solver_names列表: {[s.value if hasattr(s, 'value') else str(s) for s in solver_names]}"
    # )

    for scale in unique_scales:
        # 筛选当前规模的所有数据
        scale_data = df_raw[df_raw["instance_scale"] == scale]

        if len(scale_data) == 0:
            continue

        # 创建新行
        row = {
            "算例名": scale_data.iloc[0]["instance_name"],
            "算例规模": f"D{scale}",
        }

        # 第一遍遍历：收集所有求解器的目标函数值
        solver_objectives = {}
        for solver_name in solver_names:
            # 将Solver_Name枚举转换为字符串进行比较
            solver_name_str = (
                solver_name.value if hasattr(solver_name, "value") else str(solver_name)
            )
            # print(f"DEBUG: 查找solver={solver_name_str}, scale={scale}")
            solver_data = scale_data[scale_data["solver_name"] == solver_name_str]
            if len(solver_data) == 0:
                raise ValueError(
                    f"未找到求解器 {solver_name_str} 在规模 {scale} 下的结果数据"
                )
            # print(f"DEBUG: 找到{len(solver_data)}条记录")

            if len(solver_data) > 0:
                solver_row = solver_data.iloc[0]
                # 保留数值类型,不转换为字符串
                obj_val = solver_row["objectives"]
                if isinstance(obj_val, (int, float)):
                    solver_objectives[solver_name_str] = obj_val
                else:
                    solver_objectives[solver_name_str] = None

        # 计算最优目标函数值（max问题，取最大值）
        valid_objectives = [v for v in solver_objectives.values() if v is not None]
        best_objective = max(valid_objectives) if valid_objectives else None
        row["最优目标函数值"] = best_objective

        # 第二遍遍历：为每个求解器添加列
        for solver_name in solver_names:
            # 将Solver_Name枚举转换为字符串进行比较
            solver_name_str = (
                solver_name.value if hasattr(solver_name, "value") else str(solver_name)
            )
            solver_data = scale_data[scale_data["solver_name"] == solver_name_str]

            if len(solver_data) > 0:
                solver_row = solver_data.iloc[0]
                # 保留数值类型,不转换为字符串
                obj_val = solver_row["objectives"]
                if isinstance(obj_val, (int, float)):
                    row[f"{solver_name_str}目标函数值"] = obj_val
                    # 计算与最优值的差距（最优值 - 当前值）
                    if best_objective is not None:
                        gap = best_objective - obj_val
                        row[f"{solver_name_str}与最优值差距"] = gap
                    else:
                        row[f"{solver_name_str}与最优值差距"] = None
                else:
                    row[f"{solver_name_str}目标函数值"] = str(obj_val)
                    row[f"{solver_name_str}与最优值差距"] = None

                # 保留数值类型
                row[f"{solver_name_str}求解时间"] = solver_row["solve_time"]

                # 如果有mip_gap,添加该列(保留数值类型)
                if "mip_gap" in solver_row and pd.notna(solver_row["mip_gap"]):
                    row[f"{solver_name_str}mipgap"] = solver_row["mip_gap"]
            else:
                # 如果没有该求解器的数据,填充None或NaN
                row[f"{solver_name_str}目标函数值"] = None
                row[f"{solver_name_str}与最优值差距"] = None
                row[f"{solver_name_str}求解时间"] = None

        sheet2_data.append(row)

    df_sheet2 = pd.DataFrame(sheet2_data)

    # 创建输出文件夹
    output_folder = Path("result_compare")
    output_folder.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"{current_time}.xlsx"

    # 写入Excel文件并设置数字格式
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="sheet1", index=False)
        df_sheet2.to_excel(writer, sheet_name="sheet2", index=False)

        # sheet1不设置格式，保留原始数据
        # 仅为sheet2设置格式
        ws2 = writer.sheets["sheet2"]
        for row in ws2.iter_rows(min_row=2, max_row=ws2.max_row):
            for cell in row:
                if cell.value is not None and isinstance(cell.value, (int, float)):
                    # 根据列名判断格式
                    header = ws2.cell(1, cell.column).value
                    if "目标函数值" in str(header) or "差距" in str(header):
                        cell.number_format = "0.0000"
                    elif "求解时间" in str(header) or "mipgap" in str(header):
                        cell.number_format = "0.00"

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"比较结果已保存到: {output_file}"
    )
    print(f"共比较了 {len(results)} 组求解结果")
