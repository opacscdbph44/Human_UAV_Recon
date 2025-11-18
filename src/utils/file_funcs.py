import json
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.element import InstanceClass, Solution
from src.evaluator import Evaluator, Population
from src.utils.ins_create import create_by_tsp


def load_instance(prob_config: Config, file_path: str = ""):
    """根据参数配置读取算例

    Args:
        prob_config (Config): 问题求解的所有参数配置，可以读取节点、车辆等信息
    """

    # [x] 按照路径存储规则，查询是否已经有成品算例
    if file_path == "":
        instance_folder = query_instance_folder(prob_config)
    else:
        instance_folder = Path(file_path)

    # 提取当前文件夹下的所有json文件
    json_files = list(instance_folder.glob("*.json"))

    # 初始化匹配文件列表
    matched_files = []

    # 从第一个开始，逐个读取并比较配置
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            saved_instance_dict = data["instance"]
            saved_config_dict = saved_instance_dict["prob_config"]

            # 将保存的配置字典转换为Config对象
            saved_config = Config.from_dict(saved_config_dict)

            # 比较配置是否一致
            if saved_config == prob_config:
                # 配置一致，加载算例实例对象并返回
                matched_files.append((json_file, saved_instance_dict))

    # 如果找到匹配文件，则根据读取数据返回算例实例对象
    if matched_files:
        # 如果找到多个匹配文件，则报警并选择第一个
        if len(matched_files) > 1:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"找到多个匹配的算例文件，共{len(matched_files)}个，"
                f"将使用第一个文件: {matched_files[0][0]}"
            )
        else:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"找到匹配的算例文件: {matched_files[0][0]}"
            )

        instance = InstanceClass.from_dict(matched_files[0][1])
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"成功建立算例对象: {instance}"
        )

        return instance

    # 否则，读取tsp文件，生成算例实例对象并返回
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:"
        "未找到匹配的成品算例文件，正在从原始tsp文件生成新算例..."
    )

    # 读取.tsp文件，生成算例对象
    instance = create_by_tsp(
        prob_config,
        instance_folder,
        prob_config.save_file,  # 根据要求，是否保存生成的新算例
    )

    return instance


def query_instance_folder(prob_config: Config) -> Path:
    """根据参数配置，查询成品算例存储文件夹路径

    Args:
        prob_config (Config): 问题求解的所有参数配置，可以读取节点、车辆等信息

    Returns:
        str: 成品算例存储文件夹路径
    """

    instance_name = prob_config.instance_param.name
    # 去掉结尾的".tsp"后缀
    if instance_name.endswith(".tsp"):
        instance_folder_name = "TSPLIB-" + instance_name[:-4]
    demand_num = prob_config.instance_param.demand_num
    steiner_num = prob_config.instance_param.steiner_num

    folder_path = Path(
        f"data/Instances/{instance_folder_name}/D{demand_num}_S{steiner_num}/"
    )

    # 检查路径是否存在,不存在则创建
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    return folder_path


def query_result_folder(
    prob_config: Config,
    solver_name: str,
) -> Path:
    """根据参数配置，查询结果存储文件夹路径

    Args:
        prob_config (Config): 问题求解的所有参数配置，可以读取节点、车辆等信息

    Returns:
        str: 结果存储文件夹路径
    """

    instance_name = prob_config.instance_param.name
    # 去掉结尾的".tsp"后缀
    if instance_name.endswith(".tsp"):
        instance_folder_name = "TSPLIB-" + instance_name[:-4]
    demand_num = prob_config.instance_param.demand_num
    steiner_num = prob_config.instance_param.steiner_num

    folder_path = Path(
        f"result/{instance_folder_name}/D{demand_num}_S{steiner_num}/{solver_name}/"
    )

    # 检查路径是否存在,不存在则创建
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    return folder_path


def save_solution(
    solution: Solution,
    folder_path: Path | None = None,
) -> None:
    current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if folder_path is None:
        folder_path = Path("result/") / current_date_time

    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    file_path = folder_path / f"solution_result_{current_date_time}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(solution.to_dict(), f, indent=2, ensure_ascii=False)

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:"
        f"成功保存单解结果到文件: {file_path}"
    )


def save_population_results(
    population: Population,
    folder_path: Path | None = None,
) -> None:
    """保存种群结果到指定文件

    Args:
        population (Population): 需要保存的种群对象
        folder_path (Path): 结果保存的文件夹路径
    """
    current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if folder_path is None:
        folder_path = Path("result/") / current_date_time

    # 创建文件夹(如果不存在)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    # 保存为 JSON 格式
    file_path = folder_path / f"population_results_{current_date_time}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        population_data = {
            "timestamp": current_date_time,
            "pop_size": population.size,
            "solver_name": population.solver_name,
            "solve_time": population.solve_time,
            "solutions": [sol.to_dict() for sol in population.solutions],
            "hv_history": population.hv_history,
        }
        json.dump(population_data, f, indent=2, ensure_ascii=False)

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
        f"成功保存种群结果到文件: {file_path}"
    )


def load_population_results(
    instance: InstanceClass,
    solver_name: str,
    file_path: Path,
) -> Population:
    """从指定文件夹加载种群结果

    Args:
        file_path (Path): 结果保存的文件夹路径

    Returns:
        Population: 加载的种群对象
    """
    if not file_path.exists():
        raise FileNotFoundError(f"指定的结果文件夹不存在: {file_path}")

    # 检查是否为文件夹
    if not file_path.is_dir():
        raise ValueError(f"指定的路径不是文件夹: {file_path}")

    # 查找所有 JSON 文件
    json_files = list(file_path.glob("population_results_*.json"))

    # 检查文件数量
    if len(json_files) == 0:
        raise FileNotFoundError(f"指定的文件夹中没有找到种群结果文件: {file_path}")
    elif len(json_files) > 1:
        # 打印所有候选文件
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"在文件夹中找到{len(json_files)}个种群结果文件:"
        )
        for json_file in json_files:
            print(f"  - {json_file.name}")

        # 从文件名中提取时间戳并排序,选择最新的文件
        # 文件名格式: population_results_20251024_021756.json
        def extract_timestamp(file_path: Path) -> datetime:
            # 仅在文件数量大于1时使用，用于从文件名提取时间戳
            try:
                # 提取文件名中的时间戳部分
                name = file_path.stem  # 去掉.json后缀
                timestamp_str = name.replace("population_results_", "")
                # 转换为datetime对象进行比较
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except Exception as e:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                    f"解析文件名时发生错误: {e}"
                )
                # 如果解析失败,返回最小时间
                return datetime.min

        result_file = max(json_files, key=extract_timestamp)

        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"将读取最新的文件: {result_file.name}"
        )
    else:
        result_file = json_files[0]
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"找到唯一的种群结果文件: {result_file.name}"
        )

    with open(result_file, "r", encoding="utf-8") as file_handle:
        population_data = json.load(file_handle)

    # 读取solutions字段
    solutions = [
        Solution.from_dict(sol_dict) for sol_dict in population_data["solutions"]
    ]

    pop_size = len(solutions)
    population = Population(
        size=pop_size,
        solver_name=solver_name,
        evaluator=Evaluator(instance),
    )
    population.solutions = solutions

    # 读取timestamp字段
    if "timestamp" not in population_data:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"[warning] 文件中没有找到 'timestamp' 字段"
        )

    # 读取pop_size字段
    if "pop_size" in population_data:
        saved_pop_size = population_data["pop_size"]
        if saved_pop_size != pop_size:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[warning] 保存的种群大小({saved_pop_size})与实际解数量({pop_size})不一致"
            )
    else:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"[warning] 文件中没有找到 'pop_size' 字段"
        )

    # 读取solver_name字段
    if "solver_name" in population_data:
        saved_solver_name = population_data["solver_name"]
        if saved_solver_name != solver_name:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"[warning] 保存的求解器名称({saved_solver_name})"
                f"与当前求解器({solver_name})不一致"
            )
        population.solver_name = saved_solver_name
    else:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"[warning] 文件中没有找到 'solver_name' 字段"
        )

    # 读取solve_time字段
    if "solve_time" in population_data:
        population.solve_time = population_data["solve_time"]
    else:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"[warning] 文件中没有找到 'solve_time' 字段"
        )

    # 读取hv_history字段
    if "hv_history" in population_data:
        population.hv_history = population_data["hv_history"]
    else:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"[warning] 文件中没有找到 'hv_history' 字段"
        )

    return population
