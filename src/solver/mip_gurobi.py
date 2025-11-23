from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from src.element import InstanceClass, Solution


@dataclass
class GurobiVariables:
    """Gurobi模型变量容器"""

    # 路径变量: 车辆k从节点i到节点j的访问路径
    x: gp.tupledict
    # 分组变量: 车辆k是否访问分组b
    y: gp.tupledict
    # 通信变量: 节点i的通信保障是否由车辆k在节点j处执行
    z: gp.tupledict
    # 任务执行变量: 节点i处任务是否被执行
    delta: gp.tupledict
    # 到达时间变量: 车辆k在节点i的访问时间
    arr_t: gp.tupledict
    # 最大返回时间变量: 所有车辆返回基地的最大时间
    tau: gp.Var

    def __str__(self):
        return (
            f"GurobiVariables(x={self.x}, z={self.z}, delta={self.delta}, "
            f"arr_t={self.arr_t}"
        )

    __repr__ = __str__


class GurobiMIPSolver:
    """Gurobi MIP求解器基类"""

    def __init__(
        self,
        solution: Solution,
        instance: InstanceClass,
    ):
        # 存入解对象、算例对象
        self.solution = solution
        self.instance = instance

        # 提取必要的集合和参数
        # ---------- 节点集合 ----------
        self.id_base = list(instance.base_ids)
        self.id_demand = list(instance.demand_ids)
        accessible_flag = instance.accessible
        self.id_isolated = [i for i in self.id_demand if not accessible_flag[i]]
        self.id_connected = [i for i in self.id_demand if accessible_flag[i]]

        self.id_departures = self.id_base + self.id_demand
        # 使用instance中定义的虚拟终点ID（total_node_num）
        self.dummy_end_id = instance.dummy_end_id
        self.id_arrivals = self.id_demand + [self.dummy_end_id]
        self.id_all_node = list(range(instance.total_node_num))
        self.id_all_node.append(self.dummy_end_id)  # 包含虚拟终点

        self.id_node_no_base = self.id_demand

        self.radius = instance.radius_array

        # ----------- 分组和分组信息 ----------
        self.group_id_list = instance.group_id
        self.group_sets = instance.group_sets
        self.id_group = sorted(self.group_sets.keys())

        # ----------- 最晚访问时间 ----------
        self.max_visit_time_list = instance.max_visit_time

        # ---------- 车辆集合 ----------
        self.id_Ground = instance.ground_veh_ids
        self.id_Drone = instance.drone_ids
        self.id_vehicle = list(range(instance.total_veh_num))
        self.capacity = list(instance.capacity_array)

        # ---------- 矩阵提取 ----------
        self.distance_matrix = instance.distance_matrix
        self.coverage_matrix = instance.comm_coverage_matrix
        self.travel_time_matrix = instance.travel_time_matrix

        # --------- 优先级得分 ----------
        self.node_score_list = instance.priority
        self.min_comm_time = instance.prob_config.instance_param.min_visit_time

        # --------- 求解器参数 ----------
        self.M = instance.prob_config.algorithm_config.big_m
        self.time_limit = instance.prob_config.algorithm_config.time_limit

        # 设置目标函数信息
        obj_config = instance.prob_config.algorithm_config.obj_config
        obj_names = obj_config.name
        obj_sense = obj_config.sense
        self.objective_info = list(zip(obj_names, obj_sense))

    def _create_model(self, model_name: str = "MIP_Optimizer"):
        """创建Gurobi模型"""
        self.model = gp.Model(model_name)

    # 创建Gurobi变量
    def _create_variables(self):
        # Vars: x_k_i,j: 车辆k的访问路径 (排除i=j的自环)
        # 虚拟终点ID现在是instance.dummy_end_id（等于total_node_num）
        x = self.model.addVars(
            (
                (k, i, j)
                for k in self.id_vehicle
                for i in self.id_departures
                for j in self.id_arrivals
                if i != j
            ),
            vtype=GRB.BINARY,
            name="x",
        )

        # Vars: y_k_b: 车辆k是否访问分组b
        y = self.model.addVars(
            self.id_Ground,
            self.id_group,
            vtype=GRB.BINARY,
            name="y",
        )
        # Vars: z_k_i,j: 节点i的通信保障是否由车辆k在节点j处执行
        # 注意: 保留i=j的情况,因为z[k,i,i]表示车辆k访问节点i作为通信枢纽
        z = self.model.addVars(
            self.id_Drone,
            self.id_demand,
            self.id_demand,
            vtype=GRB.BINARY,
            name="z",
        )

        # Vars: delta_k_i, 节点i处任务是否被编队k执行
        delta = self.model.addVars(
            self.id_vehicle,
            self.id_demand,
            name="delta",
            vtype=GRB.BINARY,
        )

        # Vars: a_k_i: 车辆k在i点的访问时间
        arr_t = self.model.addVars(
            self.id_vehicle,
            self.id_all_node,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="a",
        )

        # Vars: tau: 所有车辆返回基地的最大时间
        tau = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="tau",
        )

        self.variables = GurobiVariables(
            x=x,
            y=y,
            z=z,
            delta=delta,
            arr_t=arr_t,
            tau=tau,
        )

    def _get_priority_term(self):
        """计算优先级收益项: delta与优先级的乘积之和"""
        delta = self.variables.delta
        priority_term = gp.quicksum(
            self.node_score_list[i] * delta[k, i]
            for k in self.id_vehicle
            for i in self.id_demand
        )
        return priority_term

    def _get_duration_term(self):
        """计算路径成本项: x与弧线长度的乘积之和"""
        arr_t = self.variables.arr_t
        duration_term = gp.quicksum(
            arr_t[k, i] for k in self.id_vehicle for i in [self.dummy_end_id]
        )
        return duration_term

    def _set_objective(self):
        """设置目标函数: 最大化(优先级收益 - 路径成本)"""
        # 获取优先级收益项
        priority_term = self._get_priority_term()

        # # 获取路径成本项
        # duration_term = self._get_duration_term()
        # 获取makespan
        duration_term = self.variables.tau

        # 组合目标函数: 最大化(收益 - 1e-5 × 成本)
        obj_expr = priority_term - 1e-2 * duration_term

        self.model.setObjective(obj_expr, GRB.MAXIMIZE)

    def _add_constraints(self):
        """添加所有约束条件"""
        x = self.variables.x
        y = self.variables.y
        z = self.variables.z
        arr_t = self.variables.arr_t
        delta = self.variables.delta

        # -------------------------- 1. 流平衡约束  ------------------------------
        # Constrs 1-1 车辆出发数量约束
        self.model.addConstrs(
            (
                gp.quicksum(x[k, i, j] for i in self.id_base for j in self.id_demand)
                <= 1
                for k in self.id_vehicle
            ),
            name="base_departure_limit",
        )

        # Constrs 1-2 车辆返回约束
        self.model.addConstrs(
            (
                gp.quicksum(
                    x[k, i, j] for i in self.id_demand for j in [self.dummy_end_id]
                )
                == gp.quicksum(x[k, i, j] for i in self.id_base for j in self.id_demand)
                for k in self.id_vehicle
            ),
            name="base_return_limit",
        )

        # Constrs 1-3 需求点流平衡约束
        self.model.addConstrs(
            (
                gp.quicksum(x[k, i, j] for i in self.id_departures if i != j)
                == gp.quicksum(x[k, j, l] for l in self.id_arrivals if j != l)
                for k in self.id_vehicle
                for j in self.id_demand
            ),
            name="demand_flow_balance",
        )

        # -------------------------- 2. 地面搜救编队任务约束 ---------------------------
        # Constrs 2-1 不允许访问孤立节点
        self.model.addConstrs(
            (
                x[k, i, j] == 0
                for k in self.id_Ground
                for i in self.id_departures
                for j in self.id_isolated
                if i != j
            ),
            name="ground_no_isolated_visit",
        )

        # Constrs 2-2 地面车辆分组访问约束
        self.model.addConstrs(
            (
                gp.quicksum(
                    x[k, i, j]
                    for i in self.id_departures
                    for j in self.group_sets[b]
                    if i != j and i not in self.group_sets[b]
                )
                == y[k, b]
                for k in self.id_Ground
                for b in self.id_group
            ),
            name="ground_group_visit_limit",
        )

        # Constrs 2-3 分组访问数量限制
        self.model.addConstrs(
            (gp.quicksum(y[k, b] for k in self.id_Ground) <= 1 for b in self.id_group),
            name="ground_single_group_visit",
        )

        # Constrs 2-4 组内禁止访问约束
        self.model.addConstrs(
            (
                x[k, i, j] == 0
                for k in self.id_Ground
                for b in self.id_group
                for i in self.group_sets[b]
                for j in self.group_sets[b]
                if i != j
            ),
            name="ground_no_within_group_visit",
        )

        # Constrs 2-5 地面车辆任务执行标记
        self.model.addConstrs(
            (
                delta[k, i] == y[k, b]
                for k in self.id_Ground
                for b in self.id_group
                for i in self.group_sets[b]
            ),
            name="ground_delta_y_link",
        )

        # -------------------------- 3. 无人机编队任务约束 ---------------------------
        # Constrs 3-1 无人机任务执行与路径关联约束
        self.model.addConstrs(
            (
                z[k, j, j]
                == gp.quicksum(x[k, i, j] for i in self.id_departures if i != j)
                for k in self.id_Drone
                for j in self.id_demand
            ),
            name="drone_z_x_link",
        )

        # Constrs 3-2 无人机可视范围覆盖约束
        self.model.addConstrs(
            (
                z[k, i, j] <= self.coverage_matrix[k, i, j]
                for k in self.id_Drone
                for i in self.id_demand
                for j in self.id_demand
            ),
            name="drone_coverage_range",
        )

        # Constrs 3-3 无人机任务执行标记
        self.model.addConstrs(
            (
                delta[k, i] == gp.quicksum(z[k, i, j] for j in self.id_demand)
                for k in self.id_Drone
                for i in self.id_demand
            ),
            name="drone_delta_z_link",
        )

        # Constrs 3-4 无人机可视与直接访问之间的关联
        self.model.addConstrs(
            (
                z[k, i, j] <= z[k, j, j]
                for k in self.id_Drone
                for i in self.id_demand
                for j in self.id_demand
            ),
            name="drone_z_insight_permit_link",
        )

        # Constrs 3-5 无人机无重复访问约束
        self.model.addConstrs(
            (
                gp.quicksum(z[k, i, i] for k in self.id_Drone) <= 1
                for i in self.id_demand
            ),
            name="drone_single_visit_limit",
        )

        # -------------------------- 5. 任务执行次数约束 ------------------------------
        # Constrs 5-1 每个需求节点任务最多执行一次
        self.model.addConstrs(
            (
                gp.quicksum(delta[k, i] for k in self.id_vehicle) <= 1
                for i in self.id_demand
            ),
            name="single_task_execution_limit",
        )

        # -------------------------- 6. 访问时间约束 ------------------------------
        # Constrs 6-1 基地访问时刻初始化约束
        self.model.addConstrs(
            (arr_t[k, i] == 0 for k in self.id_vehicle for i in self.id_base),
            name="initial_arrival_time",
        )

        # Constrs 6-2 路径相邻节点时间连续性约束
        # 注意：虚拟终点已包含在扩展的travel_time_matrix中，其时间值为大M
        # 由于到虚拟终点的x变量存在，时间约束会自动处理，但因时间为大M会被约束排除
        self.model.addConstrs(
            (
                arr_t[k, j]
                >= arr_t[k, i]
                + self.travel_time_matrix[k, i, j]
                - 10 * self.M * (1 - x[k, i, j])
                for k in self.id_vehicle
                for i in self.id_base
                for j in self.id_arrivals
                if (k, i, j) in x
            ),
            name="departure_time_continuity",
        )

        self.model.addConstrs(
            (
                arr_t[k, j]
                >= arr_t[k, i]
                + self.min_comm_time
                + self.travel_time_matrix[k, i, j]
                - 10 * self.M * (1 - x[k, i, j])
                for k in self.id_vehicle
                for i in self.id_demand
                for j in self.id_arrivals
                if (k, i, j) in x
            ),
            name="time_continuity",
        )

        # Constrs 6-3 访问有效性约束
        self.model.addConstrs(
            (
                arr_t[k, i] <= self.max_visit_time_list[i]
                for k in self.id_vehicle
                for i in self.id_demand
            ),
            name="visit_time_limit",
        )

        # todo: 合理设置车辆路径容量，选取合适算例，注意单位合理性！
        # Constrs 6-4 路径容量限制
        self.model.addConstrs(
            (
                gp.quicksum(
                    self.distance_matrix[i][j] * x[k, i, j]
                    for i in self.id_departures
                    for j in self.id_arrivals
                    if (k, i, j) in x
                )
                <= self.capacity[k]
                for k in self.id_vehicle
            ),
            name="path_capacity_limit",
        )

        # Constrs 6-5 最大返回时间约束
        self.model.addConstrs(
            (
                arr_t[k, self.dummy_end_id] <= self.variables.tau
                for k in self.id_vehicle
            ),
            name="makespan_definition",
        )

    def _setup_basic_model(self):
        """设置基础模型，包括创建模型、变量和目标函数"""

        # todo： 初始化尚未检查
        # 创建Gurobi模型
        self._create_model()

        # 创建Gurobi变量
        self._create_variables()

        # 添加约束
        self._add_constraints()

        # 更新模型，应对Gurobi惰性更新方法
        self.model.update()

    def _extract_solution(self):
        """从Gurobi模型中提取解并存入Solution对象中"""
        # 定义求解状态映射
        status_map = {
            GRB.OPTIMAL: "最优解",
            GRB.INFEASIBLE: "无可行解",
            GRB.UNBOUNDED: "无界解",
            GRB.TIME_LIMIT: "时间限制，提前结束",
            GRB.INTERRUPTED: "中断",
            GRB.USER_OBJ_LIMIT: "用户目标限制",
        }
        # 获取求解状态
        try:
            self.solution.status = status_map.get(
                self.model.Status, f"未知状态 ({self.model.Status})"
            )
            self.solution.solve_time = self.model.Runtime
            self.solution.mip_gap = self.model.MIPGap
        except AttributeError as e:
            self.solution.status = "模型未运行或状态未知"
            raise e

        # 如果没有找到可行解,直接返回
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.USER_OBJ_LIMIT]:
            raise ValueError("模型无解，无法提取解信息。")

        # 检查是否有解
        sol_count = self.model.SolCount
        if sol_count == 0:
            print(f"当前可行解数量为{sol_count}, 返回空solution")
            return self.solution

        # 提取变量
        x = self.variables.x
        y = self.variables.y
        z = self.variables.z
        delta = self.variables.delta

        # 1. 构建路径 (routes) - 基于x变量
        # x[k,i,j]=1表示车辆k从节点i物理移动到节点j
        for k in self.id_vehicle:
            route = []

            # 找到起始节点(从基地出发)
            start_node = None
            for i in self.id_base:
                for j in self.id_demand:
                    if (k, i, j) in x and x[k, i, j].X > 0.5:
                        start_node = i
                        route.append(i)  # 添加基地作为起点
                        current_node = j
                        route.append(j)
                        break
                if start_node is not None:
                    break

            # 如果该车辆没有出发,跳过
            if start_node is None:
                continue

            # 按照路径顺序遍历
            visited = set(route)
            while True:
                found_next = False

                # 尝试找下一个需求节点
                for j in self.id_demand:
                    if (
                        j not in visited
                        and (k, current_node, j) in x
                        and x[k, current_node, j].X > 0.5
                    ):
                        route.append(j)
                        visited.add(j)
                        current_node = j
                        found_next = True
                        break

                # 如果没找到需求节点，检查是否返回虚拟终点
                if not found_next:
                    if (k, current_node, self.dummy_end_id) in x and x[
                        k, current_node, self.dummy_end_id
                    ].X > 0.5:
                        # 返回虚拟终点，实际上是返回起始基地，将起始基地加入路径末尾形成闭环
                        route.append(start_node)
                        break
                    else:
                        # 没有找到下一个节点，路径异常终止
                        break

            self.solution.routes[k] = route

        # 2. 构建服务覆盖关系 (coverage) - 基于delta、z、y变量
        # delta[k,i]=1表示车辆k为节点i提供服务
        # 需要根据车辆类型，结合不同变量确定服务位置：
        # - 无人机(Drone): 结合z变量，z[k,i,j]=1表示在节点j为节点i提供服务
        # - 地面车辆(Ground): 结合y变量，y[k,b]=1表示访问分组b，为该分组所有节点提供服务

        # 2.1 无人机的覆盖关系 (set coverage问题)
        for k in self.id_Drone:
            for j in self.id_demand:
                # 检查无人机k是否访问节点j作为服务点
                if (k, j, j) in z and z[k, j, j].X > 0.5:
                    covered_nodes = set()
                    # 找到所有由无人机k在节点j处提供服务的节点
                    for i in self.id_demand:
                        # delta[k,i]=1 且 z[k,i,j]=1: 车辆k为节点i提供服务，且在节点j处提供
                        if (k, i) in delta and delta[k, i].X > 0.5:
                            if (k, i, j) in z and z[k, i, j].X > 0.5:
                                covered_nodes.add(i)

                    # 设置覆盖关系（只在有服务节点时设置）
                    if covered_nodes:
                        self.solution.set_coverage(k, j, covered_nodes)

        # 2.2 地面车辆的覆盖关系 (定向越野问题)
        for k in self.id_Ground:
            # 遍历所有分组
            for b, group_nodes in self.group_sets.items():
                # 检查地面车辆k是否访问分组b
                if (k, b) in y and y[k, b].X > 0.5:
                    # 找到该分组中实际被服务的节点（基于delta）
                    served_nodes = set()
                    for i in group_nodes:
                        if (k, i) in delta and delta[k, i].X > 0.5:
                            served_nodes.add(i)

                    # 地面车辆在访问分组时，可能访问该分组的多个节点
                    # 这里简化处理：将分组中第一个被访问的节点作为服务点
                    # 更精确的做法是找到路径中属于该分组的节点
                    for visit_node in self.solution.routes[k]:
                        if visit_node in group_nodes and served_nodes:
                            self.solution.set_coverage(k, visit_node, served_nodes)
                            break

        # 3. 提取目标函数值
        try:
            # 获取目标函数值
            obj_expr = self.model.getObjective()
            obj_value = obj_expr.getValue()

            # 存储目标值
            self.solution.objectives = [obj_value]

        except (AttributeError, gp.GurobiError) as e:
            # 如果无法获取目标值,保持默认值
            raise ValueError(f"警告: 无法提取目标函数值: {e}")

        return self.solution

    def _set_model_params(self):
        """设置模型求解参数，主要包括
        1. 设置最大求解时间
        2. 关闭Gurobi原生输出
        """
        algorithm_config = self.instance.prob_config.algorithm_config

        time_limit = algorithm_config.time_limit

        # # 关闭gurobi输出
        # self.model.setParam("OutputFlag", 0)

        self.model.setParam("TimeLimit", time_limit)

        # # 设置数值容差参数
        self.model.setParam("FeasibilityTol", 1e-6)  # 可行性容差
        self.model.setParam("OptimalityTol", 1e-6)  # 最优性容差
        self.model.setParam("IntFeasTol", 1e-5)  # 整数可行性容差

        # 设置整数优先级
        self.model.setParam(GRB.Param.IntegralityFocus, 1)

    def mip_solve(self) -> Solution:
        """求解方法"""

        # 设置基础模型（初始化模型、添加变量、添加约束）
        self._setup_basic_model()

        # 设置目标函数(直接将二者加和)
        self._set_objective()

        # 设置求解器参数
        self._set_model_params()

        # 优化模型
        self.model.optimize()

        # 提取解信息
        self._extract_solution()

        # debug: 保存gurobi模型
        save_gurobi_model(
            self.model,
            "gurobi_models",
            base_name="eccp_model",
            demand_num=len(self.id_demand),
        )

        return self.solution


def gurobi_single_obj_solver(solution: Solution, instance: InstanceClass) -> Solution:
    """使用Gurobi求解单目标ECCP问题"""
    # 初始化mip求解器
    mip_solver = GurobiMIPSolver(solution, instance)
    # 执行求解
    solution = mip_solver.mip_solve()

    return solution


def save_gurobi_model(model, folder_path, base_name="model", demand_num=None):
    """
    保存Gurobi模型和非零变量到指定文件夹

    参数:
        model: Gurobi模型对象
        folder_path: 目标文件夹路径
        base_name: 文件基础名称，默认为"model"
    """
    # 仅使用pathlib进行路径拼接

    # 创建文件夹（如果不存在）
    save_folder = (
        Path(folder_path) / f"D{demand_num}" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_folder.mkdir(parents=True, exist_ok=True)

    # 定义便捷构造函数
    def p(name: str) -> Path:
        return save_folder / name

    # 1. 保存模型文件（多种格式）到时间戳子目录 save_folder
    model.write(str(p(f"{base_name}.lp")))  # LP格式 - 人类可读
    model.write(str(p(f"{base_name}.mps")))  # MPS格式 - 标准格式

    # 2. 保存求解结果（如果已求解）
    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        # 保存解（所有变量值）
        model.write(str(p(f"{base_name}.sol")))

        # 3. 单独保存非零变量
        with open(p(f"{base_name}_nonzero_vars.txt"), "w", encoding="utf-8") as f:
            f.write(f"模型名称: {model.ModelName}\n")
            f.write(f"目标值: {model.ObjVal}\n")
            f.write("=" * 60 + "\n")
            f.write("非零变量:\n")
            f.write("=" * 60 + "\n\n")

            # 获取所有变量
            vars_list = model.getVars()

            # 筛选非零变量
            nonzero_count = 0
            for var in vars_list:
                if abs(var.X) > 1e-6:  # 容差阈值
                    f.write(f"{var.VarName:30s} = {var.X:15.6f}\n")
                    nonzero_count += 1

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"非零变量总数: {nonzero_count} / {len(vars_list)}\n")

        print(f"✓ 模型和结果已保存到: {save_folder}")
        print(f"  - {base_name}.lp (模型LP格式)")
        print(f"  - {base_name}.mps (模型MPS格式)")
        print(f"  - {base_name}.sol (完整解)")
        print(f"  - {base_name}_nonzero_vars.txt (非零变量)")
    else:
        # 如果是无可行解，进行IIS（不可约不可行子系统）分析
        if model.Status == GRB.INFEASIBLE:
            try:
                print("⚠ 检测到模型无可行解，开始执行IIS分析……")
                model.computeIIS()  # 计算IIS
                iis_lp_path = p(f"{base_name}.ilp")
                model.write(str(iis_lp_path))  # 写出IIS文件（.ilp）

                # 写出详细IIS组成文本
                iis_txt_path = p(f"{base_name}_iis.txt")
                with open(iis_txt_path, "w", encoding="utf-8") as f:
                    f.write(f"模型名称: {model.ModelName}\n")
                    f.write("=" * 60 + "\n")
                    f.write("IIS 冲突信息 (不可约不可行子系统组成要素)\n")
                    f.write("=" * 60 + "\n\n")

                    # 约束
                    f.write("[约束]\n")
                    iis_constr_count = 0
                    for c in model.getConstrs():
                        if c.IISConstr:
                            f.write(f"  - {c.ConstrName}\n")
                            iis_constr_count += 1
                    f.write(f"约束数量: {iis_constr_count}\n\n")

                    # 变量下界/上界
                    f.write("[变量界限]\n")
                    iis_lb_count = 0
                    iis_ub_count = 0
                    for v in model.getVars():
                        if v.IISLB:
                            f.write(f"  - 下界参与: {v.VarName} >= {v.LB}\n")
                            iis_lb_count += 1
                        if v.IISUB:
                            f.write(f"  - 上界参与: {v.VarName} <= {v.UB}\n")
                            iis_ub_count += 1
                    f.write(
                        f"下界参与数量: {iis_lb_count}, 上界参与数量: {iis_ub_count}\n\n"
                    )

                    # SOS约束（若存在）
                    sos_list = model.getSOSs()
                    if sos_list:
                        f.write("[SOS约束]\n")
                        sos_count = 0
                        for sos in sos_list:
                            if sos.IISConstr:
                                f.write(f"  - SOS: {sos.getAttr('Type')} 名称未命名\n")
                                sos_count += 1
                        f.write(f"SOS约束数量: {sos_count}\n\n")

                    f.write("=" * 60 + "\n")
                    f.write("IIS分析完成。\n")

                print("⚠ 模型无可行解，已生成IIS文件和详细说明：")
                print(f"  - {base_name}.lp")
                print(f"  - {base_name}.mps")
                print(f"  - {base_name}.ilp (IIS文件)")
                print(f"  - {base_name}_iis.txt (IIS详细组成)")
            except gp.GurobiError as e:
                print(f"IIS分析失败: {e}")
                print("仅保存模型文件：")
                print(f"  - {base_name}.lp")
                print(f"  - {base_name}.mps")
        else:
            print("⚠ 模型未求解或非最优，已保存模型文件")
            print(f"  - {base_name}.lp")
            print(f"  - {base_name}.mps")

    return save_folder
